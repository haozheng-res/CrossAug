"""CrossAug LTM output cache."""

import os
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps

import numpy as np
import torch
import torch.distributed as dist


CACHE_SPLITS = ('train', 'val')


class LTMOutputCacheError(RuntimeError):
    """The LTM output cache is unavailable or contains invalid data."""


def ltm_cache_session(train_method):
    """Run the original ``train()`` in a cache session; equivalent to a direct call when caching is disabled."""
    @wraps(train_method)
    def run_in_cache_session(experiment, *args, **kwargs):
        with experiment.ltm_cache.training_session(experiment.model):
            return train_method(experiment, *args, **kwargs)
    return run_in_cache_session


class CrossAugLTMCache:
    """Integrate caching transparently with DataLoader and LTMWrapper without modifying the training loop."""

    def __init__(self, args, cache_root=None):
        self.enabled = args.cache_ltm_outputs
        self.pred_len = args.pred_len
        self.cache_root = cache_root
        self._storage = None
        self._batch = None
        self._online_forward = None

    def wrap_loader(self, data_set, data_loader, flag):
        """Wrap train/val loaders when caching is enabled; return the original loader for test or when caching is disabled."""
        if not self.enabled or flag not in CACHE_SPLITS:
            return data_loader
        return _CacheAwareLoader(
            self, data_loader, flag, dataset_size=len(data_set))

    def _start_pass(self, split, dataset_size, expected_rows):
        if self._storage.is_ready(split):
            return 'read'
        self._storage.begin_write(split, dataset_size, expected_rows)
        return 'write'

    def _finish_pass(self, split, mode, completed):
        if completed and mode == 'write':
            self._storage.finalize(split)

    def _forward_ltm(self, batch_x):
        """Run online inference and write on the first pass; read by the current batch indices on subsequent passes."""
        if self._batch is None:
            return self._online_forward(batch_x)
        split, mode, indices = self._batch
        if mode == 'read':
            return self._storage.read(split, indices, device=batch_x.device)
        if mode == 'write':
            prediction = self._online_forward(batch_x)
            self._storage.append(split, indices, prediction)
            return prediction
        raise ValueError('Unknown LTM cache mode: {!r}.'.format(mode))

    @contextmanager
    def training_session(self, model):
        """Proxy the LTM forward pass during training; restore the model and clean up this run's cache on exit."""
        if not self.enabled:
            yield
            return

        self._storage = LTMOutputCache(self.pred_len, self.cache_root)
        crossaug = model.module if hasattr(model, 'module') else model
        ltm_wrapper = crossaug.ltm_wrapper
        self._online_forward = ltm_wrapper.forward
        ltm_wrapper.forward = self._forward_ltm
        completed = False
        try:
            yield
            completed = True
        finally:
            ltm_wrapper.forward = self._online_forward
            self._online_forward = self._batch = None
            self._cleanup(completed)

    def _cleanup(self, completed):
        try:
            self._storage.cleanup(synchronize=completed)
        except Exception as error:
            if completed:
                raise
            print('Warning: failed to clean current LTM cache run: {}'.format(error))
        finally:
            self._storage = None


class _CacheAwareLoader:
    """Hide sample_index and register the current batch with the cache controller."""

    def __init__(self, cache, data_loader, split, dataset_size):
        self.cache = cache
        self.data_loader = data_loader
        self.split = split
        self.dataset_size = dataset_size

    def __len__(self):
        return len(self.data_loader)

    def __getattr__(self, name):
        return getattr(self.data_loader, name)

    def __iter__(self):
        mode = self.cache._start_pass(
            self.split, self.dataset_size, len(self.data_loader.sampler))
        completed = False
        try:
            for batch in self.data_loader:
                if len(batch) != 3:
                    raise LTMOutputCacheError(
                        '{} cache batch must contain sample_index.'.format(self.split))
                self.cache._batch = (self.split, mode, batch[-1])
                try:
                    yield batch[:2]
                finally:
                    self.cache._batch = None
            completed = True
        finally:
            self.cache._batch = None
            self.cache._finish_pass(self.split, mode, completed)


# ============================== Underlying shard storage ==============================

def _shard_dtype(pred_len):
    return np.dtype([
        ('index', np.int64),
        ('prediction', np.float32, (int(pred_len), 1)),
    ])


def _as_indices(sample_indices):
    if isinstance(sample_indices, torch.Tensor):
        sample_indices = sample_indices.detach().cpu().numpy()
    return np.asarray(sample_indices, dtype=np.int64).reshape(-1)


def _close_memmap(array):
    mmap = getattr(array, '_mmap', None)
    if mmap is not None:
        mmap.close()


@dataclass
class _Writer:
    dataset_size: int
    expected_rows: int
    shard: np.memmap
    cursor: int = 0


class LTMOutputCache:
    """Manage train/val FP32 mmap shards for a single run."""

    def __init__(self, pred_len, cache_root=None):
        self.pred_len = int(pred_len)
        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cache_root = os.path.abspath(
            cache_root or os.path.join(project_root, '_cache', 'ltm_outputs'))
        self.run_id = self._create_run_id()
        self.run_dir = os.path.join(self.cache_root, self.run_id)
        self._writers, self._readers = {}, {}
        if self.rank == 0:
            print('[LTM Cache] Created task cache: {}'.format(self.run_dir))

    def _create_run_id(self):
        run_id = None
        if self.rank == 0:
            os.makedirs(self.cache_root, exist_ok=True)
            run_id = '{}-{}-{}'.format(
                time.strftime('%Y%m%d-%H%M%S'), os.getpid(), uuid.uuid4().hex[:10])
            os.makedirs(os.path.join(self.cache_root, run_id), exist_ok=False)
        if self.distributed:
            payload = [run_id]
            dist.broadcast_object_list(payload, src=0)
            run_id = payload[0]
        return run_id

    def _barrier(self):
        if self.distributed:
            dist.barrier()

    def _split_dir(self, split):
        if split not in CACHE_SPLITS:
            raise ValueError(
                "LTM cache split must be 'train' or 'val', got {!r}.".format(split))
        return os.path.join(self.run_dir, split)

    @staticmethod
    def _shard_path(split_dir, rank):
        return os.path.join(split_dir, 'rank{}.npy'.format(rank))

    def is_ready(self, split):
        return os.path.isfile(os.path.join(self._split_dir(split), 'READY'))

    def begin_write(self, split, dataset_size, expected_rows):
        """Preallocate a shard storing both indices and predictions for the current rank."""
        if split in self._writers:
            raise LTMOutputCacheError('{} cache writer is already open.'.format(split))
        if self.is_ready(split):
            raise LTMOutputCacheError('{} cache is already READY.'.format(split))
        dataset_size, expected_rows = int(dataset_size), int(expected_rows)
        if dataset_size <= 0 or expected_rows <= 0:
            raise ValueError('dataset_size and expected_rows must both be positive.')

        split_dir = self._split_dir(split)
        if self.rank == 0:
            os.makedirs(split_dir, exist_ok=False)
        self._barrier()
        shard = np.lib.format.open_memmap(
            self._shard_path(split_dir, self.rank), mode='w+',
            dtype=_shard_dtype(self.pred_len), shape=(expected_rows,))
        self._writers[split] = _Writer(dataset_size, expected_rows, shard)
        if self.rank == 0:
            print('[LTM Cache] Building {} cache during its first traversal.'.format(split))

    def append(self, split, sample_indices, prediction):
        """Append a batch, always storing original-scale FP32 predictions of shape ``[B,H,1]``."""
        if split not in self._writers:
            raise LTMOutputCacheError('{} cache writer is not open.'.format(split))
        if not isinstance(prediction, torch.Tensor):
            raise TypeError('pred_ltm must be a torch.Tensor.')
        expected_shape = (prediction.shape[0], self.pred_len, 1)
        if tuple(prediction.shape) != expected_shape:
            raise LTMOutputCacheError(
                'pred_ltm shape mismatch: expected {}, got {}.'.format(
                    expected_shape, tuple(prediction.shape)))
        if not torch.isfinite(prediction).all():
            raise LTMOutputCacheError('pred_ltm contains NaN or Inf.')

        indices = _as_indices(sample_indices)
        if len(indices) != len(prediction):
            raise LTMOutputCacheError(
                'sample index count {} does not match batch size {}.'.format(
                    len(indices), len(prediction)))
        writer = self._writers[split]
        start, end = writer.cursor, writer.cursor + len(indices)
        if end > writer.expected_rows:
            raise LTMOutputCacheError(
                '{} cache received more rows than expected: {} > {}.'.format(
                    split, end, writer.expected_rows))
        writer.shard['index'][start:end] = indices
        writer.shard['prediction'][start:end] = prediction.detach().float().cpu().numpy()
        writer.cursor = end

    def finalize(self, split):
        """Close the shard, then have rank 0 validate shards and build the global lookup table."""
        if split not in self._writers:
            raise LTMOutputCacheError('{} cache writer is not open.'.format(split))
        writer = self._writers.pop(split)
        writer.shard.flush()
        _close_memmap(writer.shard)
        if writer.cursor != writer.expected_rows:
            raise LTMOutputCacheError(
                '{} cache is incomplete on rank {}: wrote {}, expected {}.'.format(
                    split, self.rank, writer.cursor, writer.expected_rows))

        self._barrier()
        error = None
        if self.rank == 0:
            try:
                self.build_split_index(
                    self._split_dir(split), split, writer.dataset_size,
                    self.pred_len, self.world_size)
            except Exception as exc:
                error = '{}: {}'.format(type(exc).__name__, exc)
        if self.distributed:
            payload = [error]
            dist.broadcast_object_list(payload, src=0)
            error = payload[0]
        if error is not None:
            raise LTMOutputCacheError(
                'Failed to finalize {} LTM cache: {}'.format(split, error))
        self.open_read(split)
        if self.rank == 0:
            print('[LTM Cache] {} cache READY; subsequent traversals bypass the LTM.'.format(split))

    @classmethod
    def build_split_index(cls, split_dir, split, dataset_size, pred_len, world_size):
        """Validate all shards and build the ``sample_index -> (rank,row)`` lookup table."""
        dataset_size, pred_len, world_size = map(
            int, (dataset_size, pred_len, world_size))
        locator = np.full((dataset_size, 2), -1, dtype=np.int64)

        for rank in range(world_size):
            path = cls._shard_path(split_dir, rank)
            if not os.path.isfile(path):
                raise LTMOutputCacheError('Missing shard file for rank {}.'.format(rank))
            shard = np.load(path, mmap_mode='r', allow_pickle=False)
            try:
                cls._validate_shard(shard, rank, dataset_size, pred_len)
                indices = shard['index']
                unset = locator[indices, 0] == -1
                new_indices = np.asarray(indices[unset], dtype=np.int64)
                locator[new_indices, 0] = rank
                locator[new_indices, 1] = np.flatnonzero(unset)
            finally:
                _close_memmap(shard)

        missing = np.flatnonzero(locator[:, 0] < 0)
        if missing.size:
            raise LTMOutputCacheError(
                'Cache does not cover all {} samples; missing {} indices, e.g. {}.'.format(
                    dataset_size, missing.size, missing[:10].tolist()))
        np.save(os.path.join(split_dir, 'locator.npy'), locator)
        with open(os.path.join(split_dir, 'READY'), 'w', encoding='utf-8') as file:
            file.write('ready\n')

    @staticmethod
    def _validate_shard(shard, rank, dataset_size, pred_len):
        if shard.ndim != 1 or shard.dtype != _shard_dtype(pred_len):
            raise LTMOutputCacheError(
                'Rank {} shard shape/dtype must contain index and FP32 prediction fields.'.format(rank))
        indices = shard['index']
        if np.any(indices < 0) or np.any(indices >= dataset_size):
            raise LTMOutputCacheError(
                'Rank {} shard contains out-of-range sample indices.'.format(rank))
        predictions = shard['prediction']
        for start in range(0, len(shard), 1024):
            if not np.isfinite(predictions[start:start + 1024]).all():
                raise LTMOutputCacheError(
                    'Rank {} prediction shard contains NaN or Inf.'.format(rank))

    def open_read(self, split):
        """Open the global lookup table and read-only shards from all ranks."""
        if split in self._readers:
            return
        split_dir = self._split_dir(split)
        if not self.is_ready(split):
            raise LTMOutputCacheError('{} cache is not READY.'.format(split))

        locator, shards = None, []
        try:
            locator = np.load(
                os.path.join(split_dir, 'locator.npy'), mmap_mode='r', allow_pickle=False)
            if locator.ndim != 2 or locator.shape[1] != 2 or locator.dtype != np.int64:
                raise LTMOutputCacheError('{} locator is invalid.'.format(split))
            shards = [
                np.load(self._shard_path(split_dir, rank), mmap_mode='r', allow_pickle=False)
                for rank in range(self.world_size)
            ]
            for rank, shard in enumerate(shards):
                self._validate_shard(shard, rank, len(locator), self.pred_len)
        except Exception as exc:
            _close_memmap(locator)
            for shard in shards:
                _close_memmap(shard)
            if isinstance(exc, LTMOutputCacheError):
                raise
            raise LTMOutputCacheError(
                'Failed to open READY {} cache: {}'.format(split, exc)) from exc
        self._readers[split] = (locator, shards)

    def read(self, split, sample_indices, device):
        """Read FP32 LTM outputs by stable global indices and transfer them to the target device."""
        if split not in self._readers:
            self.open_read(split)
        locator, shards = self._readers[split]
        indices = _as_indices(sample_indices)
        if np.any(indices < 0) or np.any(indices >= len(locator)):
            raise LTMOutputCacheError('{} cache lookup contains out-of-range indices.'.format(split))

        ranks, rows = locator[indices].T
        if np.any(ranks < 0) or np.any(rows < 0):
            raise LTMOutputCacheError('{} READY cache contains missing indices.'.format(split))
        output = np.empty((len(indices), self.pred_len, 1), dtype=np.float32)
        for rank in np.unique(ranks):
            mask = ranks == rank
            shard = shards[int(rank)]
            if np.any(rows[mask] >= len(shard)):
                raise LTMOutputCacheError('{} locator contains an invalid shard row.'.format(split))
            output[mask] = shard['prediction'][rows[mask]]
        if not np.isfinite(output).all():
            raise LTMOutputCacheError('{} READY cache returned NaN or Inf.'.format(split))
        return torch.from_numpy(output).to(
            device=device, dtype=torch.float32, non_blocking=True)

    def close(self):
        for locator, shards in self._readers.values():
            _close_memmap(locator)
            for shard in shards:
                _close_memmap(shard)
        for writer in self._writers.values():
            _close_memmap(writer.shard)
        self._readers.clear()
        self._writers.clear()

    def cleanup(self, synchronize=True):
        """Delete only the current run_id subdirectory and always retain the cache root directory."""
        self.close()
        if synchronize:
            self._barrier()
        if self.rank == 0:
            shutil.rmtree(self.run_dir, ignore_errors=True)
            os.makedirs(self.cache_root, exist_ok=True)
            print('[LTM Cache] Removed task cache {}; retained root {}.'.format(
                self.run_id, self.cache_root))
        if synchronize:
            self._barrier()
