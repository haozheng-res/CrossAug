import os
import shutil
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.ltm_output_cache import CrossAugLTMCache, ltm_cache_session
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, build_lr_scheduler, step_batch_lr_scheduler

warnings.filterwarnings('ignore')


class WeightedLoss:
    def __init__(self, alpha, loss_mode='L2'):
        self.alpha = alpha
        self.loss_mode = loss_mode
        if self.loss_mode == 'L1':
            self.loss_fun = nn.L1Loss(reduction='none')
        elif self.loss_mode == 'L2':
            self.loss_fun = nn.MSELoss(reduction='none')
        elif self.loss_mode == 'L1L2':
            self.loss_fun1 = nn.L1Loss(reduction='none')
            self.loss_fun2 = nn.MSELoss(reduction='none')

    def __call__(self, pred, gt):
        # [b,l,n]
        L = pred.shape[1]
        weights = torch.tensor([(i + 1) ** (-self.alpha) for i in range(L)], device=pred.device)  # With alpha=0, weights do not decay over the forecast horizon; larger values reduce weights at later time steps.
        weights = weights / weights.sum() * L  # Normalize weights to sum to L, keeping the loss scale consistent with standard MSE.
        weights = weights.unsqueeze(dim=0).unsqueeze(dim=-1)  # [1, L, 1]
        if self.loss_mode in ['L1', 'L2']:
            loss_vec = self.loss_fun(pred, gt)
            weightedLoss = torch.mean(loss_vec * weights)
        elif self.loss_mode == 'L1L2':
            loss_vec = self.loss_fun1(pred, gt)
            loss_vec2 = self.loss_fun2(pred, gt)
            weightedLoss = torch.mean(loss_vec * weights + loss_vec2 * weights)
        else:
            raise NotImplementedError
        return weightedLoss


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.ltm_cache = CrossAugLTMCache(args)

    def _build_model(self):
        model = self.model_dict['CrossAug'].Model(self.args).float()
        if self.args.use_dp:
            if len(self.args.device_ids) > 1:
                model = nn.DataParallel(model, device_ids=self.args.device_ids)
            return model
        return DDP(model.to(self.device), device_ids=[self.args.local_rank], find_unused_parameters=True)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, self.ltm_cache.wrap_loader(data_set, data_loader, flag)

    def vali(self, data_loader, criterion):
        """Validate using only inputs and future labels; the model uses no time markers or decoder inputs."""
        total_loss = []
        loss_value = torch.zeros((), device=self.device)
        loss_count = torch.zeros((), device=self.device)
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.float().to(self.device)  # [B, seq_len, N]
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    outputs = self.model(batch_x)  # [B, pred_len, 1]
                loss = criterion(outputs.detach().cpu(), batch_y.float()).item()
                total_loss.append(loss)
                if not self.args.use_dp:
                    loss_value += loss * batch_x.size(0)
                    loss_count += batch_x.size(0)
        if self.args.use_dp:
            total_loss = np.average(total_loss)
        else:
            dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
            total_loss = (loss_value / loss_count).item()
        self.model.train()
        return total_loss

    @ltm_cache_session
    def train(self, setting):
        _, train_loader = self._get_data('train')
        _, vali_loader = self._get_data('val')
        test_loader = None
        if not self.args.disable_test_every_epoch:
            _, test_loader = self._get_data('test')

        path = os.path.join(self.args.checkpoints, setting)
        if int(os.environ.get('RANK', '0')) == 0:
            os.makedirs(path, exist_ok=True)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, args=self.args)
        model_optim = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        criterion = WeightedLoss(self.args.loss_alpha, loss_mode=self.args.loss_mode)
        train_steps = len(train_loader)
        scheduler = build_lr_scheduler(model_optim, train_steps, self.args)
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))

        for epoch in range(self.args.train_epochs):
            if not self.args.use_dp:
                train_loader.sampler.set_epoch(epoch)
            train_loss = []
            loss_value = torch.zeros((), device=self.device)
            loss_count = torch.zeros((), device=self.device)
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                if epoch == 0 and i == 0:
                    print(f'batch_x.shape: {batch_x.shape}, batch_y.shape: {batch_y.shape}')
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)  # [B, seq_len, N]
                batch_y = batch_y.float().to(self.device)  # [B, pred_len, 1]
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    outputs = self.model(batch_x)  # [B, pred_len, 1]
                    loss = criterion(outputs, batch_y)
                if self.args.use_dp:
                    train_loss.append(loss.item())
                else:
                    loss_value += loss.detach() * batch_x.size(0)
                    loss_count += batch_x.size(0)

                scaler.scale(loss).backward()
                if self.args.max_norm > 0:
                    scaler.unscale_(model_optim)
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        self.args.max_norm,
                    )
                scaler.step(model_optim)
                scaler.update()
                step_batch_lr_scheduler(scheduler, self.args)

            elapsed = time.time() - epoch_time
            if self.args.use_dp:
                train_loss = np.average(train_loss)
            else:
                dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
                train_loss = (loss_value / loss_count).item()
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion) if test_loader is not None else None
            test_loss_text = 'None' if test_loss is None else f'{test_loss:.7f}'
            print(f'\nEpoch: {epoch + 1}, Steps: {train_steps} | '
                  f'Time: {int(elapsed // 60)}min {elapsed % 60:.1f}s | '
                  f'Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss_text}')
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print('Early stopping')
                break
            if self.args.lradj not in ('onecycle', 'cosine_batch'):
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        self.load_checkpoint(os.path.join(path, 'checkpoint.pth'))
        if int(os.environ.get('RANK', '0')) == 0:
            shutil.rmtree(path)
        return self.model

    def load_checkpoint(self, path):
        """Load full CrossAug weights with support for single-GPU, DP, and DDP parameter prefixes."""
        if not self.args.use_dp:
            dist.barrier()
        checkpoint = torch.load(path, map_location=self.device)
        checkpoint = {key.removeprefix('module.'): value for key, value in checkpoint.items()}
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.load_state_dict(checkpoint)
        if not self.args.use_dp:
            dist.barrier()

    def _run_prediction(self, test_loader):
        """Store predictions on the CPU and gather DDP chunks to rank 0; shape: [N, pred_len, 1]."""
        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.float().to(self.device)
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    outputs = self.model(batch_x)
                preds.append(outputs.detach().cpu().contiguous())
                trues.append(batch_y.float().contiguous())

        local_preds = torch.cat(preds, dim=0).contiguous()
        local_trues = torch.cat(trues, dim=0).contiguous()
        preds.clear()
        trues.clear()
        if self.args.use_dp:
            preds_array, trues_array = local_preds, local_trues
        else:
            preds_array = self.ddp_gather_tensor_varlen_chunked_to_rank0_cpu(local_preds, self.device)
            del local_preds
            trues_array = self.ddp_gather_tensor_varlen_chunked_to_rank0_cpu(local_trues, self.device)
            del local_trues
            if dist.get_rank() != 0:
                return None, None
        preds, trues = preds_array.numpy(), trues_array.numpy()
        print('test shape:', preds.shape, trues.shape)
        return preds, trues

    def test(self, setting=None):
        _, test_loader = self._get_data(flag='test')
        preds, trues = self._run_prediction(test_loader)

        # The new DDP flow saves and evaluates all predictions only on rank 0; other ranks have finished communication and can return.
        if not self.args.use_dp and dist.get_rank() != 0:
            return

        mae, mse, rmse, r2, mape, smape, nmae, mase, mspe = metric(preds, trues)
        print(f'mse: {mse:.5f}, mae: {mae:.5f}, rmse: {rmse:.5f}, r2: {r2:.5f}, '
              f'mape: {mape:.3f}, smape: {smape:.3f}, nmae: {nmae:.5f}, mase: {mase:.5f}, mspe: {mspe:.5f}')
        horizon_metrics_str = ''
        eval_horizons = [96, 192, 336, 720]
        pred_len = preds.shape[1]
        eval_horizons = [h for h in eval_horizons if h <= pred_len]
        for h in eval_horizons:
            preds_horizon, trues_horizon = preds[:, h-1:h, -1:], trues[:, h-1:h, -1:]
            preds_cumulative, trues_cumulative = preds[:, :h, -1:], trues[:, :h, -1:]
            mae_h, mse_h, rmse_h, r2_h, mape_h, smape_h, nmae_h, mase_h, mspe_h = metric(preds_horizon, trues_horizon)
            mae_c, mse_c, rmse_c, r2_c, mape_c, smape_c, nmae_c, mase_c, mspe_c = metric(preds_cumulative, trues_cumulative)
            if self.args.test_step_detail:
                line = (f'\n[{h}h] mse: {mse_h:.5f}, mse_c: {mse_c:.5f}, mae: {mae_h:.5f}, mae_c: {mae_c:.5f}, '
                        f'rmse: {rmse_h:.5f}, rmse_c: {rmse_c:.5f}, r2: {r2_h:.5f}, r2_c: {r2_c:.5f}, '
                        f'smape: {smape_h:.3f}, smape_c: {smape_c:.3f}, nmae: {nmae_h:.5f}, nmae_c: {nmae_c:.5f}')
                print(line)
            else:
                line = (f'[{h}h] r2: {r2_h:.5f}, r2_c: {r2_c:.5f} ')
                print(line)
            horizon_metrics_str += line
        if not self.args.test_step_detail:
            horizon_metrics_str = '\n' + horizon_metrics_str

        if int(os.environ.get("RANK", "0")) == 0:
            f = open("result_long_term_forecast.txt", 'a')
            f.write(setting + "  \n")
            f.write(f'mse: {mse:.5f}, mae: {mae:.5f}, rmse: {rmse:.5f}, r2: {r2:.5f}, '
                    f'mape: {mape:.3f}, smape: {smape:.3f}, nmae: {nmae:.5f}, mase: {mase:.5f}, mspe: {mspe:.5f}')
            if horizon_metrics_str:
                f.write(horizon_metrics_str)

            f.write('\n')
            f.write('\n')
            f.close()

        return

    def ddp_gather_tensor_varlen_chunked_to_rank0_cpu(
            self, local_tensor, device, chunk_size=8):
        """
        Gather variable-length tensors from each rank in chunks onto rank 0's CPU to keep full test results off the GPU.

        local_tensor: CPU Tensor, shape [N_local, L, C]
        return: By default, only rank 0 returns a CPU Tensor [sum(N_local), L, C]; other ranks return None.

        Results are written strictly in rank-major order, matching the original ddp_gather_tensor_varlen:
        [all valid samples from rank 0, all valid samples from rank 1, ...]. Duplicate samples
        introduced by DistributedSampler to align ranks are preserved, so metric inputs are identical to those of the original implementation.
        """
        assert isinstance(local_tensor, torch.Tensor), \
            f"ddp_gather_tensor_varlen_chunked_to_rank0_cpu expects torch.Tensor, but got {type(local_tensor)}"
        assert local_tensor.device.type == 'cpu', \
            f"local_tensor must stay on CPU, but got device={local_tensor.device}"
        assert chunk_size > 0, f"chunk_size must be positive, but got {chunk_size}"

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Only sample-count metadata stays on the GPU, with negligible overhead.
        local_n = torch.tensor([local_tensor.shape[0]], device=device, dtype=torch.long)
        n_list_gpu = [torch.zeros_like(local_n) for _ in range(world_size)]
        dist.all_gather(n_list_gpu, local_n)
        n_list = [int(n.item()) for n in n_list_gpu]
        max_n = max(n_list)

        # Rank 0 preallocates the final CPU tensor to avoid an extra full copy from list concatenation.
        collect_result = rank == 0
        if collect_result:
            global_shape = [sum(n_list)] + list(local_tensor.shape[1:])
            global_tensor = torch.empty(global_shape, dtype=local_tensor.dtype, device='cpu')
            rank_offsets = []
            offset = 0
            for n in n_list:
                rank_offsets.append(offset)
                offset += n
        else:
            global_tensor = None
            rank_offsets = None

        for start in range(0, max_n, chunk_size):
            current_chunk_size = min(chunk_size, max_n - start)
            local_valid_n = max(0, min(local_tensor.shape[0] - start, current_chunk_size))

            # Move only one small chunk to the GPU at a time; ranks with fewer than current_chunk_size samples use zero padding.
            if local_valid_n > 0:
                local_chunk = local_tensor[start:start + local_valid_n].to(device).contiguous()
            else:
                empty_shape = [0] + list(local_tensor.shape[1:])
                local_chunk = torch.empty(empty_shape, dtype=local_tensor.dtype, device=device)

            if local_valid_n < current_chunk_size:
                pad_shape = [current_chunk_size - local_valid_n] + list(local_tensor.shape[1:])
                pad_tensor = torch.zeros(pad_shape, dtype=local_tensor.dtype, device=device)
                local_chunk = torch.cat([local_chunk, pad_tensor], dim=0)

            gather_list = [torch.empty_like(local_chunk) for _ in range(world_size)]
            dist.all_gather(gather_list, local_chunk)

            if collect_result:
                for source_rank in range(world_size):
                    valid_n = max(0, min(n_list[source_rank] - start, current_chunk_size))
                    if valid_n == 0:
                        continue
                    dst_start = rank_offsets[source_rank] + start
                    dst_end = dst_start + valid_n
                    global_tensor[dst_start:dst_end].copy_(
                        gather_list[source_rank][:valid_n].detach().cpu())

            del local_chunk, gather_list

        return global_tensor

