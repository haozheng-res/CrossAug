from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_provider.data_loader_exogenous import ExogenousDatasetBenchmark


def data_provider(args, flag):
    """Build the CrossAug dataset, preserving the original data windows, sampling, and DDP behavior."""
    data_set = ExogenousDatasetBenchmark(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.pred_len],
        features=args.features,
    )
    print(flag, len(data_set))
    shuffle_flag = flag not in ('test', 'TEST')

    if not args.use_dp:
        # Use the original DistributedSampler for DDP; the training loop sets its random seed each epoch.
        sampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            pin_memory=True,
            drop_last=False,
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=False,
        )

    return data_set, data_loader
