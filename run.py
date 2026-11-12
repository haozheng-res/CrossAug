import argparse
import os
import random
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.tools import HiddenPrints


def parse_args():
    parser = argparse.ArgumentParser(description='CrossAug + frozen Timer forecasting')
    parser.add_argument('--is_training', type=int, default=1, choices=[0, 1])
    parser.add_argument('--model_id', type=str, default='CrossAug_Timer')
    parser.add_argument('--checkpoint', type=str, default='', help='full CrossAug checkpoint for test-only mode')
    parser.add_argument('--ltm_ckpt_path', type=str, required=True, help='pre-trained Timer checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        choices=['custom', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Solar'])
    parser.add_argument('--root_path', type=str, default='./data/ETT/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M', choices=['M', 'MS'])
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--subset_sampling_ratio', type=float, default=1.0)

    # Trainable interaction branch; the Timer architecture is fixed by its pretrained weights.
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='gelu', choices=['gelu', 'relu'])
    parser.add_argument('--n_query', type=int, default=4)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--lradj', type=str, default='type1',
                        choices=['type0', 'type1', 'type2', 'type3', 'cosine', 'onecycle', 'cosine_batch'])
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--pct_start', type=float, default=0.3)
    parser.add_argument('--max_norm', type=float, default=0.0)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--loss_alpha', type=float, default=0.0)
    parser.add_argument('--loss_mode', type=str, default='L2', choices=['L1', 'L2', 'L1L2'])
    parser.add_argument('--disable_test_every_epoch', action='store_true')
    parser.add_argument('--test_step_detail', action='store_true')
    parser.add_argument('--cache_ltm_outputs', action='store_true')
    parser.add_argument('--use_dp', action='store_true', help='single GPU or DataParallel; default is DDP')
    parser.add_argument('--seed', type=int, default=2021)

    args = parser.parse_args()
    if args.seq_len < 96:
        parser.error('Timer requires --seq_len >= 96.')
    if args.pred_len <= 0 or args.batch_size <= 0 or args.train_epochs <= 0 or args.itr <= 0:
        parser.error('pred_len, batch_size, train_epochs and itr must be positive.')
    if not 0 < args.subset_sampling_ratio <= 1:
        parser.error('subset_sampling_ratio must be in (0, 1].')
    if args.is_training == 0 and not args.checkpoint:
        parser.error('--is_training 0 requires --checkpoint (full CrossAug weights).')
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required.')
    args.device_ids = list(range(torch.cuda.device_count()))
    if not args.use_dp:
        if len(args.device_ids) < 2:
            raise RuntimeError('DDP requires at least two visible GPUs; use --use_dp for one GPU.')
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=5))

    try:
        with HiddenPrints(dist.get_rank() if not args.use_dp else 0):
            for iteration in range(args.itr if args.is_training else 1):
                setting = (
                    f'{args.model_id}_{args.data}_ft{args.features}_sl{args.seq_len}_pl{args.pred_len}'
                    f'_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_df{args.d_ff}_nq{args.n_query}'
                    f'_{args.des}_bs{args.batch_size}_dr{args.dropout}_lr{args.learning_rate}'
                    f'_pa{args.patience}_{args.lradj}_ep{args.train_epochs}_la{args.loss_alpha}_{iteration}'
                )
                experiment = Exp_Long_Term_Forecast(args)
                if args.is_training:
                    print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    experiment.train(setting)
                else:
                    experiment.load_checkpoint(args.checkpoint)
                print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                experiment.test(setting)
                del experiment
                torch.cuda.empty_cache()
        if not args.use_dp:
            dist.barrier()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
