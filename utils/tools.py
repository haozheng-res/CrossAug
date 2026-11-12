import os
import sys

import numpy as np
import torch
import torch.distributed as dist


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # Per-epoch strategies.
    if args.lradj == 'type0':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'cosine':
        import math
        warmup_epochs = getattr(args, 'warmup_epochs', 3)
        if epoch <= warmup_epochs:
            lr = args.learning_rate * epoch / warmup_epochs
        else:
            lr = args.learning_rate * 0.5 * (1 + math.cos(
                math.pi * (epoch - warmup_epochs) / (args.train_epochs - warmup_epochs)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            print('Updating learning rate to {}'.format(lr))
        return

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            print('Updating learning rate to {}'.format(lr))


def build_lr_scheduler(optimizer, train_steps, args):
    """Build a learning rate scheduler advanced by the training loop after each batch."""
    if args.lradj == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            steps_per_epoch=train_steps,
            pct_start=args.pct_start,
            epochs=args.train_epochs,
            max_lr=args.learning_rate,
        )
    if args.lradj == 'cosine_batch':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=args.train_epochs * train_steps,
        )
    return None


def step_batch_lr_scheduler(scheduler, args):
    """Advance the per-batch scheduler after optimizer.step()."""
    if args.lradj in ('onecycle', 'cosine_batch'):
        scheduler.step()


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, args=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.args = args

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if int(os.environ.get("RANK", "0")) == 0:
            checkpoint = model.state_dict()
            torch.save(checkpoint, path + '/checkpoint.pth')
        if not self.args.use_dp:
            dist.barrier()
        self.val_loss_min = val_loss


class HiddenPrints:
    def __init__(self, rank):
        # If rank is None, this is a single-machine, single-GPU run; keep printing enabled by setting rank to 0.
        if rank is None:
            rank = 0
        self.rank = rank
    def __enter__(self):
        if self.rank == 0:
            return
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.rank == 0:
            return
        sys.stdout.close()
        sys.stdout = self._original_stdout
