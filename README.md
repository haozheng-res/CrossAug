# CrossAug

This repository contains the code for the paper "CrossAug: A Cross-Variate Interaction Augmentation Framework for Time Series Foundation Models".

## Overview

CrossAug augments a frozen time series foundation model with cross-variate information for forecasting. It uses two branches: a frozen foundation model produces a univariate forecast for the target variable, while a trainable cross-attention branch extracts complementary information from exogenous variables. An adaptive time-step-wise gate fuses the two forecasts in a normalized space.

## Implementation

#### 1. Environment

```text
python == 3.10.14
pytorch == 2.2.2
numpy >= 1.23, < 2
pandas >= 1.5, < 3
scikit-learn >= 1.2, < 2
```
A CUDA-capable NVIDIA GPU is required. The default training mode uses two GPUs with DistributedDataParallel (DDP).

#### 2. Code Running

Training and evaluation are launched through `run.py`. Prepare the dataset and a compatible Timer checkpoint, then run the following command from the repository root, replacing the data and checkpoint paths:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 run.py \
  --model_id ETTh1_Timer_512_96 \
  --ltm_ckpt_path /path/to/Timer_finetuning.ckpt \
  --data ETTh1 --root_path /path/to/ETT-small --data_path ETTh1.csv \
  --features M --seq_len 512 --pred_len 96
```

Checkpoint must match the backbone configuration. Supported checkpoint formats are `.ckpt` and `.pth`.

Use `--features M` to rotate each variable as the prediction target, or `--features MS` to predict the last numerical column.

## Acknowledgments

We thank the authors of the following open-source projects for making their code publicly available.

- [Timer](https://github.com/thuml/Large-Time-Series-Model)
- [Timer-XL](https://github.com/thuml/Timer-XL)
