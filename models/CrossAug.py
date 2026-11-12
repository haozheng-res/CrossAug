import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.SelfAttention_Family import AttentionLayer, FullAttention
from models.Timer import Model as TimerModel



class CrossAttentionLayer(nn.Module):
    """Cross-Attention → Add&Norm → FFN → Add&Norm"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        # Reuse the existing AttentionLayer + FullAttention without a causal mask.
        self.cross_attn = AttentionLayer(
            FullAttention(mask_flag=False, attention_dropout=dropout),
            d_model, n_heads
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        Args:
            query: [B, L_q, d_model]
            key:   [B, L_kv, d_model]
            value: [B, L_kv, d_model]
        Returns:
            [B, L_q, d_model]
        """
        # Cross-Attention
        new_q = self.cross_attn(query, key, value)  # [B, L_q, d_model]
        query = self.norm1(query + self.dropout(new_q))  # [B, L_q, d_model]
        # FFN
        out = self.ffn(query)  # [B, L_q, d_model]
        query = self.norm2(query + self.dropout(out))  # [B, L_q, d_model]
        return query


class LTMWrapper(nn.Module):
    """Wrap the frozen LTM branch with a consistent output shape of ``[B, forecast_len, 1]``."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.ltm = TimerModel(self._build_ltm_config(configs))
        self.patch_len = 96

        # Freeze all LTM parameters and keep the model in evaluation mode.
        self.ltm.requires_grad_(False)
        self.ltm.eval()

    @staticmethod
    def _build_ltm_config(configs):
        """Build a separate configuration matching the fixed architecture of the pretrained Timer weights."""
        checkpoint_path = os.fspath(configs.ltm_ckpt_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError('Timer checkpoint does not exist: {}'.format(checkpoint_path))
        ltm_cfg = copy.copy(configs)
        for key, value in {
            'ckpt_path': checkpoint_path,
            'patch_len': 96,
            'd_model': 1024,
            'd_ff': 2048,
            'e_layers': 8,
            'n_heads': 8,
            'dropout': 0.1,
            'activation': 'gelu',
        }.items():
            setattr(ltm_cfg, key, value)
        return ltm_cfg

    def train(self, mode=True):
        """Override train() to keep the LTM in evaluation mode at all times."""
        super().train(mode)
        self.ltm.eval()
        return self

    def _timer_single_step(self, x_input):
        """Run one Timer inference step and return the prediction for the next token (patch).
        Args:
            x_input: [B, k*patch_len, 1]
        Returns:
            [B, patch_len, 1]
        """
        # Timer applies RevIN normalization internally.
        timer_out = self.ltm(x_input)  # [B, k*patch_len, 1]
        # Decoder-only autoregression: the last patch_len steps predict the next token.
        return timer_out[:, -self.patch_len:, :]  # [B, patch_len, 1]

    def _forward_timer(self, batch_x):
        """Run Timer autoregressively in patch_len increments until pred_len is covered."""
        # Extract the target variable (the last column).
        x_target = batch_x[:, :, -1:]  # [B, seq_len, 1]

        # Align the input to the last k*patch_len steps so Timer unfold retains the most recent time steps.
        k = max(self.seq_len // self.patch_len, 1)
        aligned_len = k * self.patch_len
        x_aligned = x_target[:, -aligned_len:, :]  # [B, k*patch_len, 1]


        with torch.no_grad():
            if self.pred_len <= self.patch_len:
                # Single-step inference.
                next_patch = self._timer_single_step(x_aligned)  # [B, patch_len, 1]
                pred_ltm = next_patch[:, :self.pred_len, :]  # [B, pred_len, 1]
            else:
                # Iterative multi-step inference with a rolling window.
                pred_chunks = []
                current_input = x_aligned  # [B, k*patch_len, 1]
                remaining = self.pred_len
                while remaining > 0:
                    next_patch = self._timer_single_step(current_input)  # [B, patch_len, 1]
                    pred_chunks.append(next_patch)
                    remaining -= self.patch_len
                    if remaining > 0:
                        # Roll the window: remove the first patch_len steps and append the new prediction.
                        current_input = torch.cat([current_input[:, self.patch_len:, :], next_patch], dim=1)  # [B, k*patch_len, 1]
                pred_ltm = torch.cat(pred_chunks, dim=1)[:, :self.pred_len, :]  # [B, pred_len, 1]

        return pred_ltm

    def _validate_output(self, pred_ltm, batch_size):
        expected_shape = (batch_size, self.pred_len, 1)
        if not isinstance(pred_ltm, torch.Tensor):
            raise TypeError('Timer output must be a torch.Tensor, got {}.'.format(type(pred_ltm).__name__))
        if tuple(pred_ltm.shape) != expected_shape:
            raise ValueError('Timer output shape mismatch: expected {}, got {}.'.format(expected_shape, tuple(pred_ltm.shape)))
        return pred_ltm

    def forward(self, batch_x):
        """
        Args:
            batch_x: [B, seq_len, N]
        Returns:
            pred_ltm: [B, pred_len, 1]
        """
        pred_ltm = self._forward_timer(batch_x)

        return self._validate_output(pred_ltm, batch_x.shape[0])


class Model(nn.Module):
    """
    CrossAug: a cross-variable augmentation framework that uses exogenous information to enhance frozen LTM forecasts.
    - Branch 1: a frozen Timer produces univariate forecasts for the target variable.
    - Branch 2: Cross-Attention models how exogenous variables enhance LTM forecasts.
    - Gated fusion combines the outputs of both branches.
    """

    def __init__(self, configs):
        super().__init__()
        self.n_query = configs.n_query

        # === Branch 1: LTM branch (frozen) ===
        self.ltm_wrapper = LTMWrapper(configs)

        # === Branch 2: cross-variable interaction branch (trainable) ===
        # Input A: exogenous variable embeddings (iTransformer style, with one shared linear layer for all variables).
        self.exog_embedding = nn.Linear(configs.seq_len, configs.d_model)
        self.exog_dropout = nn.Dropout(configs.dropout)

        # Input B: LTM forecast embedding.
        self.ltm_proj = nn.Linear(configs.pred_len, configs.d_model)

        # Learnable queries.
        self.learned_queries = nn.Parameter(torch.randn(self.n_query, configs.d_model))  # [n_query, d_model]

        # Stack of Cross-Attention layers.
        self.cross_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=configs.d_model,
                n_heads=configs.n_heads,
                d_ff=configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
            )
            for _ in range(configs.e_layers)
        ])

        # Prediction head: weighted aggregation + linear projection.
        self.pool_weights = nn.Parameter(torch.ones(self.n_query + 1, 1))  # [n_query+1, 1]
        self.pool_norm = nn.LayerNorm(configs.d_model)
        self.pred_head = nn.Linear(configs.d_model, configs.pred_len)

        # Adaptive gating network: per-timestep fusion weights.
        self.gate_net = nn.Sequential(
            nn.Linear(configs.pred_len * 2, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, configs.pred_len),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate_net[-2].bias, 2.0)

    def forward(self, batch_x):
        """
        Args:
            batch_x:      [B, seq_len, N]
        Returns:
            [B, pred_len, 1]
        """
        B = batch_x.shape[0]  # [B, seq_len, N]

        # ======================== RevIN: input normalization ========================
        means = batch_x.mean(1, keepdim=True).detach()  # [B, 1, N]
        stdev = torch.sqrt(batch_x.var(1, keepdim=True, unbiased=False) + 1e-5).detach()  # [B, 1, N]

        target_mean = means[:, :, -1:]  # [B, 1, 1]
        target_std = stdev[:, :, -1:]  # [B, 1, 1]

        # ======================== Branch 1: LTM forecast ========================
        pred_ltm = self.ltm_wrapper(batch_x)  # [B, pred_len, 1] (original scale)
        # Normalize the LTM output into the common space.
        pred_ltm_norm = (pred_ltm - target_mean) / target_std  # [B, pred_len, 1]

        # ======================== Branch 2: cross-variable interaction ========================
        x_exog = batch_x[:, :, :-1]  # [B, seq_len, N-1]

        # Input A: exogenous variable embeddings.
        x_exog_t = x_exog.permute(0, 2, 1)  # [B, N-1, seq_len]
        exog_emb = self.exog_dropout(self.exog_embedding(x_exog_t))  # [B, N-1, d_model]

        # Input B: normalized LTM forecast embedding.
        ltm_flat = pred_ltm_norm.squeeze(-1)  # [B, pred_len]
        ltm_emb = self.ltm_proj(ltm_flat).unsqueeze(1)  # [B, 1, d_model]

        # Concatenate Q: LTM embedding + learnable queries.
        queries = self.learned_queries.unsqueeze(0).expand(B, -1, -1)  # [B, n_query, d_model]
        q = torch.cat([ltm_emb, queries], dim=1)  # [B, n_query+1, d_model]
        key = value = exog_emb  # [B, N-1, d_model]

        # Multiple Cross-Attention layers.
        for layer in self.cross_layers:
            q = layer(q, key, value)  # [B, n_query+1, d_model]

        # Prediction head: weighted aggregation -> independent forecast.
        weights = F.softmax(self.pool_weights, dim=0)  # [n_query+1, 1]
        pooled = (q * weights.unsqueeze(0)).sum(dim=1)  # [B, d_model]
        pooled = self.pool_norm(pooled)  # [B, d_model]
        pred_cross = self.pred_head(pooled).unsqueeze(-1)  # [B, pred_len, 1]

        # Adaptive per-timestep gated fusion.
        gate_input = torch.cat([
            pred_ltm_norm.squeeze(-1),  # [B, pred_len]
            pred_cross.squeeze(-1)      # [B, pred_len]
        ], dim=-1)  # [B, pred_len*2]
        alpha = self.gate_net(gate_input).unsqueeze(-1)  # [B, pred_len, 1]

        # alpha -> 1 favors the LTM; alpha -> 0 favors the cross-variable branch.
        final_pred_norm = alpha * pred_ltm_norm + (1 - alpha) * pred_cross  # [B, pred_len, 1]
        final_pred = final_pred_norm * target_std + target_mean  # [B, pred_len, 1]

        return final_pred
