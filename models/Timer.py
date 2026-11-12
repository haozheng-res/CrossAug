import torch
from torch import nn

from layers.Embed import PatchEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import AttentionLayer, FullAttention


class TimerBackbone(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, configs.patch_len, configs.dropout)

        # Decoder-only Transformer: Refer to issue: https://github.com/thuml/Large-Time-Series-Model/issues/23
        self.decoder = Encoder(
            [
                 EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=configs.dropout), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Prediction Head
        self.proj = nn.Linear(configs.d_model, configs.patch_len, bias=True)


class Model(nn.Module):
    """
    Timer: Generative Pre-trained Transformers Are Large Time Series Models (ICML 2024)
    Paper: https://arxiv.org/abs/2402.02368
    GitHub: https://github.com/thuml/Large-Time-Series-Model
    """

    def __init__(self, configs):
        super().__init__()
        self.ckpt_path = configs.ckpt_path
        self.patch_len = configs.patch_len


        self.backbone = TimerBackbone(configs)
        # Decoder
        self.decoder = self.backbone.decoder
        self.proj = self.backbone.proj
        self.enc_embedding = self.backbone.patch_embedding

        if self.ckpt_path != '':
            print('loading model: ', self.ckpt_path)
            if self.ckpt_path.endswith('.pth'):
                sd = torch.load(self.ckpt_path, map_location="cpu")
                # Strip the DDP module. prefix.
                if any(k.startswith('module.') for k in sd.keys()):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                # Support both full-model state_dicts (with the backbone. prefix) and backbone-only state_dicts.
                if any(k.startswith('backbone.') for k in sd.keys()):
                    sd = {k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')}
                self.backbone.load_state_dict(sd)
            elif self.ckpt_path.endswith('.ckpt'):
                sd = torch.load(self.ckpt_path, map_location="cpu")["state_dict"]  # ["state_dict"] extracts the actual model parameters from the dictionary.
                sd = {k[6:]: v for k, v in sd.items()}  # k[6:] removes the "model." prefix from saved keys such as model.fc.weight.
                self.backbone.load_state_dict(sd, strict=True)
            else:
                raise NotImplementedError

    def forward(self, x_enc):
        B, L, M = x_enc.shape  # [B, L, 1]: univariate sequence of the target variable.

        # === CrossAug-style end alignment (comment out this block to restore the official Timer implementation) ===
        # If the input length is not divisible by patch_len, discard the earliest remainder points and retain the most recent complete patches,
        # so the last input patch still ends at the current forecast origin, avoiding temporal misalignment.
        aligned_len = (L // self.patch_len) * self.patch_len
        if 0 < aligned_len < L:
            x_enc = x_enc[:, -aligned_len:, :]  # [B, aligned_len, M]

        # Normalization from Non-stationary Transformer  # Reversible Instance Norm
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)  # [B, M, T]
        dec_in = self.enc_embedding(x_enc)  # [B * M, N, L] / [B * M, N, D]

        # Transformer Blocks
        dec_out = self.decoder(dec_in)  # [B * M, N, D]
        dec_out = self.proj(dec_out)  # [B * M, N, L]
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2)  # [B, T, M], Timer's forecast output length does not exceed the input length.

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev + means
        return dec_out
