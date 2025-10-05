# Taken from https://github.com/suvash/nnze2he/blob/main/makemore/src/gpt.py
import sys
import os
from pathlib import Path
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
CURRENT_DIR = Path(__file__).resolve().parent
STREAM_DVFS_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(STREAM_DVFS_DIR))
from src.config import TransformerConfig
from src.named_ops import NamedAdd, NamedMatmul

device = "cpu"
dropout = 0.3


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()  # type: ignore
        self.cfg = cfg

        # We compute each linear projection as one big MatMul to ensure spatial array utilization
        self.key_proj = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=False)
        self.query_proj = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=False)
        self.value_proj = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=False)

        self.mul_qk_t = NamedMatmul()
        self.mul_logits_v = NamedMatmul()

        self.out_proj = nn.Linear(cfg.num_head * cfg.head_size, cfg.embedding_dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        B = self.cfg.batch_size
        L = self.cfg.decode_idx - 1
        d_h = self.cfg.head_size
        D = self.cfg.embedding_dim
        H = D // d_h

        key_token: Tensor = self.key_proj(x).reshape(B, 1, H, d_h).permute((0, 2, 1, 3))  # (B, H, 1, d_h)
        query_token: Tensor = self.query_proj(x).reshape(B, 1, H, d_h).permute((0, 2, 1, 3))
        value_token: Tensor = self.value_proj(x).reshape(B, 1, H, d_h).permute((0, 2, 1, 3))

        # This would be loaded from memory
        key_cache = torch.empty((B, H, L, d_h), device=device)
        full_key = torch.cat((key_cache, key_token), dim=-2)  # (B, H, L+1, d_h)
        key_transpose = full_key.transpose(-2, -1)  # (B, H, d_h, L+1)

        # This would be loaded from memory
        value_cache = torch.empty((B, H, L, d_h), device=device)
        full_value = torch.cat((value_cache, value_token), dim=-2)  # (B, H, L+1, d_h)

        # One row of the attention matrix:  (B, H, 1, d_h) @ (B, H, d_h, L+1) -> (B, H, 1, L+1)
        attention_token: Tensor = self.mul_qk_t(query_token, key_transpose)
        attention_token = attention_token / math.sqrt(self.cfg.head_size)
        # Don't have to do masking, since this is the bottom row of a triangular matrix

        logits = F.softmax(attention_token, dim=-1)  # (B, H, 1, L+1)
        logits = self.dropout1(logits)
        out = self.mul_logits_v(logits, full_value)  # (B, H, 1, d_h)
        out = out.permute((0, 2, 1, 3))  # (B, 1, H, d_h)
        out = out.reshape(B, 1, D)

        out = self.out_proj(out)
        out = self.dropout2(out)
        return out


class FeedForward(nn.Module):
    "simple linear layer followed by non linearity"

    def __init__(self, cfg: TransformerConfig):
        super().__init__()  # type: ignore
        self.up_proj = nn.Linear(cfg.embedding_dim, cfg.dim_ff, bias=True)
        self.down_proj = nn.Linear(cfg.dim_ff, cfg.embedding_dim, bias=True)

        self.net = nn.Sequential(
            self.up_proj,
            nn.ReLU(),
            self.down_proj,
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor):
        return self.net(x)


class Block(nn.Module):
    """a transformer block : communication then computation"""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()  # type: ignore

        self.sa = MultiHeadAttention(cfg)
        self.feed_forward = FeedForward(cfg)
        self.residual_attn = NamedAdd()
        self.residual_ffn = NamedAdd()
        self.layer_norm1 = nn.LayerNorm(cfg.embedding_dim)
        self.layer_norm2 = nn.LayerNorm(cfg.embedding_dim)

    def forward(self, x: Tensor):
        attn = self.sa(self.layer_norm1(x))
        x = self.residual_attn(x, attn)
        ffn = self.feed_forward(self.layer_norm2(x))
        x = self.residual_ffn(x, ffn)
        return x


class LanguageModelDecode(nn.Module):
    """Run inference in the decode stage for a single token. The token at index decode_idx in the context window will
    be generated"""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()  # type: ignore
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.position_embedding_table = nn.Embedding(1, cfg.embedding_dim)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.num_layer)])
        self.layer_norm_final = nn.LayerNorm(cfg.embedding_dim)
        # self.de_embed = nn.Linear(cfg.embedding_dim, cfg.vocab_size)

    def forward(self, idx: Tensor):
        _, L = idx.shape

        # idx and targets are both (B, L) tensor of integers
        token_emb = self.token_embedding_table(idx)  # (B, 1, D)
        pos_emb = self.position_embedding_table(torch.arange(L, device=device))  # (1, D)
        x = token_emb + pos_emb  # (B, 1, D)
        x = self.blocks(x)
        x = self.layer_norm_final(x)
        # x = self.de_embed(x)  # (B,1,VOCAB_SIZE)

        return x
