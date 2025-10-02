# Adapted from https://github.com/suvash/nnze2he/blob/main/makemore/src/gpt.py

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

        # self.register_buffer("tril", torch.tril(torch.ones(cfg.prefill_size, cfg.prefill_size)))
        self.out_proj = nn.Linear(cfg.num_head * cfg.head_size, cfg.embedding_dim, bias=False)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        B = self.cfg.batch_size
        L = self.cfg.prefill_size
        d_h = self.cfg.head_size
        D = self.cfg.embedding_dim
        H = D // d_h

        key: Tensor = self.key_proj(x).reshape(B, L, H, d_h).permute((0, 2, 1, 3))  # (B, H, L, d_h)
        query: Tensor = self.query_proj(x).reshape(B, L, H, d_h).permute((0, 2, 1, 3))
        value: Tensor = self.value_proj(x).reshape(B, L, H, d_h).permute((0, 2, 1, 3))

        key_transpose = key.transpose(-2, -1)  # (B, H, L, d_h) -> (B, H, d_h, L)

        attention: Tensor = self.mul_qk_t(query, key_transpose)  # (B, H, L, d_h) @ (B, H, d_h, L) -> (B, H, L, L)
        attention = attention / math.sqrt(d_h)
        # attention = attention.masked_fill(attention[:L, :L] == 0, float("-inf"))  # (B, H, L, L)

        logits = F.softmax(attention, dim=-1)  # (B, H, L, L)
        logits = self.dropout1(logits)
        out = self.mul_logits_v(logits, value)  # (B, H, L, d_h)
        out = out.permute((0, 2, 1, 3))  # (B, L, H, d_h)
        out = out.reshape(B, L, D)

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


class LanguageModel(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()  # type: ignore
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.position_embedding_table = nn.Embedding(cfg.prefill_size, cfg.embedding_dim)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.num_layer)])
        self.layer_norm_final = nn.LayerNorm(cfg.embedding_dim)
        # self.de_embed = nn.Linear(cfg.embedding_dim, cfg.vocab_size)

    def forward(self, tokens: Tensor):
        _, L = tokens.shape

        # tokens and targets are both (B, L) tensor of integers
        token_emb = self.token_embedding_table(tokens)  # (B, L, D)
        pos_emb = self.position_embedding_table(torch.arange(L, device=device))  # (L, D)
        x = token_emb + pos_emb  # (B, L, D)
        x = self.blocks(x)
        x = self.layer_norm_final(x)
        # x = self.de_embed(x)  # (B,L,VOCAB_SIZE)

        return x
