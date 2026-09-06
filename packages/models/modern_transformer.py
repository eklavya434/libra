"""
Libra Models - Modern Llama-Style Decoder-Only Transformer Language Model
Features:
  1. Rotary Positional Embeddings (RoPE) replacing absolute positional embeddings
  2. Root Mean Square Layer Normalization (RMSNorm) replacing LayerNorm
  3. SwiGLU Gated Feed-Forward Networks replacing standard MLP
  4. Optional Weight Tying (tying input embeddings with output head)
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from packages.models.components.rmsnorm import RMSNorm
from packages.models.components.rope import RotaryEmbedding
from packages.models.components.swiglu import SwiGLU
from packages.models.modern_config import ModernTransformerConfig


class ModernCausalAttention(nn.Module):
    """Causal Multi-Head Attention equipped with Rotary Position Embeddings (RoPE)."""

    def __init__(self, config: ModernTransformerConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # Rotary Positional Embedding module
        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=config.max_context_length,
            theta_base=config.rope_theta_base,
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.max_context_length, config.max_context_length))
        self.register_buffer(
            "causal_mask", mask.view(1, 1, config.max_context_length, config.max_context_length)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # 1. Project into Queries, Keys, Values
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 2. Apply RoPE 2D rotations to Query and Key
        q, k = self.rope(q, k, seq_len=T)

        # 3. Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 4. Context aggregation & output projection
        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))


class ModernTransformerBlock(nn.Module):
    """Pre-RMSNorm Transformer Block with RoPE Attention and SwiGLU MLP."""

    def __init__(self, config: ModernTransformerConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = ModernCausalAttention(config)
        self.norm2 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = SwiGLU(config.d_model, hidden_dim=config.hidden_dim, dropout=config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ModernTransformerLM(nn.Module):
    """Modern Llama-style Autoregressive Language Model."""

    def __init__(self, config: ModernTransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [ModernTransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm_f = RMSNorm(config.d_model, eps=config.norm_eps)
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Optional Weight Tying
        if config.tie_weights:
            self.output_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _B, T = idx.shape
        if T > self.config.max_context_length:
            raise ValueError(
                f"Sequence length ({T}) exceeds maximum context length ({self.config.max_context_length})"
            )

        # No absolute positional embedding lookup! RoPE handles position inside attention.
        x = self.drop(self.tok_emb(idx))

        for block in self.blocks:
            x = block(x)

        x = self.norm_f(x)
        logits = self.output_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
