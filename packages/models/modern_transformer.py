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

from packages.models.components.kv_cache import KVCache
from packages.models.components.rmsnorm import RMSNorm
from packages.models.components.rope_scaling import ScaledRotaryEmbedding, ScalingType
from packages.models.components.swiglu import SwiGLU
from packages.models.modern_config import ModernTransformerConfig


class ModernCausalAttention(nn.Module):
    """Causal Multi-Head / Grouped-Query Attention equipped with Scaled Rotary Position Embeddings (RoPE)."""

    def __init__(self, config: ModernTransformerConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else config.n_heads
        self.head_dim = config.head_dim
        self.num_queries_per_kv = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # Scaled Rotary Positional Embedding module
        orig_seq_len = (
            config.original_max_seq_len
            if config.original_max_seq_len is not None
            else config.max_context_length
        )
        self.rope = ScaledRotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=config.max_context_length,
            original_max_seq_len=orig_seq_len,
            theta_base=config.rope_theta_base,
            scaling_type=config.rope_scaling_type,
            scale=config.rope_scale,
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.max_context_length, config.max_context_length))
        self.register_buffer(
            "causal_mask", mask.view(1, 1, config.max_context_length, config.max_context_length)
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
        start_pos: int = 0,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # 1. Project into Queries, Keys, Values
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 2. Apply RoPE with position offset
        q, k = self.rope(q, k, seq_len=T, start_pos=start_pos)

        # 3. Dynamic KV Cache update
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, layer_idx=layer_idx)
            total_seq_len = k.size(2)
        else:
            total_seq_len = T

        # 4. Grouped-Query Attention (GQA) Expansion
        if self.num_queries_per_kv > 1:
            k = torch.repeat_interleave(k, repeats=self.num_queries_per_kv, dim=1)
            v = torch.repeat_interleave(v, repeats=self.num_queries_per_kv, dim=1)

        # 5. Scaled dot-product attention
        # Apply YaRN attention temperature factor if configured
        scale_factor = (1.0 / math.sqrt(self.head_dim)) * getattr(self.rope, "attn_temperature_factor", 1.0)
        scores = (q @ k.transpose(-2, -1)) * scale_factor

        if T > 1:
            # Dynamic causal mask support if sequence exceeds pre-registered mask buffer
            if start_pos + T > self.causal_mask.size(2) or total_seq_len > self.causal_mask.size(3):
                max_len = max(start_pos + T, total_seq_len)
                mask = torch.tril(torch.ones(max_len, max_len, device=x.device)).view(1, 1, max_len, max_len)
                submask = mask[:, :, start_pos : start_pos + T, :total_seq_len]
            else:
                submask = self.causal_mask[:, :, start_pos : start_pos + T, :total_seq_len]
            scores = scores.masked_fill(submask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 6. Context aggregation & output projection
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

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
        start_pos: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x), kv_cache=kv_cache, layer_idx=layer_idx, start_pos=start_pos
        )
        x = x + self.mlp(self.norm2(x))
        return x


class ModernTransformerLM(nn.Module):
    """Modern Llama-style Autoregressive Language Model with GQA and KV Cache support."""

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
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _B, T = idx.shape
        total_len = start_pos + T
        if total_len > self.config.max_context_length:
            raise ValueError(
                f"Sequence length ({total_len}) exceeds maximum context length ({self.config.max_context_length})"
            )

        # No absolute positional embedding lookup! RoPE handles position inside attention.
        x = self.drop(self.tok_emb(idx))

        for layer_idx, block in enumerate(self.blocks):
            x = block(
                x,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                start_pos=start_pos,
            )

        x = self.norm_f(x)
        logits = self.output_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
