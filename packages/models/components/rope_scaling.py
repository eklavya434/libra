"""
Libra Models - Rotary Positional Embedding (RoPE) Scaling & Long Context
References:
- Su et al., 2021: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Chen et al., 2023: "Extending Context Window of Large Language Models via Position Interpolation" (PI)
- bloc97, Peng et al., 2023: "NTK-Aware Scaled RoPE allows LLaMA models to have an extended context size"
- Peng et al., 2023: "YaRN: Efficient Context Window Extension of Large Language Models"

Provides first-principles implementations of:
1. Linear Position Interpolation (PI)
2. Dynamic NTK-Aware RoPE Scaling
3. YaRN (Band-split ramp interpolation + attention temperature entropy scaling)
"""

from __future__ import annotations

import math
from enum import Enum
import torch
from torch import nn

from packages.models.components.rope import apply_rotary_emb


class ScalingType(str, Enum):
    """Supported RoPE scaling strategies."""

    NONE = "none"
    LINEAR = "linear"
    DYNAMIC_NTK = "dynamic_ntk"
    YARN = "yarn"


def compute_base_freqs(dim: int, theta_base: float = 10000.0) -> torch.Tensor:
    """Computes base inverse angular frequencies for d/2 dimensions.

    theta_i = base^(-2i/d) for i in [0, 1, ..., d/2 - 1]
    """
    dim_indices = torch.arange(0, dim, 2)[: (dim // 2)].float()
    return 1.0 / (theta_base ** (dim_indices / dim))


def compute_freqs_linear(
    dim: int,
    max_seq_len: int,
    scale: float = 1.0,
    theta_base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear Position Interpolation (PI).

    Compresses input position indices t_scaled = t / s.
    Equivalent to multiplying frequencies by 1/s.
    """
    freqs = compute_base_freqs(dim, theta_base=theta_base)
    t = torch.arange(max_seq_len, dtype=torch.float32) / scale
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos, sin


def compute_freqs_dynamic_ntk(
    dim: int,
    seq_len: int,
    max_train_seq_len: int = 512,
    theta_base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic Neural Tangent Kernel (NTK)-Aware RoPE Scaling.

    Rather than uniformly stretching positions (which loses high-frequency local resolution),
    NTK-aware scaling scales the base frequency theta_base dynamically when seq_len > max_train_seq_len:
        s = seq_len / max_train_seq_len
        theta_base_new = theta_base * (s ** (d / (d - 2)))
    """
    if seq_len > max_train_seq_len:
        scale = seq_len / max_train_seq_len
        exponent = dim / (dim - 2.0)
        adjusted_base = theta_base * (scale**exponent)
    else:
        adjusted_base = theta_base

    freqs = compute_base_freqs(dim, theta_base=adjusted_base)
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos, sin


def compute_freqs_yarn(
    dim: int,
    max_seq_len: int,
    scale: float = 1.0,
    max_train_seq_len: int = 512,
    theta_base: float = 10000.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """YaRN (Yet another RoPE extensioN).

    Partitions the head dimension into 3 frequency bands using wavelength ratio r_i = L_train / lambda_i:
    1. r_i > beta_fast (High-freq): Extrapolation (scale = 1.0).
    2. r_i < beta_slow (Low-freq): Linear Interpolation (scale = s).
    3. beta_slow <= r_i <= beta_fast: Smooth ramp interpolation.
    """
    freqs = compute_base_freqs(dim, theta_base=theta_base)

    if scale <= 1.0:
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freq_grid = torch.outer(t, freqs)
        cos = torch.cos(freq_grid).repeat_interleave(2, dim=-1)
        sin = torch.sin(freq_grid).repeat_interleave(2, dim=-1)
        return cos, sin, 1.0

    wavelengths = 2.0 * math.pi / freqs
    r = max_train_seq_len / wavelengths
    ramp = torch.clamp((r - beta_slow) / (beta_fast - beta_slow), min=0.0, max=1.0)
    yarn_freqs = (1.0 - ramp) * (freqs / scale) + ramp * freqs

    t = torch.arange(max_seq_len, dtype=torch.float32)
    freq_grid = torch.outer(t, yarn_freqs)
    cos = torch.cos(freq_grid).repeat_interleave(2, dim=-1)
    sin = torch.sin(freq_grid).repeat_interleave(2, dim=-1)

    mscale = 0.1 * math.log(scale) + 1.0
    attn_scale_factor = 1.0 / math.sqrt(mscale)

    return cos, sin, attn_scale_factor


class ScaledRotaryEmbedding(nn.Module):
    """Dynamic & Scaled Rotary Position Embedding supporting standard, Linear, Dynamic NTK, and YaRN."""

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 512,
        original_max_seq_len: int = 512,
        theta_base: float = 10000.0,
        scaling_type: ScalingType | str = ScalingType.NONE,
        scale: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.original_max_seq_len = original_max_seq_len
        self.theta_base = theta_base
        if isinstance(scaling_type, ScalingType):
            self.scaling_type = scaling_type
        else:
            self.scaling_type = ScalingType(str(scaling_type).lower())
        self.scale = max(1.0, float(scale))
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.attn_temperature_factor: float = 1.0

        self._build_cache(self.max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        if self.scaling_type == ScalingType.LINEAR:
            cos, sin = compute_freqs_linear(
                dim=self.head_dim,
                max_seq_len=seq_len,
                scale=self.scale,
                theta_base=self.theta_base,
            )
            self.attn_temperature_factor = 1.0

        elif self.scaling_type == ScalingType.DYNAMIC_NTK:
            cos, sin = compute_freqs_dynamic_ntk(
                dim=self.head_dim,
                seq_len=seq_len,
                max_train_seq_len=self.original_max_seq_len,
                theta_base=self.theta_base,
            )
            self.attn_temperature_factor = 1.0

        elif self.scaling_type == ScalingType.YARN:
            cos, sin, attn_scale = compute_freqs_yarn(
                dim=self.head_dim,
                max_seq_len=seq_len,
                scale=self.scale,
                max_train_seq_len=self.original_max_seq_len,
                theta_base=self.theta_base,
                beta_fast=self.beta_fast,
                beta_slow=self.beta_slow,
            )
            self.attn_temperature_factor = attn_scale

        else:
            cos, sin = compute_freqs_linear(
                dim=self.head_dim,
                max_seq_len=seq_len,
                scale=1.0,
                theta_base=self.theta_base,
            )
            self.attn_temperature_factor = 1.0

        self.register_buffer("cos_cached", cos.unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", sin.unsqueeze(0).unsqueeze(0))

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        seq_len: int,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        required_len = start_pos + seq_len

        if (
            self.scaling_type == ScalingType.DYNAMIC_NTK
            and required_len > self.original_max_seq_len
            and (not hasattr(self, "cos_cached") or required_len > self.cos_cached.shape[2])
        ):
            self._build_cache(required_len)
        elif not hasattr(self, "cos_cached") or required_len > self.cos_cached.shape[2]:
            self._build_cache(max(required_len, self.max_seq_len))

        cos = self.cos_cached[:, :, start_pos : start_pos + seq_len, :].to(xq.device)
        sin = self.sin_cached[:, :, start_pos : start_pos + seq_len, :].to(xk.device)

        return apply_rotary_emb(xq, xk, cos, sin)
