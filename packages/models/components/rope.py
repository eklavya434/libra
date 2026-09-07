"""
Libra Models - Rotary Positional Embeddings (RoPE)
Reference: Su et al., 2021 ("RoFormer: Enhanced Transformer with Rotary Position Embedding")

Mathematically encodes relative positions by rotating Query and Key vectors in 2D coordinate pairs.
For positions m and n, the dot product (q_m)^T (k_n) naturally reflects the relative distance (m - n),
providing length extrapolation and superior attention decay over distance.
"""

import torch
from torch import nn


def precompute_freqs_cis(
    dim: int, max_seq_len: int, theta_base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precomputes cosine and sine frequency tables for RoPE."""
    # Compute inverse frequencies for d/2 dimensions: theta_i = base^(-2i/d)
    freqs = 1.0 / (theta_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # Positions t = [0, 1, ..., max_seq_len - 1]
    t = torch.arange(max_seq_len, dtype=torch.float32)
    # Outer product: (max_seq_len, dim // 2)
    freqs = torch.outer(t, freqs)
    # Repeat along the last dimension so each pair (x_1, x_2) shares the frequency
    # Shape becomes (max_seq_len, dim)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates coordinate pairs: (x1, x2) -> (-x2, x1)."""
    # Reshape (..., dim) -> (..., dim//2, 2)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    # Interleave (-x2, x1) back to shape (..., dim)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to Query and Key tensors.

    Args:
        xq: Query tensor of shape (B, n_heads, T, head_dim)
        xk: Key tensor of shape (B, n_heads, T, head_dim)
        cos: Precomputed cosine table of shape (1, 1, T, head_dim)
        sin: Precomputed sine table of shape (1, 1, T, head_dim)
    """
    # 2D rotation formula: x' = (x * cos) + (rotate_half(x) * sin)
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out, xk_out


class RotaryEmbedding(nn.Module):
    """Module managing precomputed rotary frequencies and application to Q/K."""

    def __init__(self, head_dim: int, max_seq_len: int = 512, theta_base: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        cos, sin = precompute_freqs_cis(head_dim, max_seq_len, theta_base)
        # Register as buffers so they move with the module to CPU/GPU
        self.register_buffer("cos_cached", cos.unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", sin.unsqueeze(0).unsqueeze(0))

    def forward(
        self, xq: torch.Tensor, xk: torch.Tensor, seq_len: int, start_pos: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[:, :, start_pos : start_pos + seq_len, :]
        sin = self.sin_cached[:, :, start_pos : start_pos + seq_len, :]
        return apply_rotary_emb(xq, xk, cos, sin)
