"""
Libra Models - Root Mean Square Layer Normalization (RMSNorm)
Reference: Zhang & Sennrich, 2019 ("Root Mean Square Layer Normalization")

Mathematically:
    RMS(x) = sqrt(mean(x^2) + eps)
    y = (x / RMS(x)) * gamma

Unlike standard LayerNorm, RMSNorm does not subtract the mean, saving ~10-50% computation
while achieving identical training stability in modern LLMs (Llama, Mistral, Gemma).
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # Learnable scaling parameter (gamma), initialized to 1.0
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # Compute root mean square across the last dimension
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale input tensor by learnable weight
        return self._norm(x.float()).type_as(x) * self.weight
