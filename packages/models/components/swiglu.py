"""
Libra Models - SwiGLU Feed-Forward Network
Reference: Noam Shazeer, 2020 ("GLU Variants Improve Transformer")

Mathematically:
    SwiGLU(x) = (Linear_gate(x) * SiLU(Linear_up(x))) * Linear_down

Used by Llama 2/3, Mistral, and DeepSeek. The gating mechanism allows the network
to dynamically filter and scale information flowing through the MLP.
"""

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit (SwiGLU) Feed-Forward Layer."""

    def __init__(self, d_model: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        # Standard Llama formula: 8/3 * d_model rounded to multiple of 32
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * d_model) / 3)
            hidden_dim = ((hidden_dim + 31) // 32) * 32

        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, d_model) -> gate & up projections
        gate = self.w_gate(x)
        up = self.w_up(x)
        # Swish gating: gate * SiLU(up)
        activated = gate * F.silu(up)
        # Down projection back to d_model
        return self.dropout(self.w_down(activated))
