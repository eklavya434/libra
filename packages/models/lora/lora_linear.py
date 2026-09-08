"""
Libra Models - Low-Rank Adaptation (LoRA) Linear Layer from First Principles
Phase 25: Parameter-Efficient Fine-Tuning (PEFT & LoRA)

Mathematical Formulation:
    W = W_0 + Delta_W = W_0 + (alpha / r) * (B @ A)
    where:
        W_0 in R^{d_out x d_in}  (Frozen base weights, requires_grad=False)
        A in R^{r x d_in}        (Trainable, Gaussian initialized)
        B in R^{d_out x r}       (Trainable, Zero initialized so Delta_W=0 initially)
        r << min(d_in, d_out)    (LoRA rank)
        alpha                    (Scaling hyperparameter)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear implementing Low-Rank Adaptation (LoRA).
    Freezes the original base weights and adds low-rank matrices A and B.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0

        # 1. Store and freeze base weight and optional bias
        self.weight = nn.Parameter(base_linear.weight.detach().clone(), requires_grad=False)
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # 2. Trainable Low-Rank Matrices: A and B
        if rank > 0:
            self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
            self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            self._reset_parameters()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_dropout = nn.Identity()

        # Tracking merge state
        self.merged = False

    def _reset_parameters(self) -> None:
        """
        Initializes matrix A with Kaiming uniform and matrix B with zeros.
        Since B is initialized to 0, Delta W = B @ A = 0 initially, ensuring
        100% exact numerical identity with the base model at start.
        """
        if self.rank > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def merge_weights(self) -> None:
        """
        Folds adapter delta weights into the base weight:
        W_merged = W_0 + (alpha / r) * (B @ A)
        Zero inference latency overhead.
        """
        if self.rank > 0 and not self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.data.add_(delta_w)
            self.merged = True

    def unmerge_weights(self) -> None:
        """
        Subtracts adapter delta weights from base weight:
        W_0 = W_merged - (alpha / r) * (B @ A)
        Restores ability to continue training or swap adapters.
        """
        if self.rank > 0 and self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.data.sub_(delta_w)
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes forward pass:
        If merged: Y = X @ W_merged^T + bias
        If unmerged: Y = X @ W_0^T + (alpha / r) * ((X @ A^T) @ B^T) + bias
        """
        if self.merged or self.rank == 0:
            return F.linear(x, self.weight, self.bias)

        # Base forward pass (frozen)
        base_out = F.linear(x, self.weight, self.bias)

        # LoRA forward pass: (x @ A^T) @ B^T * scaling
        lora_x = self.lora_dropout(x)
        lora_out = (lora_x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling

        return base_out + lora_out

    def trainable_parameters(self) -> int:
        """Returns number of trainable parameters in this layer."""
        if self.rank == 0:
            return 0
        return self.lora_A.numel() + self.lora_B.numel()

    def total_parameters(self) -> int:
        """Returns total parameter count in this layer."""
        total = self.weight.numel()
        if self.bias is not None:
            total += self.bias.numel()
        total += self.trainable_parameters()
        return total
