"""
Libra Models - Sparse Mixture of Experts (MoE) Components
Implements Noisy Top-K Gating Router, Expert Layers, SparseMoEBlock,
and Switch/Shazeer Load-Balancing Auxiliary Loss.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from packages.models.components.swiglu import SwiGLU


class ExpertLayer(nn.Module):
    """Individual expert feed-forward module (SwiGLU-based)."""

    def __init__(self, d_model: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.ffn = SwiGLU(d_model=d_model, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class MoERouter(nn.Module):
    """Noisy Top-K Gating Router with Switch/Shazeer Load-Balancing Loss."""

    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        top_k: int = 2,
        aux_loss_coef: float = 0.01,
        noisy_gating: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.aux_loss_coef = aux_loss_coef
        self.noisy_gating = noisy_gating

        # Linear gating projection
        self.w_gate = nn.Linear(d_model, num_experts, bias=False)
        # Optional noise weight for exploration during training
        if self.noisy_gating:
            self.w_noise = nn.Linear(d_model, num_experts, bias=False)
        else:
            self.register_parameter("w_noise", None)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Compute routing logits, top-k expert selection, and auxiliary load balance loss.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            topk_weights: (B, T, top_k) normalized gating weights
            topk_indices: (B, T, top_k) selected expert IDs
            aux_loss: Scalar auxiliary load-balancing loss
            routing_metadata: Diagnostic telemetry
        """
        _B, _T, _C = x.shape
        router_logits = self.w_gate(x)  # (B, T, num_experts)

        # 1. Optional Exploration Noise during Training (Shazeer et al.)
        if self.noisy_gating and self.training and self.w_noise is not None:
            noise_scale = F.softplus(self.w_noise(x))
            eps = torch.randn_like(router_logits)
            noisy_logits = router_logits + eps * noise_scale
        else:
            noisy_logits = router_logits

        # 2. Select Top-K Experts
        topk_logits, topk_indices = torch.topk(noisy_logits, k=self.top_k, dim=-1)  # (B, T, top_k)

        # 3. Renormalize Top-K Weights (weights sum to 1.0 per token)
        topk_weights = F.softmax(topk_logits, dim=-1)

        # 4. Compute Switch/Shazeer Load-Balancing Auxiliary Loss
        # Full softmax routing probabilities over all experts
        router_probs = F.softmax(router_logits, dim=-1)  # (B, T, num_experts)
        # P_i: Average probability assigned to expert i
        p_mean = router_probs.view(-1, self.num_experts).mean(dim=0)  # (num_experts,)

        # f_i: Fraction of tokens routed to expert i
        # Flatten indices: (B * T, top_k)
        flat_indices = topk_indices.view(-1, self.top_k)
        # One-hot indicator for each expert selection
        mask_any = torch.zeros(
            flat_indices.shape[0], self.num_experts, device=x.device, dtype=torch.float
        )
        for k_idx in range(self.top_k):
            mask_any.scatter_add_(
                1,
                flat_indices[:, k_idx : k_idx + 1],
                torch.ones_like(flat_indices[:, k_idx : k_idx + 1], dtype=torch.float),
            )
        f_fraction = (mask_any > 0).float().mean(dim=0)  # (num_experts,)

        # aux_loss = coef * num_experts * sum_i (f_i * p_i)
        # Minimized when both f_i and p_i are uniform (1 / num_experts)
        aux_loss = self.aux_loss_coef * self.num_experts * torch.sum(f_fraction * p_mean)

        metadata = {
            "expert_fractions": [round(float(f.item()), 4) for f in f_fraction],
            "expert_probabilities": [round(float(p.item()), 4) for p in p_mean],
            "load_imbalance_cv": round(
                float(torch.std(f_fraction).item() / max(1e-5, f_fraction.mean().item())), 4
            ),
        }

        return topk_weights, topk_indices, aux_loss, metadata


class SparseMoEBlock(nn.Module):
    """Sparse Mixture of Experts (MoE) Feed-Forward Block."""

    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        aux_loss_coef: float = 0.01,
        noisy_gating: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = MoERouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_coef=aux_loss_coef,
            noisy_gating=noisy_gating,
        )

        self.experts = nn.ModuleList(
            [
                ExpertLayer(d_model=d_model, hidden_dim=hidden_dim, dropout=dropout)
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Route tokens to top-k experts and combine outputs.

        Returns:
            out: (B, T, d_model) combined expert output
            aux_loss: Load balancing loss scalar
            routing_info: Dictionary containing routing telemetry
        """
        topk_weights, topk_indices, aux_loss, metadata = self.router(x)

        # Output accumulator
        out = torch.zeros_like(x)

        # Dispatch tokens for each top-k rank
        for k_idx in range(self.top_k):
            expert_rank_indices = topk_indices[:, :, k_idx]  # (B, T)
            expert_rank_weights = topk_weights[:, :, k_idx].unsqueeze(-1)  # (B, T, 1)

            for exp_id in range(self.num_experts):
                mask = expert_rank_indices == exp_id
                if mask.any():
                    selected_tokens = x[mask]
                    exp_out = self.experts[exp_id](selected_tokens)
                    out[mask] += expert_rank_weights[mask] * exp_out

        routing_info = {
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
            "metadata": metadata,
        }

        return out, aux_loss, routing_info
