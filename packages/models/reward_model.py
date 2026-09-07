"""
Libra Models - First-Principles Transformer Reward Model

Implements a scalar regression reward model r_theta(x, y) on top of the
modern transformer architecture for RLHF alignment using the Bradley-Terry
preference objective.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import nn

from packages.models.components.rmsnorm import RMSNorm
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerBlock


class RewardTelemetry(BaseModel):
    """Execution telemetry for a reward model ranking step."""

    loss: float
    accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Pairwise ranking accuracy (r_w > r_l)"
    )
    mean_chosen_reward: float
    mean_rejected_reward: float
    reward_margin: float = Field(
        ..., description="Mean difference between chosen and rejected rewards"
    )


class TransformerRewardModel(nn.Module):
    """Transformer equipped with a scalar regression head for completion reward scoring."""

    def __init__(self, config: ModernTransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [ModernTransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm_f = RMSNorm(config.d_model, eps=config.norm_eps)
        # Scalar regression head: d_model -> 1
        self.reward_head = nn.Linear(config.d_model, 1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        end_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes scalar rewards for input sequences.

        Args:
            input_ids: Tensor of shape (B, T)
            end_indices: Optional tensor of shape (B,) indicating the last token of completion.
                         If None, the final token in the sequence (T - 1) is used.

        Returns:
            Tensor of shape (B,) containing scalar rewards.
        """
        _B, T = input_ids.shape
        if T > self.config.max_context_length:
            raise ValueError(
                f"Sequence length ({T}) exceeds maximum context length ({self.config.max_context_length})"
            )

        x = self.drop(self.tok_emb(input_ids))
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)

        # Token-level scalar scores: (B, T, 1)
        scores = self.reward_head(x).squeeze(-1)  # (B, T)

        if end_indices is not None:
            # Gather score at each sequence's specific end index
            batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
            return scores[batch_indices, end_indices]
        else:
            # Default to last token
            return scores[:, -1]

    def compute_loss(
        self,
        chosen_input_ids: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        chosen_ends: torch.Tensor | None = None,
        rejected_ends: torch.Tensor | None = None,
        margin: float = 0.0,
    ) -> tuple[torch.Tensor, RewardTelemetry]:
        """
        Computes Bradley-Terry preference loss between chosen and rejected sequences:
            L_RM = -E[log sigma(r_w - r_l - margin)]
        """
        r_chosen = self(chosen_input_ids, end_indices=chosen_ends)
        r_rejected = self(rejected_input_ids, end_indices=rejected_ends)

        # Bradley-Terry pairwise cross entropy loss
        logits = r_chosen - r_rejected - margin
        loss = -F.logsigmoid(logits).mean()

        # Telemetry
        with torch.no_grad():
            acc = (r_chosen > r_rejected).float().mean().item()
            mean_c = r_chosen.mean().item()
            mean_r = r_rejected.mean().item()
            margin_val = (r_chosen - r_rejected).mean().item()

        telemetry = RewardTelemetry(
            loss=round(loss.item(), 4),
            accuracy=round(acc, 4),
            mean_chosen_reward=round(mean_c, 4),
            mean_rejected_reward=round(mean_r, 4),
            reward_margin=round(margin_val, 4),
        )

        return loss, telemetry
