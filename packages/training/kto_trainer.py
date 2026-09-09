"""
Libra Training - Kahneman-Tversky Optimization (KTO) Trainer

Implements Kahneman-Tversky Optimization (Ethayarajh et al., 2024).
Directly optimizes language models on unpaired binary feedback (desirable vs undesirable)
grounded in behavioral economics and Prospect Theory, incorporating loss aversion
weights (lambda_U > lambda_D) without requiring pairwise ranked data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import nn, optim


class KTOTelemetry(BaseModel):
    """Telemetry metrics emitted during a KTO training step."""

    loss: float
    desirable_loss: float
    undesirable_loss: float
    desirable_reward: float
    undesirable_reward: float
    reward_margin: float = Field(
        ..., description="Implicit reward margin: r_desirable - r_undesirable"
    )
    kl_divergence: float = Field(
        ..., description="Mean policy KL divergence against reference policy"
    )
    loss_aversion_ratio: float = Field(..., description="Ratio lambda_U / lambda_D")


@dataclass
class KTOConfig:
    """Hyperparameters for Kahneman-Tversky Optimization."""

    beta: float = 0.1  # KL regularization / reward scale temperature
    desirable_weight: float = 1.0  # lambda_D: weight for desirable outcomes
    undesirable_weight: float = 1.33  # lambda_U: weight for undesirable outcomes (loss aversion)
    lr: float = 5e-5  # Learning rate
    weight_decay: float = 0.01  # Weight decay
    max_grad_norm: float = 1.0  # Gradient clipping norm
    length_normalized: bool = True  # Whether to length-normalize completion log-probs


class KTOTrainer:
    """Orchestrates KTO optimization for an autoregressive language model."""

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        config: KTOConfig | None = None,
        optimizer: optim.Optimizer | None = None,
    ) -> None:
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.config = config or KTOConfig()

        # Completely freeze reference model
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False

        self.optimizer = optimizer or optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    @staticmethod
    def get_completion_logps(
        logits: torch.Tensor,
        labels: torch.Tensor,
        length_normalized: bool = True,
    ) -> torch.Tensor:
        """
        Computes per-sample log probabilities over completion tokens (labels != -100).
        """
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_mask = shift_labels != -100
        safe_labels = shift_labels.clone()
        safe_labels[~loss_mask] = 0

        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(
            -1
        )
        masked_logps = per_token_logps * loss_mask.float()

        token_counts = loss_mask.sum(dim=-1).clamp(min=1)
        seq_logps = masked_logps.sum(dim=-1)

        if length_normalized:
            return seq_logps / token_counts.float()
        return seq_logps

    def forward_implicit_rewards(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes implicit rewards r_theta(x, y) = beta * (log pi_theta - log pi_ref).
        """
        policy_out = self.policy_model(input_ids)
        policy_logits = policy_out[0] if isinstance(policy_out, tuple) else policy_out

        with torch.no_grad():
            ref_out = self.reference_model(input_ids)
            ref_logits = ref_out[0] if isinstance(ref_out, tuple) else ref_out

        policy_logps = self.get_completion_logps(
            policy_logits, labels, length_normalized=self.config.length_normalized
        )
        ref_logps = self.get_completion_logps(
            ref_logits, labels, length_normalized=self.config.length_normalized
        )

        rewards = self.config.beta * (policy_logps - ref_logps)
        return rewards, policy_logps, ref_logps

    def compute_kto_loss(
        self,
        rewards: torch.Tensor,
        is_desirable: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Calculates the asymmetric Kahneman-Tversky Prospect Theory loss.
        """
        z_ref = rewards.detach().mean()

        desirable_mask = is_desirable.bool()
        undesirable_mask = ~desirable_mask

        desirable_loss = torch.tensor(0.0, device=rewards.device, requires_grad=True)
        undesirable_loss = torch.tensor(0.0, device=rewards.device, requires_grad=True)

        if desirable_mask.any():
            r_desirable = rewards[desirable_mask]
            d_losses = 1.0 - torch.sigmoid(r_desirable - z_ref)
            desirable_loss = self.config.desirable_weight * d_losses.mean()

        if undesirable_mask.any():
            r_undesirable = rewards[undesirable_mask]
            u_losses = 1.0 - torch.sigmoid(z_ref - r_undesirable)
            undesirable_loss = self.config.undesirable_weight * u_losses.mean()

        total_loss = desirable_loss + undesirable_loss
        return total_loss, desirable_loss, undesirable_loss, z_ref.item()

    def train_step(self, batch: dict[str, torch.Tensor]) -> KTOTelemetry:
        """Executes a single KTO optimization step on a batched tensor dictionary."""
        self.policy_model.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        is_desirable = batch["is_desirable"]

        rewards, policy_logps, ref_logps = self.forward_implicit_rewards(input_ids, labels)
        total_loss, des_loss, und_loss, _ = self.compute_kto_loss(rewards, is_desirable)

        total_loss.backward()

        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(), self.config.max_grad_norm
            )

        self.optimizer.step()

        des_mask = is_desirable.bool()
        und_mask = ~des_mask

        mean_des_reward = rewards[des_mask].mean().item() if des_mask.any() else 0.0
        mean_und_reward = rewards[und_mask].mean().item() if und_mask.any() else 0.0
        mean_kl = (policy_logps - ref_logps).mean().item()

        aversion_ratio = round(
            self.config.undesirable_weight / max(1e-5, self.config.desirable_weight), 2
        )

        return KTOTelemetry(
            loss=round(total_loss.item(), 5),
            desirable_loss=round(des_loss.item(), 5),
            undesirable_loss=round(und_loss.item(), 5),
            desirable_reward=round(mean_des_reward, 4),
            undesirable_reward=round(mean_und_reward, 4),
            reward_margin=round(mean_des_reward - mean_und_reward, 4),
            kl_divergence=round(mean_kl, 4),
            loss_aversion_ratio=aversion_ratio,
        )
