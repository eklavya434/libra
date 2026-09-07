"""
Libra Training - Direct Preference Optimization (DPO) Trainer

Implements the Direct Preference Optimization algorithm (Rafailov et al., NeurIPS 2023).
Directly optimizes policy parameters pi_theta from preference pairs (x, y_w, y_l)
regularized against a frozen reference policy pi_ref without needing an explicit
reward model or PPO actor-critic loops.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import nn, optim


class DPOTelemetry(BaseModel):
    """Telemetry metrics emitted during a DPO training step."""

    loss: float
    accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of pairs where chosen implicit reward > rejected"
    )
    chosen_implicit_reward: float
    rejected_implicit_reward: float
    reward_margin: float = Field(..., description="Implicit reward margin: r_chosen - r_rejected")
    policy_chosen_logp: float
    policy_rejected_logp: float


@dataclass
class DPOConfig:
    """Configuration hyperparameters for DPO optimization."""

    beta: float = 0.1  # KL regularization temperature
    lr: float = 5e-5  # Learning rate
    weight_decay: float = 0.01  # Weight decay
    max_grad_norm: float = 1.0  # Gradient clipping ceiling
    label_smoothing: float = 0.0  # Optional label smoothing factor


class DPOTrainer:
    """Orchestrates Direct Preference Optimization for an autoregressive language model."""

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        config: DPOConfig | None = None,
        optimizer: optim.Optimizer | None = None,
    ) -> None:
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.config = config or DPOConfig()

        # Freeze reference model completely
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False

        self.optimizer = optimizer or optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    @staticmethod
    def get_batch_logps(
        logits: torch.Tensor,
        labels: torch.Tensor,
        average_log_prob: bool = False,
    ) -> torch.Tensor:
        """
        Computes per-sequence log probabilities specifically over completion tokens.

        Args:
            logits: Output logits of shape (B, T, Vocab)
            labels: Target labels of shape (B, T) with prompt and pad positions masked to -100
            average_log_prob: If True, returns length-normalized log probabilities
        """
        # Shift logits and labels so that position t predicts token at t+1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Mask where tokens are to be evaluated
        loss_mask = shift_labels != -100

        # Replace -100 with 0 for gather operation
        safe_labels = shift_labels.clone()
        safe_labels[~loss_mask] = 0

        # Compute log-softmax over vocabulary
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(
            -1
        )

        # Sum over completion tokens
        seq_logps = (per_token_logps * loss_mask).sum(dim=-1)

        if average_log_prob:
            token_counts = loss_mask.sum(dim=-1).clamp(min=1)
            seq_logps = seq_logps / token_counts

        return seq_logps

    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> tuple[torch.Tensor, DPOTelemetry]:
        """
        Calculates the DPO objective:
            L_DPO = -E[log sigma(beta * (log pi_theta(y_w|x)/pi_ref(y_w|x) - log pi_theta(y_l|x)/pi_ref(y_l|x)))]
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = self.config.beta * (pi_logratios - ref_logratios)

        # Implicit rewards
        chosen_rewards = self.config.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = (
            self.config.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )

        # DPO loss with optional label smoothing
        if self.config.label_smoothing > 0.0:
            loss = (
                -F.logsigmoid(logits) * (1 - self.config.label_smoothing)
                - F.logsigmoid(-logits) * self.config.label_smoothing
            ).mean()
        else:
            loss = -F.logsigmoid(logits).mean()

        # Accuracy: fraction where chosen implicit reward > rejected implicit reward
        with torch.no_grad():
            acc = (chosen_rewards > rejected_rewards).float().mean().item()
            mean_c = chosen_rewards.mean().item()
            mean_r = rejected_rewards.mean().item()
            margin = (chosen_rewards - rejected_rewards).mean().item()
            p_c = policy_chosen_logps.mean().item()
            p_r = policy_rejected_logps.mean().item()

        telemetry = DPOTelemetry(
            loss=round(loss.item(), 4),
            accuracy=round(acc, 4),
            chosen_implicit_reward=round(mean_c, 4),
            rejected_implicit_reward=round(mean_r, 4),
            reward_margin=round(margin, 4),
            policy_chosen_logp=round(p_c, 4),
            policy_rejected_logp=round(p_r, 4),
        )

        return loss, telemetry

    def train_step(self, batch: dict[str, torch.Tensor]) -> DPOTelemetry:
        """Executes a single optimization step on a preference batch."""
        self.policy_model.train()
        device = next(self.policy_model.parameters()).device

        chosen_ids = batch["chosen_input_ids"].to(device)
        chosen_labels = batch["chosen_labels"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        rejected_labels = batch["rejected_labels"].to(device)

        # 1. Forward passes through Policy Model
        policy_chosen_logits, _ = self.policy_model(chosen_ids)
        policy_rejected_logits, _ = self.policy_model(rejected_ids)

        policy_chosen_logps = self.get_batch_logps(policy_chosen_logits, chosen_labels)
        policy_rejected_logps = self.get_batch_logps(policy_rejected_logits, rejected_labels)

        # 2. Forward passes through frozen Reference Model
        with torch.no_grad():
            ref_chosen_logits, _ = self.reference_model(chosen_ids)
            ref_rejected_logits, _ = self.reference_model(rejected_ids)

            ref_chosen_logps = self.get_batch_logps(ref_chosen_logits, chosen_labels)
            ref_rejected_logps = self.get_batch_logps(ref_rejected_logits, rejected_labels)

        # 3. Compute DPO Loss
        loss, telemetry = self.compute_dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=ref_chosen_logps,
            reference_rejected_logps=ref_rejected_logps,
        )

        # 4. Backward Pass & Gradient Clipping
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(), self.config.max_grad_norm
            )
        self.optimizer.step()

        return telemetry
