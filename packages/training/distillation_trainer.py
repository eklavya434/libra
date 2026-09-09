"""
Libra Training - Knowledge Distillation Trainer
Implements temperature-scaled KL divergence soft loss, hard label cross-entropy,
and model shrinking supervision from teacher to student models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import nn, optim

from packages.models.components.model_shrinking import ModelShrinker
from packages.models.modern_transformer import ModernTransformerLM


class DistillationTelemetry(BaseModel):
    """Telemetry metrics emitted during a knowledge distillation training step."""

    step: int
    total_loss: float
    soft_loss: float
    hard_loss: float
    hidden_loss: float = 0.0
    kl_div: float
    top1_agreement: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of tokens where student top-1 argmax matches teacher",
    )
    student_ppl: float = Field(..., description="Student cross-entropy perplexity")
    teacher_ppl: float = Field(..., description="Teacher cross-entropy perplexity")


@dataclass
class DistillationConfig:
    """Hyperparameters for student-teacher knowledge distillation."""

    temperature: float = 2.0  # Softmax temperature tau
    alpha: float = 0.5  # Weight for soft loss; (1 - alpha) for hard loss
    lr: float = 1e-3  # Student learning rate
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    hidden_loss_weight: float = 0.0  # Optional intermediate layer alignment weight
    intermediate_layers: list[tuple[int, int]] | None = None  # (student_layer, teacher_layer)


class DistillationTrainer:
    """Orchestrates knowledge distillation from a teacher model into a compact student model."""

    def __init__(
        self,
        student_model: ModernTransformerLM,
        teacher_model: ModernTransformerLM,
        config: DistillationConfig | None = None,
        optimizer: optim.Optimizer | None = None,
    ) -> None:
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.config = config or DistillationConfig()

        if not (0.0 <= self.config.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.config.alpha}")
        if self.config.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.config.temperature}")

        # Strictly freeze teacher model in evaluation mode
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.student_model.train()
        self.optimizer = optimizer or optim.AdamW(
            self.student_model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # Projection layer for intermediate hidden state alignment if dimensions differ
        self.hidden_projectors: nn.ModuleDict = nn.ModuleDict()
        if (
            self.config.hidden_loss_weight > 0.0
            and self.student_model.config.d_model != self.teacher_model.config.d_model
        ):
            proj = nn.Linear(
                self.student_model.config.d_model,
                self.teacher_model.config.d_model,
                bias=False,
            )
            self.hidden_projectors["proj"] = proj
            self.optimizer.add_param_group({"params": proj.parameters()})

        self.current_step = 0

    @staticmethod
    def compute_soft_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute temperature-scaled KL divergence soft loss.

        Loss = tau^2 * KL(softmax(z_t / tau) || log_softmax(z_s / tau))
        Returns:
            (soft_loss_scaled, unscaled_kl_divergence)
        """
        # student log-probabilities: (B, T, V)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        # teacher probabilities: (B, T, V)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        # PyTorch kl_div: input is log_probs, target is probs
        kl_div = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        soft_loss = kl_div * (temperature**2)
        return soft_loss, kl_div

    @staticmethod
    def compute_hard_loss(
        student_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute standard cross-entropy loss against ground truth targets."""
        return F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )

    @staticmethod
    def compute_top1_agreement(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> float:
        """Compute fraction of tokens where student top-1 argmax prediction equals teacher top-1 argmax."""
        s_preds = torch.argmax(student_logits, dim=-1)
        t_preds = torch.argmax(teacher_logits, dim=-1)

        if targets is not None:
            mask = targets != -100
            if mask.sum() > 0:
                matching = (s_preds == t_preds) & mask
                return float(matching.sum().item() / mask.sum().item())

        matching = s_preds == t_preds
        return float(matching.sum().item() / max(1, matching.numel()))

    def compute_intermediate_loss(
        self,
        student_hidden_states: list[torch.Tensor],
        teacher_hidden_states: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute MSE loss between student and teacher intermediate hidden states."""
        if not self.config.intermediate_layers:
            # Default: compare last layer hidden states
            s_h = student_hidden_states[-1]
            t_h = teacher_hidden_states[-1]
            if "proj" in self.hidden_projectors:
                s_h = self.hidden_projectors["proj"](s_h)
            return F.mse_loss(s_h, t_h)

        losses = []
        for s_idx, t_idx in self.config.intermediate_layers:
            if s_idx < len(student_hidden_states) and t_idx < len(teacher_hidden_states):
                s_h = student_hidden_states[s_idx]
                t_h = teacher_hidden_states[t_idx]
                if "proj" in self.hidden_projectors:
                    s_h = self.hidden_projectors["proj"](s_h)
                losses.append(F.mse_loss(s_h, t_h))

        return sum(losses) / max(1, len(losses)) if losses else torch.tensor(0.0)

    def train_step(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
    ) -> DistillationTelemetry:
        """Execute a single distillation optimization step."""
        self.student_model.train()
        self.teacher_model.eval()

        # 1. Forward teacher with no_grad
        with torch.no_grad():
            if self.config.hidden_loss_weight > 0.0:
                teacher_logits, teacher_hiddens, _ = ModelShrinker.forward_with_hidden_states(
                    self.teacher_model, input_ids
                )
            else:
                teacher_logits, _ = self.teacher_model(input_ids)
                teacher_hiddens = []

            teacher_ce = F.cross_entropy(
                teacher_logits.view(-1, teacher_logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            teacher_ppl = math.exp(min(20.0, teacher_ce.item()))

        # 2. Forward student
        if self.config.hidden_loss_weight > 0.0:
            student_logits, student_hiddens, _ = ModelShrinker.forward_with_hidden_states(
                self.student_model, input_ids
            )
        else:
            student_logits, _ = self.student_model(input_ids)
            student_hiddens = []

        # 3. Compute Soft Loss & KL Divergence
        soft_loss, kl_div = self.compute_soft_loss(
            student_logits, teacher_logits, self.config.temperature
        )

        # 4. Compute Hard Cross-Entropy Loss
        hard_loss = self.compute_hard_loss(student_logits, targets)

        # 5. Compute Hidden State Alignment Loss
        hidden_loss = torch.tensor(0.0)
        if self.config.hidden_loss_weight > 0.0 and teacher_hiddens and student_hiddens:
            hidden_loss = self.compute_intermediate_loss(student_hiddens, teacher_hiddens)

        # 6. Composite Loss
        total_loss = (
            self.config.alpha * soft_loss
            + (1.0 - self.config.alpha) * hard_loss
            + self.config.hidden_loss_weight * hidden_loss
        )

        # 7. Backward pass & optimize
        self.optimizer.zero_grad()
        total_loss.backward()

        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.student_model.parameters(), self.config.max_grad_norm
            )

        self.optimizer.step()
        self.current_step += 1

        # 8. Compute Telemetry
        top1_agreement = self.compute_top1_agreement(student_logits, teacher_logits, targets)
        student_ppl = math.exp(min(20.0, hard_loss.item()))

        return DistillationTelemetry(
            step=self.current_step,
            total_loss=round(float(total_loss.item()), 4),
            soft_loss=round(float(soft_loss.item()), 4),
            hard_loss=round(float(hard_loss.item()), 4),
            hidden_loss=round(float(hidden_loss.item()), 4),
            kl_div=round(float(kl_div.item()), 4),
            top1_agreement=round(float(top1_agreement), 4),
            student_ppl=round(float(student_ppl), 4),
            teacher_ppl=round(float(teacher_ppl), 4),
        )

    def train_sequence(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        steps: int = 10,
    ) -> list[DistillationTelemetry]:
        """Train student on given tensor sequence for specified number of steps."""
        history = []
        for _ in range(steps):
            telemetry = self.train_step(input_ids, targets)
            history.append(telemetry)
        return history
