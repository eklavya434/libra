"""
Libra Training - Parameter-Efficient LoRA Trainer
Phase 25: Parameter-Efficient Fine-Tuning (PEFT & LoRA)

Trains ONLY low-rank adapter weights (lora_A, lora_B) while keeping the base
transformer frozen, cutting optimizer memory footprint by 95%+ and running
efficiently on consumer CPU.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from packages.models.lora.lora_model import get_lora_parameter_summary


class LoRATrainer:
    """
    Lightweight training engine for LoRA adapters.
    Optimizes only parameters with requires_grad=True using AdamW.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
    ) -> None:
        self.model = model
        self.learning_rate = learning_rate

        # Filter only trainable adapter parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("Model has no trainable parameters! Did you forget to apply_lora()?")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.param_summary = get_lora_parameter_summary(model)

    def train_step(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> dict[str, Any]:
        """
        Executes a single forward-backward optimization step on a training batch.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        logits, loss = self.model(input_ids, targets=target_ids)
        if loss is None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        # Backward pass (gradients flow exclusively to lora_A and lora_B)
        loss.backward()

        # Compute gradient norm across trainable params
        total_norm = 0.0
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        for p in trainable_params:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm**2
        total_norm = total_norm**0.5

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

        # Optimizer update
        self.optimizer.step()

        return {
            "loss": round(loss.item(), 4),
            "grad_norm": round(total_norm, 4),
            "trainable_parameters": self.param_summary["trainable_parameters"],
            "trainable_percentage": self.param_summary["trainable_percentage"],
        }

    def train_epochs(
        self,
        dataset: list[tuple[torch.Tensor, torch.Tensor]],
        epochs: int = 3,
    ) -> dict[str, Any]:
        """Runs multiple training iterations over a small dataset."""
        start_time = time.perf_counter()
        history = []

        for epoch in range(epochs):
            epoch_losses = []
            for x, y in dataset:
                step_res = self.train_step(x, y)
                epoch_losses.append(step_res["loss"])

            avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            history.append(
                {
                    "epoch": epoch + 1,
                    "avg_loss": round(avg_loss, 4),
                }
            )

        elapsed = time.perf_counter() - start_time
        return {
            "epochs": epochs,
            "final_loss": history[-1]["avg_loss"] if history else None,
            "elapsed_seconds": round(elapsed, 3),
            "history": history,
            "param_summary": self.param_summary,
        }
