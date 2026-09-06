"""
Libra Training - Learning Rate Scheduler with Cosine Decay and Linear Warmup
Reference: Loshchilov & Hutter, 2016 ("SGDR: Stochastic Gradient Descent with Warm Restarts")

Schedules learning rate across 3 stages:
  1. Linear Warmup: Ramp up lr from 0 to max_lr to avoid gradient shock at initialization.
  2. Cosine Annealing: Smoothly decay lr from max_lr to min_lr following half of a cosine wave.
  3. Minimum Floor: Maintain min_lr after max_steps to avoid complete parameter freezing.
"""

import math
from typing import Any


class CosineWarmupScheduler:
    """Cosine learning rate scheduler with linear warmup."""

    def __init__(
        self,
        max_lr: float = 1e-3,
        min_lr: float = 1e-4,
        warmup_steps: int = 50,
        max_steps: int = 500,
    ) -> None:
        if warmup_steps > max_steps:
            raise ValueError(f"warmup_steps ({warmup_steps}) cannot exceed max_steps ({max_steps})")

        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def get_lr(self, step: int) -> float:
        """Returns the learning rate for a given step (1-indexed)."""
        # 1. Linear Warmup
        if step < self.warmup_steps:
            return self.max_lr * float(step) / float(max(1, self.warmup_steps))

        # 2. Post-decay Floor
        if step > self.max_steps:
            return self.min_lr

        # 3. Cosine Annealing Decay
        decay_ratio = float(step - self.warmup_steps) / float(
            max(1, self.max_steps - self.warmup_steps)
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def state_dict(self) -> dict[str, Any]:
        return {
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.max_lr = state["max_lr"]
        self.min_lr = state["min_lr"]
        self.warmup_steps = state["warmup_steps"]
        self.max_steps = state["max_steps"]
