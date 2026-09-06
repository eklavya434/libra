"""
Libra Training - Production-Grade Training Engine
Features:
  1. Gradient Accumulation (simulating larger batch sizes on CPU)
  2. Cosine Learning Rate Schedule with Linear Warmup
  3. Gradient Norm Clipping (preventing gradient explosion)
  4. Real Perplexity and Token Velocity Tracking
  5. Checkpoint Resumption (pause and recover training seamlessly)
  6. CPU Budget Enforcement (< 15 minutes)
"""

import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
import yaml
from torch import nn, optim

from packages.training.dataset import TextDataset
from packages.training.metrics import ThroughputTracker, TrainingLogEntry, compute_perplexity
from packages.training.scheduler import CosineWarmupScheduler


@dataclass
class TrainingConfig:
    max_steps: int = 200
    batch_size: int = 4
    gradient_accumulation_steps: int = 2  # Effective batch size = 8
    max_lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 25
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    eval_interval: int = 50
    eval_batches: int = 5
    checkpoint_dir: str = "checkpoints"
    device: str = "cpu"
    cpu_budget_seconds: float = 900.0  # 15 minute non-negotiable ceiling

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


class TrainingEngine:
    """Production Training Engine for Libra models."""

    def __init__(
        self,
        model: nn.Module,
        dataset: TextDataset,
        config: TrainingConfig,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.config = config

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.max_lr,
            weight_decay=config.weight_decay,
        )

        self.scheduler = CosineWarmupScheduler(
            max_lr=config.max_lr,
            min_lr=config.min_lr,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
        )

        self.tracker = ThroughputTracker()
        self.history: list[TrainingLogEntry] = []
        self.best_val_loss: float = float("inf")
        self.current_step: int = 0

    def _get_block_size(self) -> int:
        """Determines safe sequence length adhering to model context limits."""
        if hasattr(self.model, "config") and hasattr(self.model.config, "max_context_length"):
            return min(self.model.config.max_context_length, 64)
        return 64

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        """Evaluates model on validation split and returns (val_loss, val_ppl)."""
        self.model.eval()
        total_loss = 0.0
        batches = self.config.eval_batches
        block_size = self._get_block_size()

        for _ in range(batches):
            xb, yb = self.dataset.get_batch(
                split="val",
                batch_size=self.config.batch_size,
                block_size=block_size,
                device=self.config.device,
            )
            # Clamp in case of vocabulary boundaries
            if hasattr(self.model, "config") and hasattr(self.model.config, "vocab_size"):
                xb = torch.clamp(xb, max=self.model.config.vocab_size - 1)
                yb = torch.clamp(yb, max=self.model.config.vocab_size - 1)

            _, loss = self.model(xb, yb)
            if loss is not None:
                total_loss += loss.item()

        avg_loss = total_loss / max(1, batches)
        avg_ppl = compute_perplexity(avg_loss)
        self.model.train()
        return round(avg_loss, 4), avg_ppl

    def train(self, start_step: int = 1) -> list[TrainingLogEntry]:
        """Runs the main training loop with gradient accumulation and budget tracking."""
        self.model.train()
        start_time = time.time()
        block_size = self._get_block_size()
        print(
            f"[TrainingEngine] Starting run from step {start_step} to {self.config.max_steps} on {self.config.device}..."
        )
        print(
            f"[TrainingEngine] Effective Batch Size: {self.config.effective_batch_size} "
            f"({self.config.batch_size} micro-batch x {self.config.gradient_accumulation_steps} accum steps)"
        )

        for step in range(start_step, self.config.max_steps + 1):
            self.current_step = step

            # 1. Update Learning Rate from Scheduler
            lr = self.scheduler.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # 2. Gradient Accumulation Loop
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for _micro in range(self.config.gradient_accumulation_steps):
                xb, yb = self.dataset.get_batch(
                    split="train",
                    batch_size=self.config.batch_size,
                    block_size=block_size,
                    device=self.config.device,
                )
                if hasattr(self.model, "config") and hasattr(self.model.config, "vocab_size"):
                    xb = torch.clamp(xb, max=self.model.config.vocab_size - 1)
                    yb = torch.clamp(yb, max=self.model.config.vocab_size - 1)

                _, loss = self.model(xb, yb)
                assert loss is not None

                # Scale loss by accumulation steps
                loss_scaled = loss / self.config.gradient_accumulation_steps
                loss_scaled.backward()
                accum_loss += loss.item()

                # Track processed tokens
                self.tracker.update(xb.numel())

            avg_train_loss = accum_loss / self.config.gradient_accumulation_steps
            train_ppl = compute_perplexity(avg_train_loss)

            # 3. Gradient Norm Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            # 4. Optimizer Step
            self.optimizer.step()

            # 5. Check CPU Budget (< 15 minutes)
            elapsed = time.time() - start_time
            if elapsed > self.config.cpu_budget_seconds:
                print(
                    f"[TrainingEngine WARNING] Reached CPU budget limit ({elapsed:.1f}s). Stopping run."
                )
                break

            # 6. Periodic Evaluation & Logging
            if step % self.config.eval_interval == 0 or step == self.config.max_steps:
                val_loss, val_ppl = self.evaluate()
                entry = TrainingLogEntry(
                    step=step,
                    train_loss=round(avg_train_loss, 4),
                    train_ppl=train_ppl,
                    val_loss=val_loss,
                    val_ppl=val_ppl,
                    learning_rate=round(lr, 6),
                    tokens_per_sec=self.tracker.tokens_per_sec,
                    total_tokens=self.tracker.total_tokens,
                    elapsed_seconds=round(elapsed, 2),
                )
                self.history.append(entry)
                print(
                    f"[{step:04d}/{self.config.max_steps}] "
                    f"Train Loss: {entry.train_loss:.4f} (PPL: {entry.train_ppl:.1f}) | "
                    f"Val Loss: {entry.val_loss:.4f} (PPL: {entry.val_ppl:.1f}) | "
                    f"LR: {entry.learning_rate:.6f} | "
                    f"Speed: {entry.tokens_per_sec:,.0f} tok/s | "
                    f"Elapsed: {entry.elapsed_seconds:.1f}s"
                )

                # Save checkpoint if better validation loss or at the end
                if val_loss < self.best_val_loss or step == self.config.max_steps:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        step, os.path.join(self.config.checkpoint_dir, "best_engine_model.pt")
                    )

        return self.history

    def save_checkpoint(self, step: int, path: str) -> None:
        """Saves complete training state for pause/resume."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_config = self.model.config.to_dict() if hasattr(self.model, "config") else {}

        checkpoint = {
            "step": step,
            "model_config": model_config,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_config": self.config.to_dict(),
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, path)

    def resume_from_checkpoint(self, path: str) -> int:
        """Restores state from checkpoint and returns the step to resume from."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        checkpoint = torch.load(path, map_location=self.config.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        resumed_step = checkpoint["step"]
        print(f"[TrainingEngine] Resumed state from {path} at step {resumed_step}.")
        return resumed_step + 1
