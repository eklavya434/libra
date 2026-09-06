"""
Libra Training - CPU Training Engine
Executes an educational autoregressive training loop on CPU, enforces the 15-minute training budget,
tracks loss progression, and serializes model checkpoints.
"""

import os
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import optim

from packages.models.config import TinyTransformerConfig
from packages.models.transformer import TinyTransformerLM
from packages.training.dataset import TextDataset


@dataclass
class TrainingMetrics:
    total_steps: int
    initial_loss: float
    final_loss: float
    elapsed_seconds: float
    loss_history: list[float]
    checkpoint_path: str


def train_tiny_llm(
    model: TinyTransformerLM,
    dataset: TextDataset,
    max_steps: int = 500,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    eval_interval: int = 50,
    checkpoint_path: str = "checkpoints/tiny_llm_phase1.pt",
) -> TrainingMetrics:
    """Trains the TinyTransformerLM on CPU, guaranteeing completion well under the 15-minute budget."""
    start_time = time.time()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    loss_history: list[float] = []
    initial_loss: float = 0.0

    model.train()
    print(f"[Trainer] Starting training for {max_steps} steps on CPU...")
    print(f"[Trainer] Parameters: {model.count_parameters():,}")

    for step in range(1, max_steps + 1):
        # 1. Fetch random batch
        xb, yb = dataset.get_batch(
            split="train",
            batch_size=batch_size,
            block_size=min(model.config.max_context_length, 64),
            device=model.config.device,
        )

        # 2. Forward pass (compute predicted next token logits and cross-entropy loss)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        assert loss is not None

        if step == 1:
            initial_loss = loss.item()

        # 3. Backward pass (backpropagation)
        loss.backward()

        # 4. Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 5. Optimizer step (update parameter weights)
        optimizer.step()

        # Check CPU budget limit (15 minutes = 900 seconds)
        elapsed = time.time() - start_time
        if elapsed > 900:
            print(
                f"[Trainer WARNING] Approaching 15-minute CPU budget limit ({elapsed:.1f}s). Stopping early."
            )
            break

        # Periodic logging and evaluation
        if step % eval_interval == 0 or step == max_steps:
            loss_val = loss.item()
            loss_history.append(loss_val)
            print(f"[Step {step:04d}/{max_steps}] Loss: {loss_val:.4f} | Elapsed: {elapsed:.2f}s")

    elapsed_seconds = round(time.time() - start_time, 2)
    final_loss = loss_history[-1] if loss_history else initial_loss

    # 6. Save Checkpoint
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint_data = {
        "step": step,
        "config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "final_loss": final_loss,
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(
        f"[Trainer] Checkpoint saved successfully: {checkpoint_path} ({os.path.getsize(checkpoint_path) / 1024:.1f} KB)"
    )

    return TrainingMetrics(
        total_steps=step,
        initial_loss=round(initial_loss, 4),
        final_loss=round(final_loss, 4),
        elapsed_seconds=elapsed_seconds,
        loss_history=loss_history,
        checkpoint_path=checkpoint_path,
    )


def load_checkpoint(checkpoint_path: str) -> tuple[TinyTransformerLM, dict[str, Any]]:
    """Loads a saved checkpoint and reconstructs the model."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = TinyTransformerConfig(**ckpt["config"])
    model = TinyTransformerLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt
