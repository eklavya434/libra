"""Evaluation module for loss, perplexity, and compression metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from packages.data.pipeline import BinaryDataset


@dataclass
class LossMetrics:
    """Quantitative loss and compression metrics for an evaluated corpus."""

    total_tokens: int
    mean_loss: float
    perplexity: float
    bits_per_token: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_tokens": self.total_tokens,
            "mean_loss": round(self.mean_loss, 4),
            "perplexity": round(self.perplexity, 4),
            "bits_per_token": round(self.bits_per_token, 4),
        }


def compute_perplexity(loss: float) -> float:
    """Compute perplexity: PPL = exp(loss). Clamps to 1e9 on overflow."""
    try:
        return math.exp(loss)
    except OverflowError:
        return 1e9


def compute_bits_per_token(loss: float) -> float:
    """Compute bits per token: BPT = loss / ln(2)."""
    return loss / math.log(2.0)


@torch.no_grad()
def evaluate_tokens_loss(
    model: nn.Module,
    token_ids: torch.Tensor,
    context_length: Optional[int] = None,
    stride: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> LossMetrics:
    """Evaluate cross-entropy loss and perplexity on a continuous 1D token sequence.

    Uses a sliding-window stride evaluation when the token sequence exceeds
    the model's maximum context length, ensuring proper context conditioning.

    Args:
        model: Autoregressive language model (takes input_ids, returns logits or (logits, loss)).
        token_ids: 1D tensor of token IDs.
        context_length: Max context length for the window. Defaults to model config.
        stride: Step size between evaluation windows. Defaults to context_length // 2.
        device: Device to run evaluation on.

    Returns:
        LossMetrics dataclass containing total_tokens, mean_loss, perplexity, bits_per_token.
    """
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    token_ids = token_ids.to(device)
    if token_ids.dim() != 1:
        token_ids = token_ids.view(-1)

    total_seq_len = token_ids.size(0)
    if total_seq_len < 2:
        return LossMetrics(total_tokens=0, mean_loss=0.0, perplexity=1.0, bits_per_token=0.0)

    # Infer max context length if not provided
    if context_length is None:
        if hasattr(model, "config") and hasattr(model.config, "max_context_length"):
            context_length = model.config.max_context_length
        elif hasattr(model, "config") and hasattr(model.config, "max_seq_len"):
            context_length = model.config.max_seq_len
        else:
            context_length = 128

    if stride is None:
        stride = max(1, context_length // 2)

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_nll = 0.0
    total_eval_tokens = 0

    for start_idx in range(0, total_seq_len - 1, stride):
        end_idx = min(start_idx + context_length, total_seq_len)
        chunk = token_ids[start_idx:end_idx].unsqueeze(0)  # (1, seq_len)

        inputs = chunk[:, :-1]
        targets = chunk[:, 1:]

        output = model(inputs)
        logits = output[0] if isinstance(output, tuple) else output

        # Shift evaluation to only count new tokens in the stride window to avoid double-counting
        if start_idx == 0:
            target_slice = targets
            logits_slice = logits
        else:
            new_tokens_count = end_idx - (start_idx + stride)
            if new_tokens_count <= 0:
                break
            target_slice = targets[:, -new_tokens_count:]
            logits_slice = logits[:, -new_tokens_count:, :]

        batch_size, seq_len, vocab_size = logits_slice.shape
        loss = loss_fn(
            logits_slice.reshape(batch_size * seq_len, vocab_size),
            target_slice.reshape(batch_size * seq_len),
        )

        num_tokens = target_slice.numel()
        total_nll += loss.item()
        total_eval_tokens += num_tokens

        if end_idx == total_seq_len:
            break

    if total_eval_tokens == 0:
        return LossMetrics(total_tokens=0, mean_loss=0.0, perplexity=1.0, bits_per_token=0.0)

    mean_loss = total_nll / total_eval_tokens
    ppl = compute_perplexity(mean_loss)
    bpt = compute_bits_per_token(mean_loss)

    return LossMetrics(
        total_tokens=total_eval_tokens,
        mean_loss=mean_loss,
        perplexity=ppl,
        bits_per_token=bpt,
    )


@torch.no_grad()
def evaluate_dataset_loss(
    model: nn.Module,
    dataset: BinaryDataset,
    num_batches: int = 20,
    batch_size: int = 4,
    device: Optional[torch.device] = None,
) -> LossMetrics:
    """Evaluate cross-entropy loss and perplexity across batches sampled from a BinaryDataset."""
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    block_size = 64
    if hasattr(model, "config") and hasattr(model.config, "max_context_length"):
        block_size = min(model.config.max_context_length, 64)
    elif hasattr(model, "config") and hasattr(model.config, "max_seq_len"):
        block_size = min(model.config.max_seq_len, 64)

    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(num_batches):
        x, y = dataset.get_batch(batch_size=batch_size, block_size=block_size, device=device)
        output = model(x)
        logits = output[0] if isinstance(output, tuple) else output
        B, T, V = logits.shape
        loss = loss_fn(logits.view(B * T, V), y.view(B * T))
        total_loss += loss.item() * (B * T)
        total_tokens += B * T

    if total_tokens == 0:
        return LossMetrics(total_tokens=0, mean_loss=0.0, perplexity=1.0, bits_per_token=0.0)

    mean_loss = total_loss / total_tokens
    ppl = compute_perplexity(mean_loss)
    bpt = compute_bits_per_token(mean_loss)

    return LossMetrics(
        total_tokens=total_tokens,
        mean_loss=mean_loss,
        perplexity=ppl,
        bits_per_token=bpt,
    )
