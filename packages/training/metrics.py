"""
Libra Training - Metrics & Telemetry Tracking
Computes real perplexity, token processing throughput, and training logs.
"""

import math
import time
from dataclasses import asdict, dataclass
from typing import Any


def compute_perplexity(loss: float) -> float:
    """Computes Perplexity: PPL = exp(loss).

    Guarded against mathematical overflow if loss is unusually large.
    """
    try:
        # Cap loss at 20.0 to prevent math.exp overflow (exp(20) ~ 485 million)
        clamped_loss = min(loss, 20.0)
        return round(math.exp(clamped_loss), 2)
    except OverflowError:
        return float("inf")


class ThroughputTracker:
    """Tracks token processing velocity (tokens per second)."""

    def __init__(self) -> None:
        self.start_time = time.time()
        self.total_tokens = 0

    def update(self, tokens: int) -> None:
        self.total_tokens += tokens

    @property
    def elapsed_seconds(self) -> float:
        return max(1e-4, time.time() - self.start_time)

    @property
    def tokens_per_sec(self) -> float:
        return round(self.total_tokens / self.elapsed_seconds, 1)


@dataclass
class TrainingLogEntry:
    step: int
    train_loss: float
    train_ppl: float
    val_loss: float | None
    val_ppl: float | None
    learning_rate: float
    tokens_per_sec: float
    total_tokens: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
