"""
Libra Evaluation - Arena Statistical Metrics Helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ArenaModelMetric:
    model: str
    provider: str
    success: bool
    output_text: str
    ttft_ms: Optional[float]
    total_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tokens_per_second: float
    cost_usd: float
    is_free: bool
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "success": self.success,
            "output_text": self.output_text,
            "ttft_ms": round(self.ttft_ms, 2) if self.ttft_ms is not None else None,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "cost_usd": round(self.cost_usd, 6),
            "is_free": self.is_free,
            "error": self.error,
        }

