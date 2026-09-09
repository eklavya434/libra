"""
Libra Core Observability - Token Velocity & System Latency Metrics

Computes:
1. Time to First Token (TTFT) in milliseconds.
2. Inter-Token Latency (ITL) distribution: mean, median (p50), and 95th percentile (p95).
3. Token Generation Velocity (tokens/second).
4. System-wide rolling latency and throughput percentiles.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Optional


@dataclass
class TokenVelocityMetrics:
    """Detailed velocity and timing metrics for a token generation stream."""

    start_time_ms: float = field(default_factory=lambda: time.perf_counter() * 1000)
    first_token_time_ms: Optional[float] = None
    last_token_time_ms: Optional[float] = None
    tokens_generated: int = 0
    inter_token_latencies_ms: list[float] = field(default_factory=list)

    def record_token(self) -> None:
        """Records the arrival of a newly generated token."""
        now_ms = time.perf_counter() * 1000
        self.tokens_generated += 1

        if self.first_token_time_ms is None:
            self.first_token_time_ms = now_ms
            self.last_token_time_ms = now_ms
            return

        delta = round(now_ms - (self.last_token_time_ms or self.start_time_ms), 3)
        self.inter_token_latencies_ms.append(delta)
        self.last_token_time_ms = now_ms

    @property
    def ttft_ms(self) -> float:
        if self.first_token_time_ms is None:
            return 0.0
        return round(self.first_token_time_ms - self.start_time_ms, 2)

    @property
    def total_latency_ms(self) -> float:
        end = self.last_token_time_ms or (time.perf_counter() * 1000)
        return round(end - self.start_time_ms, 2)

    @property
    def tokens_per_second(self) -> float:
        duration_sec = self.total_latency_ms / 1000.0
        if duration_sec <= 0.0 or self.tokens_generated == 0:
            return 0.0
        return round(self.tokens_generated / duration_sec, 2)

    @property
    def itl_mean_ms(self) -> float:
        if not self.inter_token_latencies_ms:
            return 0.0
        return round(statistics.mean(self.inter_token_latencies_ms), 2)

    @property
    def itl_p50_ms(self) -> float:
        if not self.inter_token_latencies_ms:
            return 0.0
        return round(statistics.median(self.inter_token_latencies_ms), 2)

    @property
    def itl_p95_ms(self) -> float:
        if not self.inter_token_latencies_ms:
            return 0.0
        sorted_itls = sorted(self.inter_token_latencies_ms)
        idx = int(0.95 * (len(sorted_itls) - 1))
        return round(sorted_itls[idx], 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens_generated": self.tokens_generated,
            "ttft_ms": self.ttft_ms,
            "total_latency_ms": self.total_latency_ms,
            "tokens_per_second": self.tokens_per_second,
            "itl_mean_ms": self.itl_mean_ms,
            "itl_p50_ms": self.itl_p50_ms,
            "itl_p95_ms": self.itl_p95_ms,
        }


class MetricsAggregator:
    """Thread-safe rolling metrics collector aggregating system throughput and percentiles."""

    def __init__(self, max_history: int = 500) -> None:
        self.max_history = max_history
        self._latencies: list[float] = []
        self._ttfts: list[float] = []
        self._tps_values: list[float] = []
        self._total_requests: int = 0
        self._total_errors: int = 0
        self._lock = Lock()

    def record_request(
        self,
        duration_ms: float,
        is_error: bool = False,
        ttft_ms: Optional[float] = None,
        tokens_per_second: Optional[float] = None,
    ) -> None:
        with self._lock:
            self._total_requests += 1
            if is_error:
                self._total_errors += 1

            self._latencies.append(duration_ms)
            if len(self._latencies) > self.max_history:
                self._latencies.pop(0)

            if ttft_ms is not None and ttft_ms > 0:
                self._ttfts.append(ttft_ms)
                if len(self._ttfts) > self.max_history:
                    self._ttfts.pop(0)

            if tokens_per_second is not None and tokens_per_second > 0:
                self._tps_values.append(tokens_per_second)
                if len(self._tps_values) > self.max_history:
                    self._tps_values.pop(0)

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            if not self._latencies:
                return {
                    "total_requests": self._total_requests,
                    "total_errors": self._total_errors,
                    "error_rate": 0.0,
                    "avg_latency_ms": 0.0,
                    "p50_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "avg_ttft_ms": 0.0,
                    "avg_tokens_per_second": 0.0,
                }

            sorted_lat = sorted(self._latencies)
            p50_idx = int(0.50 * (len(sorted_lat) - 1))
            p95_idx = int(0.95 * (len(sorted_lat) - 1))

            avg_ttft = statistics.mean(self._ttfts) if self._ttfts else 0.0
            avg_tps = statistics.mean(self._tps_values) if self._tps_values else 0.0
            err_rate = (
                (self._total_errors / self._total_requests) if self._total_requests > 0 else 0.0
            )

            return {
                "total_requests": self._total_requests,
                "total_errors": self._total_errors,
                "error_rate": round(err_rate, 4),
                "avg_latency_ms": round(statistics.mean(self._latencies), 2),
                "p50_latency_ms": round(sorted_lat[p50_idx], 2),
                "p95_latency_ms": round(sorted_lat[p95_idx], 2),
                "avg_ttft_ms": round(avg_ttft, 2),
                "avg_tokens_per_second": round(avg_tps, 2),
            }

    def reset(self) -> None:
        with self._lock:
            self._latencies.clear()
            self._ttfts.clear()
            self._tps_values.clear()
            self._total_requests = 0
            self._total_errors = 0


_global_metrics_aggregator = MetricsAggregator()


def get_metrics_aggregator() -> MetricsAggregator:
    return _global_metrics_aggregator
