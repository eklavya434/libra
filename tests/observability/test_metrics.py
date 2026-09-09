"""
Tests for TokenVelocityMetrics & MetricsAggregator (packages/core/observability/metrics.py)
"""

import time

from packages.core.observability.metrics import (
    MetricsAggregator,
    TokenVelocityMetrics,
)


def test_token_velocity_metrics_calculation():
    metrics = TokenVelocityMetrics()
    assert metrics.tokens_generated == 0

    # Simulate token stream arrival
    time.sleep(0.02)
    metrics.record_token()  # First token (TTFT)
    assert metrics.ttft_ms >= 15.0

    time.sleep(0.01)
    metrics.record_token()

    time.sleep(0.01)
    metrics.record_token()

    assert metrics.tokens_generated == 3
    assert len(metrics.inter_token_latencies_ms) == 2
    assert metrics.tokens_per_second > 0
    assert metrics.itl_mean_ms >= 5.0
    assert metrics.itl_p50_ms >= 5.0

    summary = metrics.to_dict()
    assert summary["tokens_generated"] == 3
    assert "ttft_ms" in summary
    assert "tokens_per_second" in summary


def test_metrics_aggregator_rolling_summary():
    aggregator = MetricsAggregator(max_history=10)

    # Record 5 requests with varying latencies
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
    for lat in latencies:
        aggregator.record_request(
            duration_ms=lat,
            is_error=False,
            ttft_ms=5.0,
            tokens_per_second=25.0,
        )

    # Record 1 error
    aggregator.record_request(duration_ms=100.0, is_error=True)

    summary = aggregator.get_summary()
    assert summary["total_requests"] == 6
    assert summary["total_errors"] == 1
    assert round(summary["error_rate"], 2) == 0.17
    assert summary["p50_latency_ms"] >= 20.0
    assert summary["avg_ttft_ms"] == 5.0
    assert summary["avg_tokens_per_second"] == 25.0

    # Reset
    aggregator.reset()
    empty_summary = aggregator.get_summary()
    assert empty_summary["total_requests"] == 0
