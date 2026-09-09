"""
Tests for Automated CPU Performance Regression Gate (Phase 35)
"""

from packages.evaluation.regression_gate import PerformanceRegressionGate, RegressionReport


def test_regression_gate_benchmark_metrics():
    gate = PerformanceRegressionGate()
    metrics = gate.benchmark_model()

    assert "forward_latency_ms" in metrics
    assert "tokens_per_sec" in metrics
    assert "loss_before" in metrics
    assert "loss_after" in metrics
    assert "loss_decreased" in metrics

    # Basic physical sanity
    assert metrics["forward_latency_ms"] > 0.0
    assert metrics["tokens_per_sec"] > 0.0
    assert metrics["loss_decreased"] is True


def test_regression_gate_run_pass():
    # Set permissive thresholds to ensure normal environment passes
    gate = PerformanceRegressionGate(
        min_tokens_per_sec=1.0,
        max_forward_latency_ms=1000.0,
        max_disk_mb=20000.0,
    )
    report = gate.run_gate()

    assert isinstance(report, RegressionReport)
    assert report.passed is True
    assert len(report.violations) == 0
    assert "Throughput:" in report.summary()


def test_regression_gate_threshold_violation():
    # Set impossibly strict threshold to verify regression detection
    strict_gate = PerformanceRegressionGate(
        min_tokens_per_sec=100000.0,  # Impossible on CPU
    )
    report = strict_gate.run_gate()

    assert report.passed is False
    assert len(report.violations) > 0
    assert any("Throughput regression" in v for v in report.violations)
