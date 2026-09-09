"""
Libra Evaluation - Automated CPU Performance Regression Gate (Phase 35)
Guards against:
1. Token generation throughput degradation (< min_tokens_per_sec)
2. Forward pass latency spikes (> max_forward_latency_ms)
3. Broken optimization/loss calculation (loss must decrease on 1 step overfit)
4. Disk space quota violations (> 15 GB total tracked footprint)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW

from packages.models.generation import generate_with_cache
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM


@dataclass
class RegressionReport:
    passed: bool
    metrics: dict[str, Any]
    violations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Regression Gate Status: {'PASSED' if self.passed else 'FAILED'}",
            f"Throughput: {self.metrics.get('tokens_per_sec', 0.0):.2f} tokens/s (threshold: >= {self.metrics.get('min_tokens_per_sec', 0.0):.2f})",
            f"Forward Latency: {self.metrics.get('forward_latency_ms', 0.0):.2f} ms (threshold: <= {self.metrics.get('max_forward_latency_ms', 0.0):.2f})",
            f"Loss Step Sanity: {self.metrics.get('loss_before', 0.0):.4f} -> {self.metrics.get('loss_after', 0.0):.4f} (decreased: {self.metrics.get('loss_decreased', False)})",
            f"Disk Usage: {self.metrics.get('disk_used_mb', 0.0):.2f} MB / {self.metrics.get('max_disk_mb', 0.0):.2f} MB",
        ]
        if self.violations:
            lines.append("Violations detected:")
            for v in self.violations:
                lines.append(f"  - {v}")
        return "\n".join(lines)


class PerformanceRegressionGate:
    """Automated benchmark gate to prevent silent CPU performance regressions."""

    def __init__(
        self,
        min_tokens_per_sec: float = 10.0,
        max_forward_latency_ms: float = 150.0,
        max_disk_mb: float = 15360.0,
        repo_root: str | None = None,
    ) -> None:
        self.min_tokens_per_sec = min_tokens_per_sec
        self.max_forward_latency_ms = max_forward_latency_ms
        self.max_disk_mb = max_disk_mb
        self.repo_root = repo_root or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

    def benchmark_model(self) -> dict[str, float]:
        """Runs micro-benchmarks on educational ModernTransformerLM on CPU."""
        torch.manual_seed(42)
        cfg = ModernTransformerConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=2,
            n_kv_heads=1,
            max_context_length=128,
        )
        model = ModernTransformerLM(cfg)
        model.eval()

        # 1. Measure forward pass latency
        input_ids = torch.randint(0, 256, (1, 16))
        # Warmup
        with torch.no_grad():
            _ = model(input_ids)

        trials = 5
        start_t = time.perf_counter()
        with torch.no_grad():
            for _ in range(trials):
                _ = model(input_ids)
        forward_latency_ms = ((time.perf_counter() - start_t) / trials) * 1000.0

        # 2. Measure autoregressive generation throughput with KV cache
        prompt_tensor = torch.tensor([[ord(c) % 256 for c in "Libra AI"]], dtype=torch.long)
        gen_tokens_count = 16
        start_gen = time.perf_counter()
        _ = generate_with_cache(
            model=model,
            idx=prompt_tensor,
            max_new_tokens=gen_tokens_count,
            temperature=0.0,
        )
        gen_duration = time.perf_counter() - start_gen
        tokens_per_sec = gen_tokens_count / max(gen_duration, 1e-6)

        # 3. Measure 1-step loss reduction sanity
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=1e-2)
        train_input = torch.randint(0, 256, (1, 16))
        train_target = torch.randint(0, 256, (1, 16))

        logits_before, _ = model(train_input)
        loss_before = criterion(logits_before.view(-1, 256), train_target.view(-1))

        optimizer.zero_grad()
        loss_before.backward()
        optimizer.step()

        with torch.no_grad():
            logits_after, _ = model(train_input)
            loss_after = criterion(logits_after.view(-1, 256), train_target.view(-1))

        loss_decreased = bool(loss_after.item() < loss_before.item())

        return {
            "forward_latency_ms": forward_latency_ms,
            "tokens_per_sec": tokens_per_sec,
            "loss_before": loss_before.item(),
            "loss_after": loss_after.item(),
            "loss_decreased": loss_decreased,
        }

    def check_disk_footprint(self) -> float:
        """Measures tracked directories against strict 15 GB quota."""
        tracked = ["models", "data", "checkpoints", ".venv"]
        total_bytes = 0
        for d in tracked:
            dp = os.path.join(self.repo_root, d)
            if os.path.exists(dp):
                for root, _, files in os.walk(dp):
                    for f in files:
                        fp = os.path.join(root, f)
                        try:
                            total_bytes += os.path.getsize(fp)
                        except OSError:
                            pass
        return total_bytes / (1024 * 1024)

    def run_gate(self) -> RegressionReport:
        """Executes all checks and reports pass/fail."""
        bench = self.benchmark_model()
        disk_used_mb = self.check_disk_footprint()

        violations = []
        if bench["tokens_per_sec"] < self.min_tokens_per_sec:
            violations.append(
                f"Throughput regression: {bench['tokens_per_sec']:.2f} t/s < {self.min_tokens_per_sec:.2f} t/s threshold"
            )

        if bench["forward_latency_ms"] > self.max_forward_latency_ms:
            violations.append(
                f"Latency regression: {bench['forward_latency_ms']:.2f} ms > {self.max_forward_latency_ms:.2f} ms threshold"
            )

        if not bench["loss_decreased"]:
            violations.append(
                f"Optimization failure: Loss failed to decrease after 1 gradient step ({bench['loss_before']:.4f} -> {bench['loss_after']:.4f})"
            )

        if disk_used_mb > self.max_disk_mb:
            violations.append(
                f"Storage quota violation: {disk_used_mb:.2f} MB exceeds 15 GB quota ({self.max_disk_mb:.2f} MB)"
            )

        metrics = {
            **bench,
            "disk_used_mb": disk_used_mb,
            "min_tokens_per_sec": self.min_tokens_per_sec,
            "max_forward_latency_ms": self.max_forward_latency_ms,
            "max_disk_mb": self.max_disk_mb,
        }

        return RegressionReport(
            passed=len(violations) == 0,
            metrics=metrics,
            violations=violations,
        )
