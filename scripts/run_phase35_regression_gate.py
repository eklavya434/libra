"""
Phase 35 CLI Demonstration:
Automated CPU Performance Regression Gate & Pre-Commit Verification
Demonstrates:
1. Micro-benchmarking CPU token generation throughput with KV cache (tokens/sec)
2. Single-step optimization loss descent verification
3. Forward pass latency profiling
4. Storage quota tracking (< 15 GB quota)
5. CI/CD matrix integration
"""

import os
import sys

# Ensure project root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from packages.evaluation.regression_gate import PerformanceRegressionGate


def main():
    print("=" * 80)
    print("LIBRA PHASE 35: CPU PERFORMANCE & REGRESSION GATE BENCHMARK")
    print("=" * 80)

    gate = PerformanceRegressionGate(
        min_tokens_per_sec=10.0,
        max_forward_latency_ms=100.0,
        max_disk_mb=15360.0,
        repo_root=REPO_ROOT,
    )

    print("\n[1] Running Micro-Benchmarks on ModernTransformerLM (CPU)...")
    report = gate.run_gate()

    print("\n[2] Benchmark Metrics:")
    print("-" * 50)
    for k, v in report.metrics.items():
        if isinstance(v, float):
            print(f"  {k:26s}: {v:10.2f}")
        else:
            print(f"  {k:26s}: {str(v):>10s}")
    print("-" * 50)

    print("\n[3] Gate Evaluation:")
    if report.passed:
        print("  [+] Throughput Check: PASSED (No CPU decoding degradation)")
        print("  [+] Latency Check:    PASSED (Forward pass within budget)")
        print("  [+] Optimization:    PASSED (Loss decreases monotonically on step)")
        print("  [+] Storage Quota:   PASSED (Footprint strictly under 15 GB)")
        print("\n" + "=" * 80)
        print("STATUS: PERFORMANCE REGRESSION GATE PASSED (READY FOR MERGE)")
        print("=" * 80)
        return 0
    else:
        print("  [-] REGRESSION DETECTED:")
        for viol in report.violations:
            print(f"    - {viol}")
        print("\n" + "=" * 80)
        print("STATUS: PERFORMANCE REGRESSION GATE FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
