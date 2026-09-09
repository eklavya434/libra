"""
Libra Phase 10 - Model Comparison Arena Demonstration
Benchmarks side-by-side model completions, measuring TTFT, latency, throughput, and cost.
"""

import asyncio
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.evaluation.arena import ModelComparisonArena


async def main() -> None:
    print("=" * 80)
    print("  PROJECT LIBRA - PHASE 10: MODEL COMPARISON ARENA")
    print("  Side-by-Side Benchmarking: TTFT, Throughput (tok/s), Latency and Cost")
    print("=" * 80)

    arena = ModelComparisonArena()

    test_prompt = "Explain why transformers use multi-head attention instead of a single head."
    models = [
        {"model": "libra-educational-tiny", "provider": "libra_lab"},
        {"model": "libra-mock-v1", "provider": "mock-provider"},
        {"model": "gpt-4o-mini", "provider": "mock-provider"},
        {"model": "claude-3-5-haiku-20241022", "provider": "mock-provider"},
    ]

    print(f'\n[?] Benchmark Prompt: "{test_prompt}"')
    print(f"[*] Contending Models: {len(models)}")
    for m in models:
        print(f"    - {m['model']} (via {m['provider']})")

    print("\n>>> Dispatching concurrent inference requests to all models...")
    result = await arena.compare(
        prompt=test_prompt,
        models=models,
        max_tokens=64,
        temperature=0.7,
    )

    print("\n" + "-" * 80)
    print(
        f"{'MODEL':<28} | {'PROVIDER':<14} | {'TTFT':<10} | {'LATENCY':<10} | {'TOK/S':<8} | {'COST'}"
    )
    print("-" * 80)

    for r in result["results"]:
        model_name = r["model"]
        prov_name = r["provider"]
        ttft = f"{r['ttft_ms']:.1f} ms" if r["ttft_ms"] is not None else "N/A"
        latency = f"{r['total_latency_ms']:.1f} ms"
        tps = f"{r['tokens_per_second']:.1f}"
        cost = "$0.000000" if r["is_free"] else f"${r['cost_usd']:.6f}"
        print(
            f"{model_name:<28} | {prov_name:<14} | {ttft:<10} | {latency:<10} | {tps:<8} | {cost}"
        )

    print("-" * 80)
    print("\n[+] Arena Winners and Leaderboard:")
    print(f"  * Fastest Time to First Token (TTFT): {result['rankings']['fastest_ttft']}")
    print(f"  * Highest Token Throughput (tok/s):   {result['rankings']['highest_throughput']}")
    print(f"  * Most Economical (Lowest Cost):       {result['rankings']['lowest_cost']}")

    print("\n[+] Sample Output Inspection:")
    for r in result["results"][:2]:
        print(f"\n[{r['model']}]:")
        print(f'  "{r["output_text"].strip()}"')

    print("\n" + "=" * 80)
    print("  PHASE 10 ARENA BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
