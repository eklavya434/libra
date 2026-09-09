"""
Project Libra - Phase 23 Demonstration
Key-Value (KV) Cache Optimization & Grouped-Query Attention (MQA / GQA)

Demonstrates:
1. Multi-Head Attention (MHA) vs Grouped-Query Attention (GQA) vs Multi-Query Attention (MQA)
2. KV Cache memory footprint reduction across architectures
3. Autoregressive inference latency comparison: O(T^2) cacheless vs O(T) cached
4. Exact mathematical token-for-token equivalence under greedy decoding
"""

import sys
import time
from pathlib import Path

import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from packages.models.generation import generate, generate_with_cache
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM


def print_banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def demo_gqa_architectures() -> None:
    print_banner("1. Attention Architectures: MHA vs GQA vs MQA")

    d_model = 128
    n_heads = 8
    vocab_size = 500
    n_layers = 4
    max_context_length = 512

    configs = [
        ("Multi-Head Attention (MHA)", n_heads),
        ("Grouped-Query Attention (GQA-4)", 4),
        ("Grouped-Query Attention (GQA-2)", 2),
        ("Multi-Query Attention (MQA)", 1),
    ]

    print(
        f"{'Architecture':<35} | {'Q Heads':<8} | {'KV Heads':<8} | {'Ratio':<6} | {'Params':<10}"
    )
    print("-" * 75)

    for name, n_kv in configs:
        cfg = ModernTransformerConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv,
            n_layers=n_layers,
            max_context_length=max_context_length,
        )
        model = ModernTransformerLM(cfg)
        params = model.count_parameters()
        ratio = f"{cfg.num_queries_per_kv}:1"
        print(f"{name:<35} | {cfg.n_heads:<8} | {cfg.n_kv_heads:<8} | {ratio:<6} | {params:<10,}")


def demo_kv_cache_memory() -> None:
    print_banner(
        "2. Theoretical KV Cache Memory Scaling (Context: 2048 tokens, 32 layers, d_model=4096)"
    )

    n_layers = 32
    d_model = 4096
    n_heads = 32
    head_dim = d_model // n_heads  # 128
    seq_len = 2048
    bytes_per_elem = 2  # FP16 / BF16

    cases = [
        ("Standard MHA (32 KV heads)", 32),
        ("GQA-8 (8 KV heads - Llama 3 8B)", 8),
        ("GQA-4 (4 KV heads)", 4),
        ("MQA (1 KV head)", 1),
    ]

    print(f"{'Configuration':<35} | {'KV Heads':<8} | {'Memory (MB)':<12} | {'Reduction':<10}")
    print("-" * 75)

    base_bytes = None
    for name, n_kv in cases:
        # Memory per token = 2 (K+V) * n_layers * n_kv * head_dim * bytes_per_elem
        mem_bytes = 2 * n_layers * n_kv * head_dim * bytes_per_elem * seq_len
        mem_mb = mem_bytes / (1024 * 1024)
        if base_bytes is None:
            base_bytes = mem_bytes
            reduction = "1.0x (Baseline)"
        else:
            ratio = base_bytes / mem_bytes
            reduction = f"{ratio:.1f}x reduction"

        print(f"{name:<35} | {n_kv:<8} | {mem_mb:<12.2f} | {reduction:<10}")


def demo_latency_and_equivalence() -> None:
    print_banner("3. Latency & Mathematical Equivalence: Cached vs Cacheless")

    torch.manual_seed(42)
    cfg = ModernTransformerConfig(
        vocab_size=200,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,  # GQA
        n_layers=2,
        max_context_length=128,
    )
    model = ModernTransformerLM(cfg)
    model.eval()

    prompt_len = 10
    gen_len = 25
    prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len))

    print(f"Prompt length: {prompt_len} tokens | Generating: {gen_len} new tokens")

    # 1. Cacheless generation
    t0 = time.perf_counter()
    with torch.no_grad():
        out_nocache = generate(model, prompt.clone(), max_new_tokens=gen_len, temperature=0.0)
    t_nocache = (time.perf_counter() - t0) * 1000

    # 2. KV Cached generation
    t0 = time.perf_counter()
    with torch.no_grad():
        out_cache, telemetry = generate_with_cache(
            model, prompt.clone(), max_new_tokens=gen_len, temperature=0.0
        )
    t_cache = (time.perf_counter() - t0) * 1000

    # Check mathematical equivalence
    exact_match = torch.equal(out_nocache, out_cache)

    print("\n[Generation Results]")
    print(f"Cacheless Latency:  {t_nocache:.2f} ms ({gen_len / (t_nocache / 1000):.1f} tok/s)")
    print(f"Cached Latency:     {t_cache:.2f} ms ({gen_len / (t_cache / 1000):.1f} tok/s)")
    speedup = t_nocache / max(t_cache, 1e-6)
    print(f"Speedup Factor:     {speedup:.2f}x faster with KV Cache")
    print(f"Cache RAM Footprint:{telemetry['cache_memory_bytes']} bytes")
    print(f"Exact Match:        {'SUCCESS (100% Token Equivalence)' if exact_match else 'FAILED'}")
    print(f"\nGenerated Tokens:   {telemetry['generated_tokens']}")


if __name__ == "__main__":
    demo_gqa_architectures()
    demo_kv_cache_memory()
    demo_latency_and_equivalence()
    print("\nPhase 23 verification demo completed successfully.\n")
