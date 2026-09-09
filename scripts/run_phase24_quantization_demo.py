"""
Project Libra - Phase 24 Demonstration
Quantization (INT8 / INT4 & Post-Training Quantization from First Principles)

Demonstrates:
1. Symmetric vs Asymmetric affine quantization error profiles
2. INT4 nibble packing (two 4-bit weights packed into 1 uint8 byte)
3. Model memory reduction (FP32 -> INT8 -> INT4)
4. Logits fidelity metrics (MSE, SQNR dB, Cosine Similarity, Top-1 Agreement)
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

from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.models.quantization import (
    audit_quantization_fidelity,
    compute_model_memory,
    compute_quantization_metrics,
    dequantize_asymmetric,
    dequantize_symmetric,
    pack_int4,
    quantize_asymmetric,
    quantize_model,
    quantize_symmetric,
    unpack_int4,
)


def print_banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def demo_tensor_quantization() -> None:
    print_banner("1. Tensor Quantization Arithmetic & Bit Packing")

    torch.manual_seed(42)
    x = torch.randn(4, 16, dtype=torch.float32)

    # 1. Symmetric INT8
    x_q8, scale8 = quantize_symmetric(x, bits=8)
    x_rec8 = dequantize_symmetric(x_q8, scale8)
    m8 = compute_quantization_metrics(x, x_rec8)

    # 2. Symmetric INT4 with Packing
    x_q4, scale4 = quantize_symmetric(x, bits=4)
    packed = pack_int4(x_q4)
    unpacked = unpack_int4(packed)
    x_rec4 = dequantize_symmetric(unpacked, scale4)
    m4 = compute_quantization_metrics(x, x_rec4)

    # 3. Asymmetric INT8
    x_asym = torch.rand(4, 16) * 5.0 + 2.0
    x_q_asym, scale_asym, zp = quantize_asymmetric(x_asym, bits=8)
    x_rec_asym = dequantize_asymmetric(x_q_asym, scale_asym, zp)
    m_asym = compute_quantization_metrics(x_asym, x_rec_asym)

    print(
        f"{'Quantization Scheme':<28} | {'Bits':<5} | {'Packed Shape':<14} | {'SQNR (dB)':<10} | {'Cosine Sim':<10}"
    )
    print("-" * 75)
    print(
        f"{'Symmetric INT8':<28} | 8-bit | {str(tuple(x_q8.shape)):<14} | {m8['sqnr_db']:<10.2f} | {m8['cosine_similarity']:<10.6f}"
    )
    print(
        f"{'Symmetric INT4 (Packed)':<28} | 4-bit | {str(tuple(packed.shape)):<14} | {m4['sqnr_db']:<10.2f} | {m4['cosine_similarity']:<10.6f}"
    )
    print(
        f"{'Asymmetric INT8':<28} | 8-bit | {str(tuple(x_q_asym.shape)):<14} | {m_asym['sqnr_db']:<10.2f} | {m_asym['cosine_similarity']:<10.6f}"
    )


def demo_model_quantization() -> None:
    print_banner("2. Post-Training Quantization on ModernTransformerLM")

    torch.manual_seed(42)
    cfg = ModernTransformerConfig(
        vocab_size=500,
        d_model=128,
        n_heads=4,
        n_layers=4,
        max_context_length=256,
    )
    fp32_model = ModernTransformerLM(cfg)
    fp32_model.eval()

    sample_input = torch.randint(0, cfg.vocab_size, (1, 16))

    # Benchmark FP32
    fp32_mem = compute_model_memory(fp32_model)
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = fp32_model(sample_input)
    fp32_lat = (time.perf_counter() - t0) * 1000

    # INT8 Quantization
    q8_model = quantize_model(fp32_model, mode="int8", per_channel=True)
    audit8 = audit_quantization_fidelity(fp32_model, q8_model, sample_input)
    q8_mem = compute_model_memory(q8_model)

    t0 = time.perf_counter()
    with torch.no_grad():
        _ = q8_model(sample_input)
    q8_lat = (time.perf_counter() - t0) * 1000

    # INT4 Quantization
    q4_model = quantize_model(fp32_model, mode="int4", per_channel=True)
    audit4 = audit_quantization_fidelity(fp32_model, q4_model, sample_input)
    q4_mem = compute_model_memory(q4_model)

    t0 = time.perf_counter()
    with torch.no_grad():
        _ = q4_model(sample_input)
    q4_lat = (time.perf_counter() - t0) * 1000

    print(
        f"{'Precision Tier':<16} | {'Memory (MB)':<12} | {'Savings':<10} | {'Cosine Sim':<11} | {'Top-1 Match':<11} | {'Latency':<9}"
    )
    print("-" * 80)
    print(
        f"{'FP32 Baseline':<16} | {fp32_mem['total_mb']:<12.3f} | {'Baseline':<10} | {'1.000000':<11} | {'100.0%':<11} | {fp32_lat:<6.2f} ms"
    )
    print(
        f"{'INT8 Quantized':<16} | {q8_mem['total_mb']:<12.3f} | {audit8['memory']['savings_pct']:<10} | {audit8['fidelity']['cosine_similarity']:<11.6f} | {audit8['fidelity']['top1_agreement_pct']:<10.1f}% | {q8_lat:<6.2f} ms"
    )
    print(
        f"{'INT4 Quantized':<16} | {q4_mem['total_mb']:<12.3f} | {audit4['memory']['savings_pct']:<10} | {audit4['fidelity']['cosine_similarity']:<11.6f} | {audit4['fidelity']['top1_agreement_pct']:<10.1f}% | {q4_lat:<6.2f} ms"
    )


if __name__ == "__main__":
    demo_tensor_quantization()
    demo_model_quantization()
    print("\nPhase 24 quantization verification demo completed successfully.\n")
