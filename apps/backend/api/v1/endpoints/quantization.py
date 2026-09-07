"""
Libra API v1 - Quantization Endpoints
Phase 24: Quantization (INT8 / INT4 & Post-Training Quantization)
"""

import time

import torch
from fastapi import APIRouter
from pydantic import BaseModel, Field

from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.models.quantization import (
    audit_quantization_fidelity,
    compute_model_memory,
    quantize_model,
)

router = APIRouter(prefix="/quantization", tags=["Quantization & Compression"])


class QuantizationBenchmarkRequest(BaseModel):
    vocab_size: int = Field(default=100, ge=10, le=50000)
    d_model: int = Field(default=64, ge=16, le=1024)
    n_heads: int = Field(default=4, ge=1, le=32)
    n_layers: int = Field(default=2, ge=1, le=16)
    seq_len: int = Field(default=16, ge=4, le=128)
    per_channel: bool = Field(default=True)


class QuantizationConvertRequest(BaseModel):
    vocab_size: int = Field(default=100, ge=10, le=50000)
    d_model: int = Field(default=64, ge=16, le=1024)
    n_heads: int = Field(default=4, ge=1, le=32)
    n_layers: int = Field(default=2, ge=1, le=16)
    mode: str = Field(default="int8", pattern="^(int8|int4)$")
    per_channel: bool = Field(default=True)


@router.post("/benchmark")
def benchmark_quantization(request: QuantizationBenchmarkRequest):
    """
    Benchmark FP32 baseline vs INT8 vs INT4 post-training quantization.
    Reports memory reduction, SQNR, MSE, top-1 token agreement, and forward latency.
    """
    cfg = ModernTransformerConfig(
        vocab_size=request.vocab_size,
        d_model=request.d_model,
        n_heads=request.n_heads,
        n_layers=request.n_layers,
    )
    fp32_model = ModernTransformerLM(cfg)
    fp32_model.eval()

    sample_input = torch.randint(0, request.vocab_size, (1, request.seq_len))

    # 1. Baseline FP32 Latency
    t0 = time.perf_counter()
    with torch.no_grad():
        _fp32_out, _ = fp32_model(sample_input)
    fp32_latency_ms = (time.perf_counter() - t0) * 1000

    fp32_mem = compute_model_memory(fp32_model)

    modes = ["int8", "int4"]
    results = {
        "fp32_baseline": {
            "total_bytes": fp32_mem["total_bytes"],
            "total_mb": fp32_mem["total_mb"],
            "latency_ms": round(fp32_latency_ms, 2),
        },
        "quantized_tiers": {},
    }

    for mode in modes:
        q_model = quantize_model(fp32_model, mode=mode, per_channel=request.per_channel)
        q_model.eval()

        # Audit fidelity
        audit = audit_quantization_fidelity(fp32_model, q_model, sample_input)

        # Measure forward latency
        t0 = time.perf_counter()
        with torch.no_grad():
            _q_out, _ = q_model(sample_input)
        q_latency_ms = (time.perf_counter() - t0) * 1000

        q_mem = compute_model_memory(q_model)

        results["quantized_tiers"][mode.upper()] = {
            "mode": mode,
            "per_channel": request.per_channel,
            "total_bytes": q_mem["total_bytes"],
            "total_mb": q_mem["total_mb"],
            "compression_ratio": audit["memory"]["compression_ratio"],
            "memory_savings": audit["memory"]["savings_pct"],
            "latency_ms": round(q_latency_ms, 2),
            "mse": audit["fidelity"]["mse"],
            "sqnr_db": audit["fidelity"]["sqnr_db"],
            "cosine_similarity": audit["fidelity"]["cosine_similarity"],
            "top1_agreement_pct": audit["fidelity"]["top1_agreement_pct"],
        }

    return {
        "status": "success",
        "model_config": request.model_dump(),
        "benchmark": results,
    }


@router.post("/convert")
def convert_model_endpoint(request: QuantizationConvertRequest):
    """
    Quantizes a model configuration to INT8 or INT4 and returns parameter and memory summary.
    """
    cfg = ModernTransformerConfig(
        vocab_size=request.vocab_size,
        d_model=request.d_model,
        n_heads=request.n_heads,
        n_layers=request.n_layers,
    )
    fp32_model = ModernTransformerLM(cfg)
    fp32_model.eval()

    sample_input = torch.randint(0, request.vocab_size, (1, 8))

    q_model = quantize_model(fp32_model, mode=request.mode, per_channel=request.per_channel)
    q_model.eval()

    audit = audit_quantization_fidelity(fp32_model, q_model, sample_input)

    return {
        "status": "success",
        "mode": request.mode,
        "per_channel": request.per_channel,
        "memory_audit": audit["memory"],
        "fidelity_audit": audit["fidelity"],
    }
