"""
Libra API v1 - Attention and KV Cache Endpoints
Phase 23: Key-Value (KV) Cache Optimization and Grouped-Query Attention (MQA/GQA)
"""

import time

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from packages.models.generation import generate, generate_with_cache
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM

router = APIRouter(prefix="/attention", tags=["Attention and KV Cache"])


class AttentionBenchmarkRequest(BaseModel):
    vocab_size: int = Field(default=100, ge=10, le=50000)
    d_model: int = Field(default=64, ge=16, le=1024)
    n_heads: int = Field(default=4, ge=1, le=32)
    n_layers: int = Field(default=2, ge=1, le=16)
    max_seq_len: int = Field(default=128, ge=32, le=2048)
    prompt_len: int = Field(default=10, ge=1, le=64)
    max_new_tokens: int = Field(default=15, ge=1, le=100)


class AttentionGenerateRequest(BaseModel):
    vocab_size: int = Field(default=100, ge=10, le=50000)
    d_model: int = Field(default=64, ge=16, le=1024)
    n_heads: int = Field(default=4, ge=1, le=32)
    n_kv_heads: int = Field(default=2, ge=1, le=32)
    n_layers: int = Field(default=2, ge=1, le=16)
    prompt_tokens: list[int] = Field(default=[1, 5, 12, 8])
    max_new_tokens: int = Field(default=10, ge=1, le=50)
    use_cache: bool = Field(default=True)


@router.post("/benchmark")
def benchmark_attention_mechanisms(request: AttentionBenchmarkRequest):
    """
    Benchmark Multi-Head Attention (MHA), Grouped-Query Attention (GQA),
    and Multi-Query Attention (MQA) with and without KV cache.
    """
    results = {}
    prompt = torch.randint(0, request.vocab_size, (1, request.prompt_len))

    configs = [
        ("MHA", request.n_heads),
        ("GQA", max(1, request.n_heads // 2)),
        ("MQA", 1),
    ]

    for name, n_kv in configs:
        if request.n_heads % n_kv != 0:
            continue

        cfg = ModernTransformerConfig(
            vocab_size=request.vocab_size,
            d_model=request.d_model,
            n_heads=request.n_heads,
            n_kv_heads=n_kv,
            n_layers=request.n_layers,
            max_context_length=request.max_seq_len,
        )
        model = ModernTransformerLM(cfg)
        model.eval()

        # 1. Without cache
        t0 = time.perf_counter()
        with torch.no_grad():
            out_nocache = generate(
                model, prompt.clone(), max_new_tokens=request.max_new_tokens, temperature=0.0
            )
        t_nocache_ms = (time.perf_counter() - t0) * 1000

        # 2. With cache
        t0 = time.perf_counter()
        with torch.no_grad():
            out_cache, _ = generate_with_cache(
                model, prompt.clone(), max_new_tokens=request.max_new_tokens, temperature=0.0
            )
        t_cache_ms = (time.perf_counter() - t0) * 1000

        tokens_match = torch.equal(out_nocache, out_cache)

        head_dim = cfg.d_model // cfg.n_heads
        bytes_per_token = 2 * cfg.n_layers * n_kv * head_dim * 4
        cache_memory_prompt_plus_gen = bytes_per_token * (
            request.prompt_len + request.max_new_tokens
        )

        results[name] = {
            "n_heads": cfg.n_heads,
            "n_kv_heads": cfg.n_kv_heads,
            "num_queries_per_kv": cfg.num_queries_per_kv,
            "latency_uncached_ms": round(t_nocache_ms, 2),
            "latency_cached_ms": round(t_cache_ms, 2),
            "speedup_factor": round(t_nocache_ms / max(t_cache_ms, 1e-6), 2),
            "tokens_match": tokens_match,
            "kv_cache_bytes_per_token": bytes_per_token,
            "total_kv_memory_bytes": cache_memory_prompt_plus_gen,
            "mha_memory_reduction_ratio": round(request.n_heads / n_kv, 2),
        }

    return {
        "status": "success",
        "benchmark_config": request.model_dump(),
        "architectures": results,
    }


@router.post("/generate")
def generate_tokens(request: AttentionGenerateRequest):
    """
    Generate tokens using ModernTransformerLM with optional KV cache acceleration.
    """
    if request.n_heads % request.n_kv_heads != 0:
        raise HTTPException(
            status_code=400,
            detail=f"n_heads ({request.n_heads}) must be divisible by n_kv_heads ({request.n_kv_heads})",
        )

    cfg = ModernTransformerConfig(
        vocab_size=request.vocab_size,
        d_model=request.d_model,
        n_heads=request.n_heads,
        n_kv_heads=request.n_kv_heads,
        n_layers=request.n_layers,
    )
    model = ModernTransformerLM(cfg)
    model.eval()

    prompt_tensor = torch.tensor([request.prompt_tokens], dtype=torch.long)
    t0 = time.perf_counter()

    with torch.no_grad():
        if request.use_cache:
            out, _ = generate_with_cache(
                model, prompt_tensor, max_new_tokens=request.max_new_tokens, temperature=0.0
            )
        else:
            out = generate(
                model, prompt_tensor, max_new_tokens=request.max_new_tokens, temperature=0.0
            )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    generated_tokens = out[0].tolist()

    return {
        "status": "success",
        "use_cache": request.use_cache,
        "n_heads": request.n_heads,
        "n_kv_heads": request.n_kv_heads,
        "prompt_tokens": request.prompt_tokens,
        "generated_tokens": generated_tokens,
        "new_tokens": generated_tokens[len(request.prompt_tokens) :],
        "latency_ms": round(elapsed_ms, 2),
        "tokens_per_second": round(request.max_new_tokens / max(elapsed_ms / 1000, 1e-6), 2),
    }
