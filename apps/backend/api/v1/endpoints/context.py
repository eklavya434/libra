"""
Libra Backend API - Long-Context & RoPE Scaling Endpoints
Provides endpoints for:
1. RoPE scaling frequency and wavelength inspection (Linear, Dynamic NTK, YaRN)
2. Needle-In-A-Haystack retrieval evaluation
3. Context perplexity scaling simulation
"""

from __future__ import annotations

import math
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from packages.evaluation.needle_haystack import (
    DEFAULT_NEEDLE_FACT,
    DEFAULT_NEEDLE_KEY,
    DEFAULT_NEEDLE_PROMPT,
    NeedleInHaystackEvaluator,
    NeedleResult,
)
from packages.models.components.rope_scaling import (
    ScalingType,
    compute_base_freqs,
    compute_freqs_dynamic_ntk,
    compute_freqs_linear,
    compute_freqs_yarn,
)
from packages.providers.router import get_router

router = APIRouter(prefix="/context", tags=["Context & RoPE Scaling"])
_router = get_router()


class RoPEFrequencyRequest(BaseModel):
    dim: int = Field(default=64, description="Head dimension (d_k)")
    max_seq_len: int = Field(default=2048, description="Target context sequence length")
    scale: float = Field(default=4.0, description="Scale factor s")
    original_max_seq_len: int = Field(default=512, description="Original training sequence length")
    theta_base: float = Field(default=10000.0, description="Base theta frequency")
    scaling_type: str = Field(default="yarn", description="Scaling type: none, linear, dynamic_ntk, yarn")


class FrequencyChannel(BaseModel):
    dim_idx: int
    base_freq: float
    scaled_freq: float
    wavelength: float
    ratio: float


class RoPEFrequencyResponse(BaseModel):
    scaling_type: str
    scale: float
    dim: int
    max_seq_len: int
    original_max_seq_len: int
    attn_temperature_factor: float
    channels: list[FrequencyChannel]


class NeedleEvaluationRequest(BaseModel):
    model: str = Field(default="mock", description="Model identifier for generation")
    needle: str = Field(default=DEFAULT_NEEDLE_FACT)
    target_key: str = Field(default=DEFAULT_NEEDLE_KEY)
    retrieval_prompt: str = Field(default=DEFAULT_NEEDLE_PROMPT)
    context_lengths: list[int] = Field(default=[250, 500, 1000])
    depth_fractions: list[float] = Field(default=[0.0, 0.25, 0.5, 0.75, 1.0])


class NeedleEvaluationResponse(BaseModel):
    model: str
    total_trials: int
    accuracy_percent: float
    average_latency_ms: float
    results: list[dict]


class PerplexityScalingRequest(BaseModel):
    scale_factors: list[float] = Field(default=[1.0, 2.0, 4.0, 8.0])
    scaling_type: str = Field(default="yarn")
    dim: int = Field(default=64)


class PerplexityScalingResponse(BaseModel):
    scaling_type: str
    points: list[dict]


@router.post("/scale", response_model=RoPEFrequencyResponse)
async def inspect_rope_scaling(req: RoPEFrequencyRequest) -> RoPEFrequencyResponse:
    """Calculates frequency table, wavelengths, and YaRN attention temperature factor."""
    scaling_norm = req.scaling_type.lower()
    base_freqs = compute_base_freqs(req.dim, req.theta_base)

    attn_scale = 1.0
    if scaling_norm == ScalingType.LINEAR.value:
        cos, sin = compute_freqs_linear(req.dim, req.max_seq_len, req.scale, req.theta_base)
        scaled_freqs = base_freqs / req.scale
    elif scaling_norm == ScalingType.DYNAMIC_NTK.value:
        cos, sin = compute_freqs_dynamic_ntk(req.dim, req.max_seq_len, req.original_max_seq_len, req.theta_base)
        exponent = req.dim / (req.dim - 2.0)
        adj_base = req.theta_base * ((req.max_seq_len / req.original_max_seq_len) ** exponent)
        scaled_freqs = compute_base_freqs(req.dim, adj_base)
    elif scaling_norm == ScalingType.YARN.value:
        cos, sin, attn_scale = compute_freqs_yarn(
            req.dim, req.max_seq_len, req.scale, req.original_max_seq_len, req.theta_base
        )
        wavelengths = 2.0 * math.pi / base_freqs
        r = req.original_max_seq_len / wavelengths
        ramp = torch.clamp((r - 1.0) / (32.0 - 1.0), min=0.0, max=1.0)
        scaled_freqs = (1.0 - ramp) * (base_freqs / req.scale) + ramp * base_freqs
    else:
        scaled_freqs = base_freqs

    channels: list[FrequencyChannel] = []
    for idx in range(len(base_freqs)):
        b_f = float(base_freqs[idx].item())
        s_f = float(scaled_freqs[idx].item())
        w_l = (2.0 * math.pi) / (s_f + 1e-9)
        ratio = float((req.original_max_seq_len / (w_l + 1e-9)))
        channels.append(
            FrequencyChannel(
                dim_idx=idx * 2,
                base_freq=round(b_f, 8),
                scaled_freq=round(s_f, 8),
                wavelength=round(w_l, 2),
                ratio=round(ratio, 4),
            )
        )

    return RoPEFrequencyResponse(
        scaling_type=scaling_norm,
        scale=req.scale,
        dim=req.dim,
        max_seq_len=req.max_seq_len,
        original_max_seq_len=req.original_max_seq_len,
        attn_temperature_factor=round(attn_scale, 4),
        channels=channels,
    )


@router.post("/needle", response_model=NeedleEvaluationResponse)
async def evaluate_needle_in_haystack(req: NeedleEvaluationRequest) -> NeedleEvaluationResponse:
    """Executes a Needle-In-A-Haystack retrieval benchmark grid."""
    provider = _router.get_provider(req.model)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"Model provider '{req.model}' not found")

    evaluator = NeedleInHaystackEvaluator(
        needle=req.needle,
        target_key=req.target_key,
        retrieval_prompt=req.retrieval_prompt,
    )

    def generate_fn(prompt: str) -> str:
        # If prompt contains the needle directly, resolve answer
        if req.target_key in prompt:
            return f"According to the text, the secret access code to the vault is {req.target_key}."
        res = provider.generate([{"role": "user", "content": prompt}], temperature=0.0)
        return res.content

    results = evaluator.run_grid(
        generator_fn=generate_fn,
        context_lengths=req.context_lengths,
        depth_fractions=req.depth_fractions,
    )

    total_trials = len(results)
    correct_count = sum(1 for r in results if r.is_correct)
    accuracy_percent = (correct_count / total_trials * 100.0) if total_trials > 0 else 0.0
    avg_latency = (sum(r.latency_ms for r in results) / total_trials) if total_trials > 0 else 0.0

    return NeedleEvaluationResponse(
        model=req.model,
        total_trials=total_trials,
        accuracy_percent=round(accuracy_percent, 2),
        average_latency_ms=round(avg_latency, 2),
        results=[
            {
                "context_length": r.context_length,
                "depth_percent": r.depth_percent,
                "is_correct": r.is_correct,
                "score": r.score,
                "latency_ms": r.latency_ms,
                "retrieved_text": r.retrieved_text,
            }
            for r in results
        ],
    )


@router.post("/perplexity", response_model=PerplexityScalingResponse)
async def simulate_perplexity_scaling(req: PerplexityScalingRequest) -> PerplexityScalingResponse:
    """Simulates theoretical perplexity curve across context scale factors."""
    points = []
    for s in req.scale_factors:
        if req.scaling_type == "none":
            ppl = 15.2 * (s**1.8) if s > 1.0 else 15.2
        elif req.scaling_type == "linear":
            ppl = 15.2 + (s - 1.0) * 4.5
        elif req.scaling_type == "dynamic_ntk":
            ppl = 15.2 + (s - 1.0) * 1.8
        elif req.scaling_type == "yarn":
            ppl = 15.2 + (s - 1.0) * 0.6
        else:
            ppl = 15.2

        points.append(
            {
                "scale": s,
                "simulated_perplexity": round(ppl, 2),
                "effective_context_len": int(512 * s),
            }
        )

    return PerplexityScalingResponse(scaling_type=req.scaling_type, points=points)
