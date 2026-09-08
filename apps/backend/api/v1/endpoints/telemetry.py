"""
Libra API v1 - Streaming Token Telemetry Endpoints
Phase 26: Token-level Confidence, Surprisal, Entropy, Top-k Alternatives, and Real-Time SSE Streaming
"""

from __future__ import annotations

import json
import time
from typing import Any

import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.models.telemetry import (
    SequenceTelemetry,
    TokenTelemetry,
    aggregate_sequence_telemetry,
    analyze_sequence_telemetry,
    astream_generate_with_telemetry,
    stream_generate_with_telemetry,
)
from packages.providers.local_transformer import LocalTransformerProvider

router = APIRouter(prefix="/telemetry", tags=["Token Telemetry"])

# Global singleton provider instance for fast reuse
_local_provider: LocalTransformerProvider | None = None


def get_or_create_model_and_tokenizer():
    """Retrieves the local transformer model and tokenizer."""
    global _local_provider
    if _local_provider is None:
        _local_provider = LocalTransformerProvider()
    try:
        model, tokenizer = _local_provider._ensure_loaded()
        return model, tokenizer
    except (RuntimeError, FileNotFoundError, OSError):
        # Fallback to in-memory educational model
        config = ModernTransformerConfig(
            vocab_size=256,
            d_model=64,
            n_heads=4,
            n_layers=2,
            max_context_length=512,
        )
        model = ModernTransformerLM(config)
        model.eval()
        return model, None


class TelemetryGenerateRequest(BaseModel):
    prompt: str = Field(
        default="Artificial intelligence will",
        description="Prompt text conditioning autoregressive generation",
    )
    max_tokens: int = Field(default=16, ge=1, le=256, description="Number of tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling randomness")
    top_k: int | None = Field(default=40, ge=1, description="Top-k sampling truncation")
    candidate_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of alternative candidate tokens to log per step",
    )
    model: str = Field(default="libra-llama-tied", description="Target model ID")


class TelemetryAnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=2,
        description="Input text to analyze under teacher-forcing cross-entropy evaluation",
    )
    candidate_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of candidate alternatives to log per token position",
    )
    model: str = Field(default="libra-llama-tied", description="Target model ID")


@router.post(
    "/generate",
    response_model=SequenceTelemetry,
    summary="Generate text with complete token telemetry",
)
def generate_with_telemetry_endpoint(request: TelemetryGenerateRequest) -> Any:
    """Generates completion text autoregressively, computing exact conditional probabilities,

    Shannon surprisals (-log2 p), next-token distribution entropies, top-k candidate
    alternatives, and sequence perplexity from first principles.
    """
    model, tokenizer = get_or_create_model_and_tokenizer()

    if tokenizer is not None:
        prompt_ids = tokenizer.encode(request.prompt)
    else:
        prompt_ids = list(request.prompt.encode("utf-8"))

    if not prompt_ids:
        prompt_ids = [0]

    device = next(model.parameters()).device
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    start_time = time.perf_counter()
    tokens: list[TokenTelemetry] = []

    generator = stream_generate_with_telemetry(
        model=model,
        idx=idx,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        candidate_top_k=request.candidate_top_k,
        tokenizer=tokenizer,
    )

    tokens = list(generator)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    sequence_telemetry = aggregate_sequence_telemetry(tokens, elapsed_ms)
    return sequence_telemetry


@router.post("/stream", summary="Stream generated tokens with real-time SSE telemetry")
async def stream_telemetry_endpoint(request: TelemetryGenerateRequest):
    """Streams generated tokens via Server-Sent Events (SSE) with per-token telemetry.

    Emits:
    - `event: token` with TokenTelemetry JSON payload for each generated token.
    - `event: done` with full SequenceTelemetry JSON summary at completion.
    - `data: [DONE]` signal for standard SSE consumers.
    """
    model, tokenizer = get_or_create_model_and_tokenizer()

    if tokenizer is not None:
        prompt_ids = tokenizer.encode(request.prompt)
    else:
        prompt_ids = list(request.prompt.encode("utf-8"))

    if not prompt_ids:
        prompt_ids = [0]

    device = next(model.parameters()).device
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    async def event_generator():
        collected_tokens: list[TokenTelemetry] = []
        stream_start_ms = time.perf_counter() * 1000.0

        try:
            async for tok_telem in astream_generate_with_telemetry(
                model=model,
                idx=idx,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                candidate_top_k=request.candidate_top_k,
                tokenizer=tokenizer,
            ):
                collected_tokens.append(tok_telem)
                # Emit token event
                yield f"event: token\ndata: {tok_telem.model_dump_json()}\n\n"

            # Compute and emit completion event
            total_duration_ms = (time.perf_counter() * 1000.0) - stream_start_ms
            summary = aggregate_sequence_telemetry(collected_tokens, total_duration_ms)
            yield f"event: done\ndata: {summary.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        except (RuntimeError, ValueError, OSError) as e:
            err_payload = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {err_payload}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post(
    "/analyze",
    response_model=SequenceTelemetry,
    summary="Evaluate sequence surprisal under teacher forcing",
)
def analyze_sequence_endpoint(request: TelemetryAnalyzeRequest) -> Any:
    """Evaluates an input text string under teacher forcing without sampling.

    Calculates the exact surprisal, entropy, and candidate alternatives for every token
    conditioned on its ground-truth preceding context.
    """
    model, tokenizer = get_or_create_model_and_tokenizer()

    if tokenizer is not None:
        input_ids = tokenizer.encode(request.text)
    else:
        input_ids = list(request.text.encode("utf-8"))

    if len(input_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="Input text must encode to at least 2 tokens for teacher-forcing evaluation.",
        )

    device = next(model.parameters()).device
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    telemetry = analyze_sequence_telemetry(
        model=model,
        input_tokens=input_tensor,
        tokenizer=tokenizer,
        candidate_top_k=request.candidate_top_k,
    )
    return telemetry
