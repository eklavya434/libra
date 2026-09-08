"""
Libra API v1 - Reasoning Engine & Test-Time Compute Endpoints (Phase 29)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from packages.models.reasoning.search_verifier import BestOfNVerifier
from packages.models.reasoning.self_consistency import SelfConsistencyEngine
from packages.models.reasoning.trace_parser import parse_reasoning_trace
from packages.providers.router import get_router

router = APIRouter()
_router = get_router()
_sc_engine = SelfConsistencyEngine(router=_router)
_bon_verifier = BestOfNVerifier(router=_router)


class ReasoningGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Question or problem requiring reasoning")
    model: str = Field("qwen3:4b", description="Model ID to invoke")
    provider: str | None = Field(
        None, description="Explicit provider name (e.g. ollama, mock-provider)"
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=32, le=4096)
    system_prompt: str | None = Field(None)


class SelfConsistencyRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str = Field("qwen3:4b")
    provider: str | None = Field(None)
    num_paths: int = Field(3, ge=1, le=7, description="Number of parallel reasoning rollouts")
    temperature: float = Field(0.7, ge=0.1, le=1.5)
    max_tokens: int = Field(256, ge=32, le=2048)
    system_prompt: str | None = Field(None)


class BestOfNRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str = Field("qwen3:4b")
    provider: str | None = Field(None)
    n_candidates: int = Field(3, ge=1, le=7, description="Number of candidate reasoning rollouts")
    temperature: float = Field(0.8, ge=0.1, le=1.5)
    max_tokens: int = Field(256, ge=32, le=2048)


class ParseTraceRequest(BaseModel):
    text: str = Field(..., description="Raw output text containing possible <think> tags")
    duration_ms: float = Field(0.0)


@router.post(
    "/reasoning/generate", summary="Execute System 2 deliberative reasoning with thought trace"
)
async def generate_reasoning(request: ReasoningGenerateRequest) -> Any:
    """Generates a step-by-step reasoning response, cleanly parsing internal thoughts from the final answer."""
    try:
        sys_prompt = request.system_prompt or (
            "You are an expert reasoning assistant. Think step by step before answering. "
            "Enclose your complete thinking process inside <think>...</think> tags, followed by your final answer."
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": request.prompt},
        ]

        resp = await _router.chat(
            messages=messages,
            model_id=request.model,
            provider_name=request.provider,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        choices = resp.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        trace = parse_reasoning_trace(content)

        return {
            "model": request.model,
            "provider": resp.get("provider", "unknown"),
            "trace": trace.model_dump(),
            "raw_response": content,
        }
    except (ValueError, TypeError, RuntimeError, OSError, KeyError, IndexError) as e:
        raise HTTPException(status_code=500, detail=f"Reasoning generation failed: {e!s}")


@router.post(
    "/reasoning/self-consistency", summary="Execute self-consistency multi-path majority voting"
)
async def evaluate_self_consistency(request: SelfConsistencyRequest) -> Any:
    """Samples multiple stochastic reasoning trajectories and identifies the modal consensus answer."""
    try:
        result = await _sc_engine.evaluate(
            prompt=request.prompt,
            model=request.model,
            num_paths=request.num_paths,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_prompt=request.system_prompt,
            provider_name=request.provider,
        )
        return result.model_dump()
    except (ValueError, TypeError, RuntimeError, OSError, KeyError, IndexError) as e:
        raise HTTPException(status_code=500, detail=f"Self-consistency failed: {e!s}")


@router.post("/reasoning/best-of-n", summary="Execute Best-of-N test-time compute search")
async def evaluate_best_of_n(request: BestOfNRequest) -> Any:
    """Generates N reasoning candidates, scores them using multi-criteria reward evaluation, and returns top pick."""
    try:
        result = await _bon_verifier.search(
            prompt=request.prompt,
            model=request.model,
            n_candidates=request.n_candidates,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            provider_name=request.provider,
        )
        return result.model_dump()
    except (ValueError, TypeError, RuntimeError, OSError, KeyError, IndexError) as e:
        raise HTTPException(status_code=500, detail=f"Best-of-N search failed: {e!s}")


@router.post(
    "/reasoning/parse", summary="Parse a raw text string into structured thought trace and answer"
)
async def parse_trace(request: ParseTraceRequest) -> Any:
    """Extracts <think> tags, numbered steps, and final answer from arbitrary text."""
    trace = parse_reasoning_trace(request.text, duration_ms=request.duration_ms)
    return trace.model_dump()
