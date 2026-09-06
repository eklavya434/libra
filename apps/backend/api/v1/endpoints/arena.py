"""
Libra API v1 - Model Comparison Arena Endpoint (Phase 10)
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from packages.evaluation.arena import ModelComparisonArena

router = APIRouter()
_arena = ModelComparisonArena()


class ModelTarget(BaseModel):
    model: str = Field(..., description="Model ID to evaluate")
    provider: Optional[str] = Field(None, description="Explicit provider name (e.g. libra_lab, mock-provider, ollama)")


class ArenaCompareRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Benchmark prompt to run across models")
    models: list[ModelTarget] = Field(..., min_length=1, description="List of model targets to compare")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(256, ge=1, le=2048, description="Maximum tokens to generate")
    system_prompt: Optional[str] = Field(None, description="Optional system instruction")


@router.post("/arena/compare", summary="Compare multiple models concurrently")
async def compare_models(request: ArenaCompareRequest) -> Any:
    """Runs comparative benchmark across models measuring TTFT, latency, tok/s, and cost."""
    try:
        models_payload = [{"model": m.model, "provider": m.provider} for m in request.models]
        result = await _arena.compare(
            prompt=request.prompt,
            models=models_payload,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_prompt=request.system_prompt,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Arena execution failed: {str(e)}")

