"""
Libra API v1 - Model Comparison Arena & Automated Elo Leaderboard (Phase 10 & Phase 28)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from packages.evaluation.arena import ModelComparisonArena
from packages.evaluation.automated_judge import AutomatedJudge
from packages.evaluation.elo import get_leaderboard_store
from packages.evaluation.tournament import BenchmarkPrompt, TournamentRunner

router = APIRouter()
_arena = ModelComparisonArena()
_judge = AutomatedJudge()
_leaderboard = get_leaderboard_store()
_tournament = TournamentRunner(leaderboard=_leaderboard, judge=_judge, arena=_arena)


class ModelTarget(BaseModel):
    model: str = Field(..., description="Model ID to evaluate")
    provider: str | None = Field(
        None, description="Explicit provider name (e.g. libra_lab, mock-provider, ollama)"
    )


class ArenaCompareRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Benchmark prompt to run across models")
    models: list[ModelTarget] = Field(
        ..., min_length=1, description="List of model targets to compare"
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(256, ge=1, le=2048, description="Maximum tokens to generate")
    system_prompt: str | None = Field(None, description="Optional system instruction")


class ArenaBattleRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Benchmark prompt for head-to-head battle")
    model_a: ModelTarget = Field(..., description="First competing model target")
    model_b: ModelTarget = Field(..., description="Second competing model target")
    category: str = Field(
        "overall", description="Evaluation domain (e.g. coding, reasoning, factual, instruction)"
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(256, ge=1, le=2048, description="Maximum tokens to generate")


class ArenaVoteRequest(BaseModel):
    model_a: str = Field(..., description="First competing model ID")
    model_b: str = Field(..., description="Second competing model ID")
    winner: str = Field(..., description="Winner: 'model_a', 'model_b', or 'tie'")
    category: str = Field("overall", description="Evaluation domain")
    prompt: str | None = Field(None, description="Optional prompt evaluated")


class ArenaTournamentRequest(BaseModel):
    models: list[ModelTarget] = Field(
        ..., min_length=2, description="At least 2 models for tournament"
    )
    categories: list[str] | None = Field(
        None, description="List of domains (coding, reasoning, factual, instruction)"
    )
    prompts_per_category: int = Field(1, ge=1, le=5, description="Prompts to evaluate per category")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(256, ge=1, le=2048)


@router.post(
    "/arena/compare", summary="Compare multiple models concurrently (metrics & throughput)"
)
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
    except (ValueError, TypeError, RuntimeError, OSError, KeyError, IndexError) as e:
        raise HTTPException(status_code=500, detail=f"Arena execution failed: {e!s}")


@router.get("/arena/leaderboard", summary="Get Bradley-Terry Elo model rankings")
async def get_leaderboard(
    category: str = Query(
        "overall", description="Category filter (e.g. overall, coding, reasoning)"
    ),
) -> list[dict[str, Any]]:
    """Returns models sorted by Bradley-Terry Elo rating with 95% confidence intervals and win rates."""
    records = _leaderboard.get_leaderboard(category=category)
    return [r.model_dump() for r in records]


@router.post("/arena/battle", summary="Run head-to-head battle with automated referee")
async def run_battle(request: ArenaBattleRequest) -> Any:
    """Executes completions for Model A and Model B, applies position-bias mitigated referee, and updates Elo."""
    try:
        prompt_item = BenchmarkPrompt(
            id="adhoc_battle",
            category=request.category,
            prompt=request.prompt,
        )
        match_rec, verdict, comp_a, comp_b = await _tournament.run_single_match(
            model_a=request.model_a.model,
            model_b=request.model_b.model,
            prompt_item=prompt_item,
            provider_a=request.model_a.provider,
            provider_b=request.model_b.provider,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return {
            "match": match_rec.model_dump(),
            "verdict": verdict.model_dump(),
            "completion_a": comp_a,
            "completion_b": comp_b,
        }
    except (ValueError, TypeError, RuntimeError, OSError, KeyError, IndexError) as e:
        raise HTTPException(status_code=500, detail=f"Battle execution failed: {e!s}")


@router.post("/arena/vote", summary="Record a human or custom vote in the arena")
async def record_vote(request: ArenaVoteRequest) -> Any:
    """Updates Elo ratings based on a human vote between two models."""
    if request.winner not in ("model_a", "model_b", "tie"):
        raise HTTPException(status_code=400, detail="Winner must be 'model_a', 'model_b', or 'tie'")

    match = _leaderboard.record_match(
        model_a=request.model_a,
        model_b=request.model_b,
        winner=request.winner,
        category=request.category,
        prompt=request.prompt,
    )
    return {
        "status": "success",
        "match": match.model_dump(),
        "model_a_rating": _leaderboard.get_model(request.model_a).rating
        if _leaderboard.get_model(request.model_a)
        else None,
        "model_b_rating": _leaderboard.get_model(request.model_b).rating
        if _leaderboard.get_model(request.model_b)
        else None,
    }


@router.post("/arena/tournament", summary="Run automated multi-model round-robin tournament")
async def run_tournament(request: ArenaTournamentRequest) -> Any:
    """Executes a round-robin tournament across benchmark categories, refereeing matches and updating Elo."""
    try:
        models_payload = [{"model": m.model, "provider": m.provider} for m in request.models]
        report = await _tournament.run_tournament(
            models=models_payload,
            categories=request.categories,
            prompts_per_category=request.prompts_per_category,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return report.model_dump()
    except (ValueError, TypeError, RuntimeError, OSError, KeyError, IndexError) as e:
        raise HTTPException(status_code=500, detail=f"Tournament failed: {e!s}")
