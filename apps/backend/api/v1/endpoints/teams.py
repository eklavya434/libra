"""
Libra API v1 - Multi-Agent Collaborative Team Endpoints

Provides REST endpoints for running role-specialized multi-agent teams:
1. Synchronous team collaboration returning the full TeamTrajectory.
2. Real-time Server-Sent Events (SSE) streaming of inter-agent dialogue and round progress.
"""

from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from packages.agents import CollaborativeTeam, TeamTrajectory

router = APIRouter()


class TeamRunRequest(BaseModel):
    """Payload to launch a multi-agent collaborative session."""

    prompt: str = Field(..., description="The user task or engineering objective to solve")
    model_id: str = Field("mock-model", description="Model ID powering the specialized agents")
    provider_name: str | None = Field(None, description="Optional provider override")
    max_rounds: int = Field(3, ge=1, le=10, description="Maximum collaboration rounds")
    timeout_sec: float = Field(
        120.0, ge=10.0, le=300.0, description="Wall-clock timeout in seconds"
    )
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")


@router.post(
    "/collaborate",
    response_model=TeamTrajectory,
    summary="Run Multi-Agent Collaboration Synchronously",
)
async def collaborate_team(request: TeamRunRequest) -> TeamTrajectory:
    """Executes a collaborative multi-agent workflow (Architect -> Coder -> Reviewer -> Tester)."""
    team = CollaborativeTeam(
        model_id=request.model_id,
        provider_name=request.provider_name,
        max_rounds=request.max_rounds,
        timeout_sec=request.timeout_sec,
        temperature=request.temperature,
    )
    return await team.collaborate(request.prompt)


@router.post("/collaborate/stream", summary="Stream Multi-Agent Collaboration via SSE")
async def stream_collaborate_team(request: TeamRunRequest) -> StreamingResponse:
    """Streams real-time inter-agent messages, code proposals, reviews, and test runs via SSE."""
    team = CollaborativeTeam(
        model_id=request.model_id,
        provider_name=request.provider_name,
        max_rounds=request.max_rounds,
        timeout_sec=request.timeout_sec,
        temperature=request.temperature,
    )

    async def event_generator():
        async for event in team.stream_collaboration(request.prompt):
            clean_event = {}
            for k, v in event.items():
                if hasattr(v, "model_dump"):
                    clean_event[k] = v.model_dump()
                else:
                    clean_event[k] = v
            yield f"data: {json.dumps(clean_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
