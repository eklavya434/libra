"""
Libra API v1 - Autonomous Agent Endpoints

Provides endpoints for running ReAct and Plan-and-Solve multi-step agents
with synchronous and real-time Server-Sent Event (SSE) trajectory streaming.
"""

from __future__ import annotations

import json
from typing import Any, Optional
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from packages.agents import (
    AgentTrajectory,
    PlanAndSolveAgent,
    ReActAgent,
)

router = APIRouter()


class AgentRunRequest(BaseModel):
    """Payload to launch an autonomous agent execution."""

    prompt: str = Field(..., description="The user task or inquiry to solve")
    model_id: str = Field("mock-model", description="Model identifier to power the agent")
    provider_name: Optional[str] = Field(None, description="Optional provider override")
    max_steps: int = Field(10, ge=1, le=30, description="Maximum number of reasoning steps")
    timeout_sec: float = Field(60.0, ge=5.0, le=300.0, description="Wall-clock timeout in seconds")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")


@router.post("/react", response_model=AgentTrajectory, summary="Run ReAct Agent synchronously")
async def run_react_agent(request: AgentRunRequest) -> AgentTrajectory:
    """Execute a ReAct agent loop to completion, returning the complete trajectory and final answer."""
    agent = ReActAgent(
        model_id=request.model_id,
        provider_name=request.provider_name,
        max_steps=request.max_steps,
        timeout_sec=request.timeout_sec,
        temperature=request.temperature,
    )
    return await agent.run(request.prompt)


@router.post("/react/stream", summary="Stream ReAct Agent execution steps via SSE")
async def stream_react_agent(request: AgentRunRequest) -> StreamingResponse:
    """Stream ReAct agent thought, action, observation, and final answer events in real-time."""
    agent = ReActAgent(
        model_id=request.model_id,
        provider_name=request.provider_name,
        max_steps=request.max_steps,
        timeout_sec=request.timeout_sec,
        temperature=request.temperature,
    )

    async def event_generator():
        async for event in agent.stream_steps(request.prompt):
            # Serialize Pydantic objects if contained in event (e.g. trajectory)
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


@router.post("/plan-and-solve", response_model=AgentTrajectory, summary="Run Plan-and-Solve Agent")
async def run_plan_and_solve_agent(request: AgentRunRequest) -> AgentTrajectory:
    """Execute a Plan-and-Solve agent, returning milestone plan and synthesized answer."""
    agent = PlanAndSolveAgent(
        model_id=request.model_id,
        provider_name=request.provider_name,
        max_steps=request.max_steps,
        timeout_sec=request.timeout_sec,
        temperature=request.temperature,
    )
    return await agent.run(request.prompt)
