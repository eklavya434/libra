"""
Libra API v1 - Code Generation, PAL & Auto-Debugging Endpoints

Provides REST endpoints for:
1. Program-Aided Language Models (PAL) math and symbolic reasoning.
2. Code generation with automated test assertions and self-correction.
3. Standalone test-driven auto-debugging of user-provided buggy code.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from packages.agents import (
    AgentTrajectory,
    AutoDebugger,
    CodeAgent,
    CodeTrajectory,
    PALAgent,
)

router = APIRouter()


class PALRequest(BaseModel):
    """Payload for Program-Aided Language Model solving."""

    prompt: str = Field(..., description="Math, logic, or combinatorics word problem")
    model_id: str = Field("mock-model", description="Model ID to generate the Python code")
    provider_name: str | None = Field(None, description="Optional provider override")
    timeout_sec: float = Field(30.0, ge=5.0, le=120.0, description="Execution timeout in seconds")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")


class CodeGenerateRequest(BaseModel):
    """Payload for synthesizing code with optional unit tests and auto-debugging."""

    prompt: str = Field(..., description="Software engineering task specification")
    assertions: str | None = Field(
        None, description="Unit test assertions to verify against (e.g. 'assert func(2) == 4')"
    )
    model_id: str = Field("mock-model", description="Model ID to generate and repair code")
    provider_name: str | None = Field(None, description="Optional provider override")
    max_iterations: int = Field(
        3, ge=1, le=10, description="Maximum self-correction repair attempts"
    )
    timeout_sec: float = Field(60.0, ge=5.0, le=300.0, description="Overall execution timeout")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")


class DebugRequest(BaseModel):
    """Payload for debugging an existing broken code snippet."""

    task_description: str = Field(
        ..., description="Description of what the code is intended to accomplish"
    )
    code: str = Field(..., description="Broken Python code snippet")
    assertions: str | None = Field(
        None, description="Optional test assertions that fail on this code"
    )
    model_id: str = Field("mock-model", description="Model ID to diagnose and fix the bug")
    provider_name: str | None = Field(None, description="Optional provider override")
    max_iterations: int = Field(
        3, ge=1, le=10, description="Maximum self-correction repair attempts"
    )
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")


@router.post(
    "/pal", response_model=AgentTrajectory, summary="Run Program-Aided Language Model (PAL)"
)
async def run_pal(request: PALRequest) -> AgentTrajectory:
    """Solves symbolic, arithmetic, or algorithmic problems via Python code synthesis and execution."""
    agent = PALAgent(
        model_id=request.model_id,
        provider_name=request.provider_name,
        timeout_sec=request.timeout_sec,
        temperature=request.temperature,
    )
    return await agent.run(request.prompt)


@router.post(
    "/generate", response_model=CodeTrajectory, summary="Generate Code with Test Verification"
)
async def generate_code(request: CodeGenerateRequest) -> CodeTrajectory:
    """Synthesizes code according to task specification, runs assertions, and auto-repairs any failures."""
    agent = CodeAgent(
        model_id=request.model_id,
        provider_name=request.provider_name,
        max_debug_iterations=request.max_iterations,
        timeout_sec=request.timeout_sec,
        temperature=request.temperature,
    )
    return await agent.generate_and_test(
        prompt=request.prompt,
        assertions=request.assertions,
        max_iterations=request.max_iterations,
    )


@router.post("/debug", response_model=CodeTrajectory, summary="Auto-Debug Existing Code")
async def debug_code(request: DebugRequest) -> CodeTrajectory:
    """Takes a broken Python code snippet, runs it in the sandbox, diagnoses failure, and iteratively fixes it."""
    debugger = AutoDebugger(
        model_id=request.model_id,
        provider_name=request.provider_name,
        temperature=request.temperature,
    )
    return await debugger.debug(
        task_description=request.task_description,
        initial_code=request.code,
        assertions=request.assertions,
        max_iterations=request.max_iterations,
    )
