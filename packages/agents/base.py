"""
Libra Agents Package - Base Agent Abstractions & Trajectory Containers

Defines the fundamental data structures for recording agent step trajectories,
execution statuses, and abstract base agent interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class AgentStep(BaseModel):
    """Encapsulates a single reasoning-action-observation step in an agent trajectory."""

    step_number: int = Field(..., description="1-indexed sequence number of the step")
    thought: str = Field("", description="Internal reasoning leading to action or final answer")
    action_name: Optional[str] = Field(None, description="Tool invoked, or None if direct answer")
    action_input: Optional[dict[str, Any]] = Field(None, description="Arguments passed to the tool")
    observation: Optional[str] = Field(None, description="Tool output or error string observed")
    is_final: bool = Field(
        False, description="Whether this step concluded the trajectory with final answer"
    )
    execution_time_ms: float = Field(0.0, description="Step duration in milliseconds")


class AgentTrajectory(BaseModel):
    """Complete execution history and summary of an agent run."""

    prompt: str = Field(..., description="Initial task prompt given to the agent")
    status: AgentStatus = Field(AgentStatus.IDLE, description="Final outcome status of the run")
    steps: list[AgentStep] = Field(
        default_factory=list, description="Ordered sequence of executed steps"
    )
    final_answer: Optional[str] = Field(
        None, description="Synthesized final answer delivered to user"
    )
    error: Optional[str] = Field(None, description="Error message if run failed or timed out")
    total_duration_ms: float = Field(
        0.0, description="Total execution wall-clock time in milliseconds"
    )


class BaseAgent(ABC):
    """Abstract Base Class for all autonomous agent implementations in Libra."""

    def __init__(
        self,
        max_steps: int = 10,
        timeout_sec: float = 60.0,
        max_tool_failures: int = 3,
    ) -> None:
        self.max_steps = max_steps
        self.timeout_sec = timeout_sec
        self.max_tool_failures = max_tool_failures

    @abstractmethod
    async def run(self, prompt: str) -> AgentTrajectory:
        """Executes the agent loop to completion for a given user prompt."""
        pass
