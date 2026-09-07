"""
Libra Agents Package - Autonomous Agent Orchestration Engine

Exports:
- BaseAgent: Abstract agent interface
- AgentStep, AgentTrajectory, AgentStatus: Execution trace structures
- ReActAgent: Interleaved reasoning and action agent loop
- PlanAndSolveAgent: Structured planning and sequential execution agent
- PALAgent: Program-Aided Language Models for math & symbolic logic
- CodeAgent: Code generation & self-evaluating agent
- AutoDebugger: Test-driven reflective error diagnosis and repair
- CodeTrajectory, DebugIteration: Code debugging traces
"""

from packages.agents.base import (
    AgentStatus,
    AgentStep,
    AgentTrajectory,
    BaseAgent,
)
from packages.agents.coder import (
    AutoDebugger,
    CodeAgent,
    CodeTrajectory,
    DebugIteration,
    combine_code_and_assertions,
    format_code_with_lines,
)
from packages.agents.pal import PALAgent, extract_code, normalize_pal_code
from packages.agents.plan_and_solve import ExecutionPlan, PlanAndSolveAgent
from packages.agents.react import ReActAgent

__all__ = [
    "BaseAgent",
    "AgentStep",
    "AgentTrajectory",
    "AgentStatus",
    "ReActAgent",
    "PlanAndSolveAgent",
    "ExecutionPlan",
    "PALAgent",
    "extract_code",
    "normalize_pal_code",
    "CodeAgent",
    "AutoDebugger",
    "CodeTrajectory",
    "DebugIteration",
    "format_code_with_lines",
    "combine_code_and_assertions",
]
