"""
Unit & Integration Tests for Multi-Agent Collaboration & Orchestration
(packages/agents/multi_agent.py & packages/agents/message_bus.py)

Validates:
1. SharedBlackboard & AgentMessage bus state manipulation and dialogue filtering.
2. 4-Agent collaborative consensus convergence on Round 1.
3. Adversarial review triggering revision in Round 2.
4. Sandbox test assertion failure triggering code revision in Round 2.
5. Maximum rounds ceiling termination when consensus cannot be reached.
6. Real-time collaboration event streaming.
"""

import pytest

from packages.agents import (
    AgentMessage,
    AgentRole,
    AgentStatus,
    CollaborativeTeam,
    MessageType,
    SharedBlackboard,
)
from packages.providers.base import BaseProvider, ModelMetadata
from packages.providers.router import ProviderRouter


class TeamScriptedProvider(BaseProvider):
    """Deterministic scripted mock provider dispatching replies based on system prompt."""

    def __init__(self, responses_by_role: dict[str, list[str]]) -> None:
        self.responses_by_role = {k: list(v) for k, v in responses_by_role.items()}
        self.indices: dict[str, int] = {k: 0 for k in responses_by_role}

    @property
    def name(self) -> str:
        return "team-mock"

    async def list_models(self) -> list[ModelMetadata]:
        return []

    async def chat(self, messages, model=None, **kwargs):
        sys_prompt = messages[0]["content"] if messages else ""

        if "Adversarial Senior Code Reviewer" in sys_prompt:
            role = "reviewer"
        elif "Lead Software Architect" in sys_prompt:
            role = "architect"
        elif "Senior Implementation Engineer" in sys_prompt:
            role = "coder"
        elif (
            "Automated QA and Testing Specialist" in sys_prompt
            or "Testing Specialist" in sys_prompt
        ):
            role = "tester"
        else:
            role = "default"

        role_list = self.responses_by_role.get(role, ["Default agent response."])
        idx = self.indices.get(role, 0)
        if idx < len(role_list):
            reply = role_list[idx]
            self.indices[role] = idx + 1
        else:
            reply = role_list[-1]

        return {"choices": [{"message": {"content": reply}}]}

    async def stream(self, messages, model=None, **kwargs):
        yield ""

    async def health(self):
        return {"status": "healthy"}

    async def embeddings(self, texts, model=None):
        return [[0.0] * 16]

    def capabilities(self):
        return {"supports_text": True}


# =============================================================================
# 1. Blackboard & Message Bus Tests
# =============================================================================


def test_shared_blackboard_operations():
    bb = SharedBlackboard(task="Implement binary search")
    msg1 = AgentMessage(
        sender_name="Architect",
        sender_role=AgentRole.ARCHITECT,
        recipient="broadcast",
        message_type=MessageType.SPECIFICATION,
        content="Binary search specification: def binary_search(arr, target) -> int",
    )
    bb.post_message(msg1)

    assert len(bb.messages) == 1
    assert bb.messages[0].sender_name == "Architect"
    assert "binary_search" in bb.format_dialogue_history()

    msgs_for_coder = bb.get_messages_for("Coder", AgentRole.CODER)
    assert len(msgs_for_coder) == 1


# =============================================================================
# 2. Multi-Agent Consensus Workflow (Round 1 Success)
# =============================================================================


@pytest.mark.asyncio
async def test_multi_agent_consensus_round_1():
    """Verifies that when Coder writes good code, Reviewer approves and Tester passes on Round 1."""
    role_responses = {
        "architect": [
            "Specification:\nDefine function `def add(a: int, b: int) -> int` that returns the sum."
        ],
        "coder": ["```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```"],
        "reviewer": ["Audit complete. Code is clean and matches specification.\nVERDICT: APPROVED"],
        "tester": [
            "```python\nassert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(0, 0) == 0\n```"
        ],
    }

    provider = TeamScriptedProvider(role_responses)
    router = ProviderRouter()
    router._providers["team-mock"] = provider

    team = CollaborativeTeam(
        model_id="team-mock",
        provider_name="team-mock",
        router=router,
        max_rounds=3,
    )

    trajectory = await team.collaborate("Implement an addition function.")
    assert trajectory.status == AgentStatus.COMPLETED
    assert trajectory.consensus_reached is True
    assert trajectory.total_rounds == 1
    assert len(trajectory.rounds) == 1
    assert trajectory.rounds[0].reviewer_approved is True
    assert trajectory.rounds[0].tests_passed is True
    assert "def add" in trajectory.final_code


# =============================================================================
# 3. Adversarial Review Loop (Revision in Round 2)
# =============================================================================


@pytest.mark.asyncio
async def test_multi_agent_reviewer_revision_loop():
    """Verifies that Reviewer's rejection in Round 1 prompts Coder to revise in Round 2."""
    role_responses = {
        "architect": [
            "Specification: Function `def safe_divide(a: float, b: float) -> float | None` returning None on 0."
        ],
        "coder": [
            # Round 1 (Buggy: does not check 0)
            "```python\ndef safe_divide(a: float, b: float):\n    return a / b\n```",
            # Round 2 (Fixed)
            "```python\ndef safe_divide(a: float, b: float):\n    if b == 0:\n        return None\n    return a / b\n```",
        ],
        "reviewer": [
            # Round 1 (Critique)
            "Code does not handle division by zero when b == 0.\nVERDICT: REVISION_REQUESTED",
            # Round 2 (Approved)
            "Now handles b == 0 correctly returning None.\nVERDICT: APPROVED",
        ],
        "tester": [
            "```python\nassert safe_divide(10, 2) == 5.0\nassert safe_divide(5, 0) is None\n```"
        ],
    }

    provider = TeamScriptedProvider(role_responses)
    router = ProviderRouter()
    router._providers["team-mock"] = provider

    team = CollaborativeTeam(
        model_id="team-mock",
        provider_name="team-mock",
        router=router,
        max_rounds=3,
    )

    trajectory = await team.collaborate("Implement safe division.")
    assert trajectory.status == AgentStatus.COMPLETED
    assert trajectory.consensus_reached is True
    assert trajectory.total_rounds == 2
    # Round 1 was rejected
    assert trajectory.rounds[0].reviewer_approved is False
    # Round 2 was approved and passed
    assert trajectory.rounds[1].reviewer_approved is True
    assert trajectory.rounds[1].tests_passed is True
    assert "if b == 0:" in trajectory.final_code


# =============================================================================
# 4. Sandbox Test Failure Loop (Revision in Round 2)
# =============================================================================


@pytest.mark.asyncio
async def test_multi_agent_test_failure_triggers_revision():
    """Verifies that failed sandbox assertions prompt Coder to fix implementation in Round 2."""
    role_responses = {
        "architect": ["Specification: `def is_even(n: int) -> bool`"],
        "coder": [
            # Round 1: buggy (returns True for odd)
            "```python\ndef is_even(n: int) -> bool:\n    return n % 2 != 0\n```",
            # Round 2: fixed
            "```python\ndef is_even(n: int) -> bool:\n    return n % 2 == 0\n```",
        ],
        "reviewer": [
            # Reviewer mistakenly approved in Round 1
            "Syntax looks clean.\nVERDICT: APPROVED",
            # Reviewer approves in Round 2
            "VERDICT: APPROVED",
        ],
        "tester": ["```python\nassert is_even(4) is True\nassert is_even(5) is False\n```"],
    }

    provider = TeamScriptedProvider(role_responses)
    router = ProviderRouter()
    router._providers["team-mock"] = provider

    team = CollaborativeTeam(
        model_id="team-mock",
        provider_name="team-mock",
        router=router,
        max_rounds=3,
    )

    trajectory = await team.collaborate("Implement is_even function.")
    assert trajectory.status == AgentStatus.COMPLETED
    assert trajectory.consensus_reached is True
    assert trajectory.total_rounds == 2
    # Round 1 tests failed (even though reviewer approved)
    assert trajectory.rounds[0].tests_passed is False
    # Round 2 tests passed
    assert trajectory.rounds[1].tests_passed is True
    assert "n % 2 == 0" in trajectory.final_code


# =============================================================================
# 5. Max Rounds Ceiling Termination
# =============================================================================


@pytest.mark.asyncio
async def test_multi_agent_max_rounds_exhaustion():
    """Verifies that team terminates with FAILED status when consensus is never reached."""
    role_responses = {
        "architect": ["Specification: Simple spec"],
        "coder": ["```python\ndef fail(): return 0\n```"],
        "reviewer": ["Still broken.\nVERDICT: REVISION_REQUESTED"],
        "tester": ["```python\nassert fail() == 100\n```"],
    }

    provider = TeamScriptedProvider(role_responses)
    router = ProviderRouter()
    router._providers["team-mock"] = provider

    team = CollaborativeTeam(
        model_id="team-mock",
        provider_name="team-mock",
        router=router,
        max_rounds=2,
    )

    trajectory = await team.collaborate("Implement impossible task.")
    assert trajectory.status == AgentStatus.FAILED
    assert trajectory.consensus_reached is False
    assert trajectory.total_rounds == 2
    assert "Best-Effort" in trajectory.final_answer


# =============================================================================
# 6. Real-Time Streaming Events
# =============================================================================


@pytest.mark.asyncio
async def test_multi_agent_streaming_events():
    """Verifies that stream_collaboration emits start, message, and finish events."""
    role_responses = {
        "architect": ["Specification: Add numbers"],
        "coder": ["```python\ndef add(a, b): return a + b\n```"],
        "reviewer": ["VERDICT: APPROVED"],
        "tester": ["```python\nassert add(1, 2) == 3\n```"],
    }

    provider = TeamScriptedProvider(role_responses)
    router = ProviderRouter()
    router._providers["team-mock"] = provider

    team = CollaborativeTeam(
        model_id="team-mock",
        provider_name="team-mock",
        router=router,
        max_rounds=1,
    )

    event_types = []
    async for event in team.stream_collaboration("Add two numbers"):
        event_types.append(event.get("type"))

    assert "start" in event_types
    assert "message" in event_types
    assert "round_start" in event_types
    assert "finish" in event_types
