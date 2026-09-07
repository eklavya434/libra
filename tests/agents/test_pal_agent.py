"""
Unit & Integration Tests for Program-Aided Language Models (PAL) Agent
(packages/agents/pal.py)

Validates:
1. Code extraction from markdown fences and raw text.
2. PAL code normalization (ensuring solution() is invoked).
3. Exact mathematical & combinatorial reasoning execution.
4. Error handling on unrecoverable code or syntax failures.
"""

import pytest

from packages.agents import AgentStatus, PALAgent, extract_code, normalize_pal_code
from packages.providers.base import BaseProvider, ModelMetadata
from packages.providers.router import ProviderRouter
from packages.tools.sandbox import SafePythonSandbox


class ScriptedMockProvider(BaseProvider):
    """Deterministic mock provider returning scripted responses."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.idx = 0

    @property
    def name(self) -> str:
        return "scripted-mock"

    async def list_models(self) -> list[ModelMetadata]:
        return []

    async def chat(self, messages, model=None, **kwargs):
        if self.idx < len(self.responses):
            reply = self.responses[self.idx]
            self.idx += 1
        else:
            reply = "No more responses."
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
# 1. Code Extraction & Normalization Tests
# =============================================================================


def test_extract_code_python_fence():
    text = (
        "Here is the solution to your problem:\n\n"
        "```python\n"
        "def solution():\n"
        "    return 42 * 2\n"
        "```\n"
        "This computes the answer directly."
    )
    code = extract_code(text)
    assert code == "def solution():\n    return 42 * 2"


def test_extract_code_generic_fence():
    text = "Solution:\n```\nx = 10\ny = 20\nprint(x + y)\n```"
    code = extract_code(text)
    assert code == "x = 10\ny = 20\nprint(x + y)"


def test_extract_code_raw_fallback():
    text = (
        "Sure, here is the code:\n"
        "import math\n"
        "val = math.factorial(5)\n"
        "print(val)\n"
        "Hope this helps!"
    )
    code = extract_code(text)
    assert "math.factorial(5)" in code
    assert "Hope this helps" not in code


def test_normalize_pal_code_appends_solution_call():
    code = "def solution():\n    return sum(range(10))"
    normalized = normalize_pal_code(code)
    assert normalized.endswith("solution()")


def test_normalize_pal_code_preserves_existing_call():
    code = "def solution():\n    return sum(range(10))\n\nsolution()"
    normalized = normalize_pal_code(code)
    assert normalized.strip() == code.strip()


# =============================================================================
# 2. End-to-End PAL Execution Tests
# =============================================================================


@pytest.mark.asyncio
async def test_pal_agent_combinatorics_calculation():
    """Verifies that PAL executes combinatorics math to get the exact value."""
    # Problem: In how many ways can a committee of 5 be chosen from 12 people?
    # Expected: math.comb(12, 5) = 792
    model_reply = (
        "To find the number of ways to choose 5 people from 12, we calculate 12 C 5:\n"
        "```python\n"
        "import math\n"
        "def solution():\n"
        "    return math.comb(12, 5)\n"
        "```"
    )

    provider = ScriptedMockProvider([model_reply])
    router = ProviderRouter()
    router._providers["scripted-mock"] = provider

    sandbox = SafePythonSandbox(default_timeout=5.0)
    agent = PALAgent(
        model_id="scripted-mock",
        provider_name="scripted-mock",
        router=router,
        sandbox=sandbox,
    )

    trajectory = await agent.run("In how many ways can 5 people be chosen from 12?")
    assert trajectory.status == AgentStatus.COMPLETED
    assert trajectory.final_answer == "792"
    assert len(trajectory.steps) == 2
    assert trajectory.steps[1].is_final is True


@pytest.mark.asyncio
async def test_pal_agent_date_arithmetic():
    """Verifies PAL handles complex date/time arithmetic accurately via datetime module."""
    model_reply = (
        "```python\n"
        "from datetime import datetime, timedelta\n"
        "def solution():\n"
        "    start = datetime(2026, 1, 1)\n"
        "    target = start + timedelta(days=100)\n"
        "    return target.strftime('%Y-%m-%d')\n"
        "```"
    )

    provider = ScriptedMockProvider([model_reply])
    router = ProviderRouter()
    router._providers["scripted-mock"] = provider

    agent = PALAgent(
        model_id="scripted-mock",
        provider_name="scripted-mock",
        router=router,
    )

    trajectory = await agent.run("What is the date 100 days after January 1, 2026?")
    assert trajectory.status == AgentStatus.COMPLETED
    assert trajectory.final_answer == "2026-04-11"


@pytest.mark.asyncio
async def test_pal_agent_execution_runtime_error():
    """Verifies that runtime errors during execution mark status as FAILED with error message."""
    model_reply = "```python\ndef solution():\n    return 100 / 0\n```"

    provider = ScriptedMockProvider([model_reply])
    router = ProviderRouter()
    router._providers["scripted-mock"] = provider

    agent = PALAgent(
        model_id="scripted-mock",
        provider_name="scripted-mock",
        router=router,
    )

    trajectory = await agent.run("Divide 100 by zero")
    assert trajectory.status == AgentStatus.FAILED
    assert "ZeroDivisionError" in trajectory.error
