"""
Unit & Adversarial Tests for Code Generation & Auto-Debugging Agent
(packages/agents/coder.py)

Validates:
1. Clean code synthesis and immediate test assertion passing.
2. Test-driven auto-debugging recovering from SyntaxError.
3. Test-driven auto-debugging recovering from IndexError (off-by-one).
4. Test-driven auto-debugging recovering from AssertionError (failed unit tests).
5. Budget exhaustion guard when unfixable code exceeds max_iterations.
6. Sandbox security guard blocking forbidden imports (import os).
"""

import pytest

from packages.agents import (
    AgentStatus,
    AutoDebugger,
    CodeAgent,
    combine_code_and_assertions,
    format_code_with_lines,
)
from packages.providers.base import BaseProvider, ModelMetadata
from packages.providers.router import ProviderRouter


class ScriptedMockProvider(BaseProvider):
    """Deterministic mock provider returning scripted responses per step."""

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
# 1. Helper Unit Tests
# =============================================================================


def test_format_code_with_lines():
    code = "def add(a, b):\n    return a + b"
    formatted = format_code_with_lines(code)
    lines = formatted.splitlines()
    assert "  1 | def add(a, b):" in lines[0]
    assert "  2 |     return a + b" in lines[1]


def test_combine_code_and_assertions():
    code = "def foo(): return 1"
    assertions = "assert foo() == 1"
    combined = combine_code_and_assertions(code, assertions)
    assert "def foo(): return 1" in combined
    assert "Libra Unit Assertions" in combined
    assert "assert foo() == 1" in combined


# =============================================================================
# 2. Code Generation & Auto-Debugging Loop Tests
# =============================================================================


@pytest.mark.asyncio
async def test_code_generation_immediate_success():
    """Verifies that clean code passing assertions succeeds in a single attempt."""
    code_reply = (
        "```python\n"
        "def fibonacci(n: int) -> int:\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    a, b = 0, 1\n"
        "    for _ in range(2, n + 1):\n"
        "        a, b = b, a + b\n"
        "    return b\n"
        "```"
    )

    provider = ScriptedMockProvider([code_reply])
    router = ProviderRouter()
    router._providers["scripted-mock"] = provider

    agent = CodeAgent(model_id="scripted-mock", provider_name="scripted-mock", router=router)

    assertions = "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(7) == 13"
    traj = await agent.generate_and_test(
        "Write a function to compute n-th fibonacci number.", assertions=assertions
    )

    assert traj.success is True
    assert traj.total_attempts == 1
    assert "def fibonacci" in traj.final_code
    assert len(traj.iterations) == 1
    assert traj.iterations[0].success is True


@pytest.mark.asyncio
async def test_auto_debugger_syntax_error_recovery():
    """Verifies that the AutoDebugger catches SyntaxError and successfully repairs it."""
    broken_code = (
        "def is_prime(n: int) -> bool\n"  # Missing colon!
        "    if n < 2:\n"
        "        return False\n"
        "    for i in range(2, int(n ** 0.5) + 1):\n"
        "        if n % i == 0:\n"
        "            return False\n"
        "    return True\n"
    )

    repaired_code_reply = (
        "The syntax error was caused by a missing colon on line 1.\n"
        "Here is the corrected code:\n"
        "```python\n"
        "def is_prime(n: int) -> bool:\n"
        "    if n < 2:\n"
        "        return False\n"
        "    for i in range(2, int(n ** 0.5) + 1):\n"
        "        if n % i == 0:\n"
        "            return False\n"
        "    return True\n"
        "```"
    )

    provider = ScriptedMockProvider([repaired_code_reply])
    router = ProviderRouter()
    router._providers["scripted-mock"] = provider

    debugger = AutoDebugger(model_id="scripted-mock", provider_name="scripted-mock", router=router)

    assertions = (
        "assert is_prime(2) is True\nassert is_prime(4) is False\nassert is_prime(17) is True"
    )
    traj = await debugger.debug(
        "Check if n is prime", initial_code=broken_code, assertions=assertions, max_iterations=3
    )

    assert traj.success is True
    assert traj.total_attempts == 2
    assert len(traj.iterations) == 2
    # Iteration 1 failed with SyntaxError
    assert traj.iterations[0].success is False
    assert "SyntaxError" in traj.iterations[0].error
    # Iteration 2 succeeded
    assert traj.iterations[1].success is True
    assert "def is_prime(n: int) -> bool:" in traj.final_code


@pytest.mark.asyncio
async def test_auto_debugger_index_error_off_by_one():
    """Verifies that the AutoDebugger diagnoses and repairs an IndexError."""
    broken_code = (
        "def get_last(items: list):\n"
        "    return items[len(items)]\n"  # Off-by-one!
    )

    repaired_code_reply = (
        "The code attempted to index at `len(items)`, which raises an IndexError. "
        "Python indices are 0-based so the last item is at -1 or `len(items) - 1`.\n"
        "```python\n"
        "def get_last(items: list):\n"
        "    if not items:\n"
        "        return None\n"
        "    return items[-1]\n"
        "```"
    )

    provider = ScriptedMockProvider([repaired_code_reply])
    router = ProviderRouter()
    router._providers["scripted-mock"] = provider

    debugger = AutoDebugger(model_id="scripted-mock", provider_name="scripted-mock", router=router)

    assertions = "assert get_last([10, 20, 30]) == 30\nassert get_last(['a', 'b']) == 'b'"
    traj = await debugger.debug(
        "Get last item", initial_code=broken_code, assertions=assertions, max_iterations=3
    )

    assert traj.success is True
    assert traj.total_attempts == 2
    assert "IndexError" in traj.iterations[0].error
    assert traj.iterations[1].success is True
    assert "items[-1]" in traj.final_code


@pytest.mark.asyncio
async def test_auto_debugger_assertion_failure_recovery():
    """Verifies that the AutoDebugger recovers from an AssertionError when test cases fail."""
    # Attempt 1 has flawed logic (doesn't handle negative numbers)
    broken_code = "def clamp_positive(x: int) -> int:\n    return x\n"

    repaired_code_reply = (
        "The function returned the value as-is, violating the test assertion clamp_positive(-5) == 0.\n"
        "We clamp it using max(0, x).\n"
        "```python\n"
        "def clamp_positive(x: int) -> int:\n"
        "    return max(0, x)\n"
        "```"
    )

    provider = ScriptedMockProvider([repaired_code_reply])
    router = ProviderRouter()
    router._providers["scripted-mock"] = provider

    debugger = AutoDebugger(model_id="scripted-mock", provider_name="scripted-mock", router=router)

    assertions = "assert clamp_positive(10) == 10\nassert clamp_positive(-5) == 0\nassert clamp_positive(0) == 0"
    traj = await debugger.debug(
        "Clamp positive integers", initial_code=broken_code, assertions=assertions, max_iterations=3
    )

    assert traj.success is True
    assert traj.total_attempts == 2
    assert "AssertionError" in traj.iterations[0].error
    assert traj.iterations[1].success is True
    assert "max(0, x)" in traj.final_code


@pytest.mark.asyncio
async def test_auto_debugger_budget_exhaustion():
    """Verifies that when broken code cannot be repaired within max_iterations, it halts cleanly."""
    always_broken = "def broken(): return 1 / 0\n"
    # Provider always returns another broken snippet
    responses = [
        "```python\ndef broken(): return 2 / 0\n```",
        "```python\ndef broken(): return 3 / 0\n```",
    ]

    provider = ScriptedMockProvider(responses)
    router = ProviderRouter()
    router._providers["scripted-mock"] = provider

    debugger = AutoDebugger(model_id="scripted-mock", provider_name="scripted-mock", router=router)

    assertions = "assert broken() == 42"
    traj = await debugger.debug(
        "Unfixable division", initial_code=always_broken, assertions=assertions, max_iterations=2
    )

    assert traj.success is False
    assert traj.total_attempts == 2
    assert "ZeroDivisionError" in traj.error or "Max debug iterations" in traj.error


@pytest.mark.asyncio
async def test_code_agent_sandbox_security_rejection():
    """Verifies that attempting forbidden operations (e.g. import os) is rejected by the sandbox."""
    forbidden_code = "import os\ndef get_cwd():\n    return os.getcwd()\n"

    provider = ScriptedMockProvider([])
    router = ProviderRouter()
    router._providers["scripted-mock"] = provider

    debugger = AutoDebugger(model_id="scripted-mock", provider_name="scripted-mock", router=router)
    traj = await debugger.debug(
        "Get working directory", initial_code=forbidden_code, max_iterations=1
    )

    assert traj.success is False
    assert (
        "SandboxSecurityError" in traj.iterations[0].error
        or "disallowed module" in traj.iterations[0].error.lower()
    )


@pytest.mark.asyncio
async def test_code_agent_run_conformance():
    """Verifies that CodeAgent.run() conforms to the BaseAgent interface and returns an AgentTrajectory."""
    code_reply = "```python\ndef square(x: int) -> int:\n    return x * x\n```"

    provider = ScriptedMockProvider([code_reply])
    router = ProviderRouter()
    router._providers["scripted-mock"] = provider

    agent = CodeAgent(model_id="scripted-mock", provider_name="scripted-mock", router=router)
    trajectory = await agent.run("Write a function to square an integer.")

    assert trajectory.status == AgentStatus.COMPLETED
    assert "def square" in trajectory.final_answer
    assert len(trajectory.steps) >= 1
