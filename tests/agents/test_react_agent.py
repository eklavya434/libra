"""
Tests for ReActAgent (packages/agents/react.py)

Validates:
1. Single-step direct answers without tool calling.
2. Multi-step reasoning with tool invocation and observation integration.
3. Budget enforcement when max_steps is exceeded.
4. Circuit breaker termination on consecutive tool errors.
5. Real-time step event streaming.
"""

import pytest

from packages.agents import AgentStatus, ReActAgent
from packages.providers.base import BaseProvider, ModelMetadata
from packages.providers.router import ProviderRouter
from packages.tools.builtin import CalculatorTool
from packages.tools.registry import ToolRegistry


class StepMockProvider(BaseProvider):
    """Deterministic mock provider yielding scripted responses per step."""

    def __init__(self, script: list[str]) -> None:
        self.script = list(script)
        self.call_idx = 0

    @property
    def name(self) -> str:
        return "step-mock"

    async def list_models(self) -> list[ModelMetadata]:
        return []

    async def chat(self, messages, model=None, **kwargs):
        if self.call_idx < len(self.script):
            reply = self.script[self.call_idx]
            self.call_idx += 1
        else:
            reply = "Final Answer: Script ended."
        return {"choices": [{"message": {"content": reply}}]}

    async def stream(self, messages, model=None, **kwargs):
        yield ""

    async def health(self):
        return {"status": "healthy"}

    async def embeddings(self, texts, model=None):
        return [[0.0] * 16]

    def capabilities(self):
        return {"supports_text": True}


@pytest.fixture
def test_registry():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    return registry


@pytest.mark.asyncio
async def test_react_direct_answer(test_registry):
    provider = StepMockProvider(
        ["Thought: I know the answer directly.\nFinal Answer: The capital of France is Paris."]
    )
    router = ProviderRouter()
    router._providers["step-mock"] = provider

    agent = ReActAgent(
        model_id="step-mock",
        provider_name="step-mock",
        router=router,
        registry=test_registry,
    )

    trajectory = await agent.run("What is the capital of France?")
    assert trajectory.status == AgentStatus.COMPLETED
    assert trajectory.final_answer == "The capital of France is Paris."
    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].is_final is True


@pytest.mark.asyncio
async def test_react_multi_step_calculation(test_registry):
    script = [
        # Step 1: Tool call
        'Thought: I need to calculate 25 * 4.\nAction:\n<tool_call>{"name": "calculator", "arguments": {"expression": "25 * 4"}}</tool_call>',
        # Step 2: Final answer using observation
        "Thought: The calculator returned 100.\nFinal Answer: 25 multiplied by 4 equals 100.",
    ]
    provider = StepMockProvider(script)
    router = ProviderRouter()
    router._providers["step-mock"] = provider

    agent = ReActAgent(
        model_id="step-mock",
        provider_name="step-mock",
        router=router,
        registry=test_registry,
    )

    trajectory = await agent.run("Calculate 25 times 4")
    assert trajectory.status == AgentStatus.COMPLETED
    assert "100" in trajectory.final_answer
    assert len(trajectory.steps) == 2

    # Verify step 1 tool execution
    step_1 = trajectory.steps[0]
    assert step_1.action_name == "calculator"
    assert step_1.action_input == {"expression": "25 * 4"}
    assert "100" in step_1.observation
    assert step_1.is_final is False

    # Verify step 2 final answer
    step_2 = trajectory.steps[1]
    assert step_2.is_final is True


@pytest.mark.asyncio
async def test_react_max_steps_budget_limit(test_registry):
    # Provider never gives final answer, loops calling calculator
    script = [
        'Thought: Looping step.\nAction:\n<tool_call>{"name": "calculator", "arguments": {"expression": "1 + 1"}}</tool_call>'
    ] * 5

    provider = StepMockProvider(script)
    router = ProviderRouter()
    router._providers["step-mock"] = provider

    agent = ReActAgent(
        model_id="step-mock",
        provider_name="step-mock",
        router=router,
        registry=test_registry,
        max_steps=3,
    )

    trajectory = await agent.run("Loop forever")
    assert trajectory.status == AgentStatus.FAILED
    assert "exceeded maximum step limit" in trajectory.error
    assert len(trajectory.steps) == 3


@pytest.mark.asyncio
async def test_react_circuit_breaker(test_registry):
    # Provider repeatedly calls a non-existent tool
    script = [
        'Thought: Trying broken tool.\nAction:\n<tool_call>{"name": "missing_tool", "arguments": {}}</tool_call>'
    ] * 5

    provider = StepMockProvider(script)
    router = ProviderRouter()
    router._providers["step-mock"] = provider

    agent = ReActAgent(
        model_id="step-mock",
        provider_name="step-mock",
        router=router,
        registry=test_registry,
        max_tool_failures=2,
    )

    trajectory = await agent.run("Break tool")
    assert trajectory.status == AgentStatus.FAILED
    assert "Circuit breaker tripped" in trajectory.error
    assert len(trajectory.steps) == 2


@pytest.mark.asyncio
async def test_react_streaming_events(test_registry):
    script = [
        'Thought: Calculating.\nAction:\n<tool_call>{"name": "calculator", "arguments": {"expression": "9 * 9"}}</tool_call>',
        "Thought: Concluding.\nFinal Answer: 81.",
    ]
    provider = StepMockProvider(script)
    router = ProviderRouter()
    router._providers["step-mock"] = provider

    agent = ReActAgent(
        model_id="step-mock",
        provider_name="step-mock",
        router=router,
        registry=test_registry,
    )

    events = []
    async for event in agent.stream_steps("What is 9*9?"):
        events.append(event["type"])

    assert "start" in events
    assert "step_start" in events
    assert "thought" in events
    assert "action" in events
    assert "observation" in events
    assert "final_answer" in events
    assert "finish" in events
