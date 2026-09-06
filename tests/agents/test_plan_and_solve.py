"""
Tests for PlanAndSolveAgent (packages/agents/plan_and_solve.py)
"""

import pytest
from packages.agents import AgentStatus, ExecutionPlan, PlanAndSolveAgent
from packages.providers.base import BaseProvider, ModelMetadata
from packages.providers.router import ProviderRouter
from packages.tools.builtin import CalculatorTool
from packages.tools.registry import ToolRegistry


class PlanSolveMockProvider(BaseProvider):
    """Simulates plan generation, milestone tool execution, and synthesis."""

    def __init__(self) -> None:
        self.call_idx = 0

    @property
    def name(self) -> str:
        return "plan-mock"

    async def list_models(self) -> list[ModelMetadata]:
        return []

    async def chat(self, messages, model=None, **kwargs):
        self.call_idx += 1
        last_prompt = messages[-1]["content"]

        if "master planning agent" in last_prompt:
            # Plan generation
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"problem_summary": "Calculate hypotenuse", '
                                '"steps": ["Compute a^2 + b^2", "Compute square root of sum"]}'
                            )
                        }
                    }
                ]
            }
        elif "milestone 1" in last_prompt:
            # Milestone 1: tool call
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                'Thought: Calculating 3^2 + 4^2 = 9 + 16 = 25.\n'
                                '<tool_call>{"name": "calculator", "arguments": {"expression": "3**2 + 4**2"}}</tool_call>'
                            )
                        }
                    }
                ]
            }
        elif "milestone 2" in last_prompt:
            # Milestone 2: tool call
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                'Thought: Taking sqrt of 25.\n'
                                '<tool_call>{"name": "calculator", "arguments": {"expression": "sqrt(25)"}}</tool_call>'
                            )
                        }
                    }
                ]
            }
        else:
            # Final synthesis
            return {
                "choices": [
                    {
                        "message": {
                            "content": "The hypotenuse of a right triangle with sides 3 and 4 is 5."
                        }
                    }
                ]
            }

    async def stream(self, messages, model=None, **kwargs):
        yield ""

    async def health(self):
        return {"status": "healthy"}

    async def embeddings(self, texts, model=None):
        return [[0.0] * 16]

    def capabilities(self):
        return {"supports_text": True}


@pytest.mark.asyncio
async def test_plan_and_solve_execution():
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    provider = PlanSolveMockProvider()
    router = ProviderRouter()
    router._providers["plan-mock"] = provider

    agent = PlanAndSolveAgent(
        model_id="plan-mock",
        provider_name="plan-mock",
        router=router,
        registry=registry,
    )

    trajectory = await agent.run("Find hypotenuse for triangle with legs 3 and 4")
    assert trajectory.status == AgentStatus.COMPLETED
    assert "hypotenuse" in trajectory.final_answer.lower()
    assert "5" in trajectory.final_answer
    # Steps: plan (1), milestone 1 (2), milestone 2 (3), synthesis (4)
    assert len(trajectory.steps) >= 3
