"""
Phase 18 Interactive Demonstration: Multi-Step Autonomous Agent Loops

Demonstrates:
1. ReAct Agent executing multi-step reasoning with tools.
2. Real-time step event streaming (thought -> action -> observation).
3. Budget guardrails (max_steps limit & wall-clock timeout).
4. Circuit breaker tripping on consecutive tool failures.
5. Plan-and-Solve Agent deliberate milestone execution.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from packages.agents import (
    AgentStatus,
    PlanAndSolveAgent,
    ReActAgent,
)
from packages.providers.base import BaseProvider, ModelMetadata
from packages.providers.router import ProviderRouter
from packages.tools.builtin import CalculatorTool, PythonInterpreterTool
from packages.tools.registry import ToolRegistry


def print_banner(title: str) -> None:
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


class ScriptedAgentProvider(BaseProvider):
    """Deterministic script provider for demo execution."""

    def __init__(self, script: list[str]) -> None:
        self.script = list(script)
        self.idx = 0

    @property
    def name(self) -> str:
        return "scripted-demo"

    async def list_models(self) -> list[ModelMetadata]:
        return []

    async def chat(self, messages, model=None, **kwargs):
        if self.idx < len(self.script):
            out = self.script[self.idx]
            self.idx += 1
        else:
            out = "Thought: Done.\nFinal Answer: Task complete."
        return {"choices": [{"message": {"content": out}}]}

    async def stream(self, messages, model=None, **kwargs):
        yield ""

    async def health(self):
        return {"status": "healthy"}

    async def embeddings(self, texts, model=None):
        return [[0.0] * 16]

    def capabilities(self):
        return {"supports_text": True}


async def demo_react_agent_steps():
    print_banner("1. REACT AGENT INTERLEAVED REASONING & ACTIONS")
    script = [
        (
            "Thought: The user wants to calculate the area of a circle with radius 12. Area = pi * r^2.\n"
            "Action:\n"
            "<tool_call>\n"
            '{"name": "calculator", "arguments": {"expression": "3.14159265 * 12**2"}}\n'
            "</tool_call>"
        ),
        (
            "Thought: The calculator returned approximately 452.389.\n"
            "Final Answer: The area of a circle with radius 12 is approximately 452.39 square units."
        ),
    ]

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(PythonInterpreterTool())

    provider = ScriptedAgentProvider(script)
    router = ProviderRouter()
    router._providers["scripted-demo"] = provider

    agent = ReActAgent(
        model_id="scripted-demo",
        provider_name="scripted-demo",
        router=router,
        registry=registry,
    )

    print("[+] Prompt: 'What is the area of a circle with radius 12?'\n")
    trajectory = await agent.run("What is the area of a circle with radius 12?")

    for step in trajectory.steps:
        print(f"  [Step {step.step_number}]")
        print(f"    Thought:     {step.thought}")
        if step.action_name:
            print(f"    Action:      {step.action_name}({step.action_input})")
            print(f"    Observation: {step.observation}")
        if step.is_final:
            print(f"    Final Step:  Yes")
        print(f"    Duration:    {step.execution_time_ms}ms\n")

    print(f"[+] Final Answer: {trajectory.final_answer}")
    print(f"[+] Status:       {trajectory.status.value} (Total: {trajectory.total_duration_ms}ms)")


async def demo_react_streaming():
    print_banner("2. REAL-TIME AGENT STEP STREAMING")
    script = [
        'Thought: I need to calculate 144 / 12.\nAction:\n<tool_call>{"name": "calculator", "arguments": {"expression": "144 / 12"}}</tool_call>',
        'Thought: The quotient is 12.\nFinal Answer: 144 divided by 12 is 12.',
    ]
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    provider = ScriptedAgentProvider(script)
    router = ProviderRouter()
    router._providers["scripted-demo"] = provider

    agent = ReActAgent(
        model_id="scripted-demo",
        provider_name="scripted-demo",
        router=router,
        registry=registry,
    )

    print("[+] Streaming events as they fire:")
    async for event in agent.stream_steps("What is 144 / 12?"):
        ev_type = event.get("type")
        if ev_type == "thought":
            print(f"  --> [THOUGHT]     {event.get('content')}")
        elif ev_type == "action":
            print(f"  --> [ACTION]      {event.get('tool')}({event.get('arguments')})")
        elif ev_type == "observation":
            print(f"  --> [OBSERVATION] {event.get('observation')}")
        elif ev_type == "final_answer":
            print(f"  --> [FINAL]       {event.get('answer')}")


async def demo_guardrails():
    print_banner("3. BUDGET GUARDRAIL & CIRCUIT BREAKER DEFENSE")
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    # 3.1 Max Steps Limit
    script_infinite = [
        'Thought: Looping.\nAction:\n<tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}</tool_call>'
    ] * 10
    provider = ScriptedAgentProvider(script_infinite)
    router = ProviderRouter()
    router._providers["scripted-demo"] = provider

    agent_limited = ReActAgent(
        model_id="scripted-demo",
        provider_name="scripted-demo",
        router=router,
        registry=registry,
        max_steps=3,
    )
    print("\n[A] Runaway Loop Test (max_steps=3):")
    traj_limited = await agent_limited.run("Loop forever")
    print(f"  Status: {traj_limited.status.value}")
    print(f"  Error:  {traj_limited.error}")
    print(f"  Steps executed: {len(traj_limited.steps)}")

    # 3.2 Circuit Breaker
    script_broken = [
        'Thought: Calling missing tool.\nAction:\n<tool_call>{"name": "non_existent", "arguments": {}}</tool_call>'
    ] * 5
    provider_broken = ScriptedAgentProvider(script_broken)
    router_broken = ProviderRouter()
    router_broken._providers["scripted-demo"] = provider_broken

    agent_breaker = ReActAgent(
        model_id="scripted-demo",
        provider_name="scripted-demo",
        router=router_broken,
        registry=registry,
        max_tool_failures=2,
    )
    print("\n[B] Circuit Breaker Test (max_tool_failures=2):")
    traj_breaker = await agent_breaker.run("Trigger breaker")
    print(f"  Status: {traj_breaker.status.value}")
    print(f"  Error:  {traj_breaker.error}")
    print(f"  Steps executed: {len(traj_breaker.steps)}")


async def demo_plan_and_solve():
    print_banner("4. PLAN-AND-SOLVE AGENT (DELIBERATE PLANNING)")
    # Provider responds to planning, milestone, and synthesis
    class PlanProvider(BaseProvider):
        @property
        def name(self): return "plan-p"
        async def list_models(self): return []
        async def chat(self, messages, model=None, **kwargs):
            last = messages[-1]["content"]
            if "master planning agent" in last:
                return {"choices": [{"message": {"content": '{"problem_summary": "Factorial of 5", "steps": ["Calculate 5 * 4 * 3 * 2 * 1", "Verify result"]}'}}]}
            elif "milestone 1" in last:
                return {"choices": [{"message": {"content": 'Thought: Multiplying numbers.\n<tool_call>{"name": "calculator", "arguments": {"expression": "5 * 4 * 3 * 2 * 1"}}</tool_call>'}}]}
            elif "milestone 2" in last:
                return {"choices": [{"message": {"content": 'Verification: 120 is the exact factorial.'}}]}
            else:
                return {"choices": [{"message": {"content": "5 factorial (5!) is 120."}}]}
        async def stream(self, messages, model=None, **kwargs): yield ""
        async def health(self): return {"status": "healthy"}
        async def embeddings(self, texts, model=None): return [[0.0] * 16]
        def capabilities(self): return {"supports_text": True}

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    router = ProviderRouter()
    router._providers["plan-p"] = PlanProvider()

    agent = PlanAndSolveAgent(
        model_id="plan-p",
        provider_name="plan-p",
        router=router,
        registry=registry,
    )

    print("[+] Prompt: 'Calculate 5!'\n")
    traj = await agent.run("Calculate 5!")
    for step in traj.steps:
        print(f"  [Step {step.step_number}] {step.thought}")
        if step.action_name:
            print(f"    Tool: {step.action_name} -> {step.observation}")

    print(f"\n[+] Synthesized Answer: {traj.final_answer}")


async def main():
    print("=" * 75)
    print("      LIBRA PHASE 18: MULTI-STEP AUTONOMOUS AGENT LOOPS DEMO")
    print("=" * 75)
    await demo_react_agent_steps()
    await demo_react_streaming()
    await demo_guardrails()
    await demo_plan_and_solve()
    print_banner("PHASE 18 DEMO COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    asyncio.run(main())
