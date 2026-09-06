"""
Libra Agents Package - Plan-and-Solve Agent

Implements a two-stage deliberate agent:
1. Decomposes problem into an explicit multi-step plan.
2. Executes each plan item sequentially with tool execution.
3. Synthesizes findings into a unified final answer.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional
from pydantic import BaseModel, Field

from packages.agents.base import AgentStatus, AgentStep, AgentTrajectory, BaseAgent
from packages.providers.base import BaseProvider
from packages.providers.router import ProviderRouter, get_router
from packages.providers.structured import StructuredOutputGenerator
from packages.tools.parser import ToolCallParser
from packages.tools.registry import ToolRegistry, get_tool_registry


class ExecutionPlan(BaseModel):
    """Structured plan composed of discrete sequential tasks."""

    problem_summary: str = Field(..., description="Brief restatement of the problem")
    steps: List[str] = Field(..., min_length=1, description="Sequential steps required to solve the problem")


class PlanAndSolveAgent(BaseAgent):
    """
    Deliberate agent that explicitly plans milestones before tool execution.
    Reduces thrashing on complex multi-step reasoning tasks.
    """

    def __init__(
        self,
        model_id: str = "mock-model",
        provider_name: Optional[str] = None,
        router: Optional[ProviderRouter] = None,
        registry: Optional[ToolRegistry] = None,
        max_steps: int = 10,
        timeout_sec: float = 60.0,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(max_steps=max_steps, timeout_sec=timeout_sec)
        self.model_id = model_id
        self.provider_name = provider_name
        self.router = router or get_router()
        self.registry = registry or get_tool_registry()
        self.temperature = temperature
        self.structured_gen = StructuredOutputGenerator(router=self.router)

    async def run(self, prompt: str) -> AgentTrajectory:
        """Executes the Plan-and-Solve lifecycle."""
        t0 = time.perf_counter()
        agent_steps: list[AgentStep] = []

        # 1. Planning Stage
        planning_prompt = (
            f"You are a master planning agent. Given the following user request:\n"
            f"'{prompt}'\n\n"
            f"Formulate a clear, sequential, step-by-step plan to solve it. Keep it concise (2 to 5 steps)."
        )

        plan_res = await self.structured_gen.generate(
            prompt=planning_prompt,
            schema=ExecutionPlan,
            model_id=self.model_id,
            provider_name=self.provider_name,
            max_retries=1,
        )

        plan: ExecutionPlan
        if plan_res.success and isinstance(plan_res.data, ExecutionPlan):
            plan = plan_res.data
        else:
            # Fallback heuristic plan if structured model failed
            plan = ExecutionPlan(
                problem_summary=prompt,
                steps=[f"Investigate problem: {prompt}", "Synthesize final answer"],
            )

        step_1 = AgentStep(
            step_number=1,
            thought=f"Devised plan with {len(plan.steps)} milestones: {', '.join(plan.steps)}",
            is_final=False,
            execution_time_ms=plan_res.latency_ms,
        )
        agent_steps.append(step_1)

        # 2. Solving Stage
        provider: BaseProvider = await self.router.resolve_provider_for_model(
            model_id=self.model_id,
            requested_provider=self.provider_name,
        )

        context_notes: list[str] = []

        for idx, milestone in enumerate(plan.steps, start=2):
            if idx > self.max_steps:
                break

            m_start = time.perf_counter()
            milestone_prompt = (
                f"You are executing milestone {idx-1} of {len(plan.steps)}: '{milestone}'.\n"
                f"Original Goal: '{prompt}'\n"
                f"Prior Findings:\n" + ("\n".join(context_notes) if context_notes else "None") + "\n\n"
                "If you need a tool, emit a <tool_call> block. Otherwise, summarize your finding for this step."
            )

            try:
                if hasattr(provider, "chat"):
                    res = await provider.chat(
                        messages=[{"role": "user", "content": milestone_prompt}],
                        model=self.model_id,
                        temperature=self.temperature,
                    )
                    raw_out = res["choices"][0]["message"]["content"] if isinstance(res, dict) else str(res)
                else:
                    raw_out = f"Completed milestone: {milestone}"
            except Exception as e:
                raw_out = f"Error during milestone {idx}: {e}"

            # Check if milestone triggered tool call
            calls = ToolCallParser.parse(raw_out)
            action_name = None
            action_input = None
            obs_text = None

            if calls:
                call = calls[0]
                action_name = call.name
                action_input = call.arguments
                tool_res = await self.registry.execute_tool_async(call.name, call.arguments)
                obs_text = str(tool_res.output) if tool_res.success else f"Error: {tool_res.error}"
                context_notes.append(f"Step '{milestone}': Used {call.name} -> {obs_text}")
            else:
                context_notes.append(f"Step '{milestone}': {raw_out.strip()}")

            m_step = AgentStep(
                step_number=idx,
                thought=f"Executed milestone: {milestone}",
                action_name=action_name,
                action_input=action_input,
                observation=obs_text,
                is_final=False,
                execution_time_ms=round((time.perf_counter() - m_start) * 1000, 2),
            )
            agent_steps.append(m_step)

        # 3. Final Synthesis
        synth_start = time.perf_counter()
        synth_prompt = (
            f"Original Goal: '{prompt}'\n\n"
            f"Milestone Findings:\n" + "\n".join(context_notes) + "\n\n"
            "Synthesize a definitive, clear final response addressing the user's original request."
        )

        try:
            if hasattr(provider, "chat"):
                synth_res = await provider.chat(
                    messages=[{"role": "user", "content": synth_prompt}],
                    model=self.model_id,
                    temperature=self.temperature,
                )
                final_answer = synth_res["choices"][0]["message"]["content"] if isinstance(synth_res, dict) else str(synth_res)
            else:
                final_answer = "\n".join(context_notes)
        except Exception as ex:
            final_answer = f"Completed plan execution:\n" + "\n".join(context_notes)

        final_step = AgentStep(
            step_number=len(agent_steps) + 1,
            thought="Synthesized all milestone findings into final answer",
            is_final=True,
            execution_time_ms=round((time.perf_counter() - synth_start) * 1000, 2),
        )
        agent_steps.append(final_step)

        total_ms = (time.perf_counter() - t0) * 1000
        return AgentTrajectory(
            prompt=prompt,
            status=AgentStatus.COMPLETED,
            steps=agent_steps,
            final_answer=final_answer,
            total_duration_ms=round(total_ms, 2),
        )
