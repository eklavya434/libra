"""
Libra Agents Package - ReAct (Reasoning + Acting) Agent

Implements the classic ReAct agent loop:
Interleaves Thought -> Action -> Observation cycles until task completion or budget exhaustion.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any, Optional

from packages.agents.base import AgentStatus, AgentStep, AgentTrajectory, BaseAgent
from packages.providers.base import BaseProvider
from packages.providers.router import ProviderRouter, get_router
from packages.tools.parser import ToolCallParser
from packages.tools.registry import ToolRegistry, get_tool_registry


class ReActAgent(BaseAgent):
    """
    Autonomous agent using the ReAct (Reasoning + Acting) loop.
    Iteratively plans, invokes tools, analyzes observations, and delivers final answers.
    """

    def __init__(
        self,
        model_id: str = "mock-model",
        provider_name: Optional[str] = None,
        router: Optional[ProviderRouter] = None,
        registry: Optional[ToolRegistry] = None,
        max_steps: int = 10,
        timeout_sec: float = 60.0,
        max_tool_failures: int = 3,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(
            max_steps=max_steps, timeout_sec=timeout_sec, max_tool_failures=max_tool_failures
        )
        self.model_id = model_id
        self.provider_name = provider_name
        self.router = router or get_router()
        self.registry = registry or get_tool_registry()
        self.temperature = temperature

    def _build_system_prompt(self) -> str:
        """Constructs the ReAct system prompt with available tool catalog."""
        tools_desc = []
        for tool in self.registry.list_tools():
            schema = tool.to_openai_schema()["function"]
            params = schema.get("parameters", {}).get("properties", {})
            param_str = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in params.items())
            tools_desc.append(f"- {tool.name}({param_str}): {tool.description}")

        tools_block = "\n".join(tools_desc)

        return (
            "You are an expert autonomous assistant capable of multi-step problem solving.\n"
            "You have access to the following tools:\n\n"
            f"{tools_block}\n\n"
            "To solve the user request, interleave your reasoning using this EXACT format:\n\n"
            "Thought: <Your reasoning about what to do next>\n"
            "Action:\n"
            "<tool_call>\n"
            '{"name": "<tool_name>", "arguments": {<json_arguments>}}\n'
            "</tool_call>\n\n"
            "When you have observed enough information to definitively answer the user question, respond with:\n"
            "Thought: I now have the complete answer.\n"
            "Final Answer: <Your comprehensive answer to the user>\n\n"
            "IMPORTANT: Always think first before taking any action. Never guess data when tools are available."
        )

    async def run(self, prompt: str) -> AgentTrajectory:
        """Executes the ReAct loop to completion and returns the full trajectory."""
        trajectory: Optional[AgentTrajectory] = None
        async for event in self.stream_steps(prompt):
            if event.get("type") == "finish":
                trajectory = event.get("trajectory")

        if trajectory is not None:
            return trajectory

        return AgentTrajectory(
            prompt=prompt,
            status=AgentStatus.FAILED,
            error="Agent execution terminated without returning a trajectory",
        )

    async def stream_steps(self, prompt: str) -> AsyncIterator[dict[str, Any]]:
        """
        Asynchronously executes the ReAct loop, yielding step-by-step events for streaming interfaces.
        """
        t0 = time.perf_counter()
        provider: BaseProvider = await self.router.resolve_provider_for_model(
            model_id=self.model_id,
            requested_provider=self.provider_name,
        )

        system_prompt = self._build_system_prompt()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        steps: list[AgentStep] = []
        consecutive_tool_failures = 0

        yield {"type": "start", "prompt": prompt}

        for step_idx in range(1, self.max_steps + 1):
            # 1. Check wall-clock budget
            elapsed = time.perf_counter() - t0
            if elapsed > self.timeout_sec:
                traj = AgentTrajectory(
                    prompt=prompt,
                    status=AgentStatus.TIMEOUT,
                    steps=steps,
                    error=f"Agent exceeded execution timeout of {self.timeout_sec:.1f}s",
                    total_duration_ms=round(elapsed * 1000, 2),
                )
                yield {"type": "timeout", "trajectory": traj}
                yield {"type": "finish", "trajectory": traj}
                return

            step_start = time.perf_counter()
            yield {"type": "step_start", "step": step_idx}

            # 2. Query model for next Thought / Action
            try:
                if hasattr(provider, "chat"):
                    res = await provider.chat(
                        messages=messages,
                        model=self.model_id,
                        temperature=self.temperature,
                    )
                    raw_output = (
                        res["choices"][0]["message"]["content"]
                        if isinstance(res, dict)
                        else str(res)
                    )
                elif hasattr(provider, "complete"):
                    res = await provider.complete(
                        messages=messages,
                        model_id=self.model_id,
                        temperature=self.temperature,
                    )
                    raw_output = res.content if hasattr(res, "content") else str(res)
                else:
                    raw_output = "Final Answer: Unable to query model provider."
            except Exception as e:
                traj = AgentTrajectory(
                    prompt=prompt,
                    status=AgentStatus.FAILED,
                    steps=steps,
                    error=f"Model generation error at step {step_idx}: {e}",
                    total_duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                )
                yield {"type": "error", "error": str(e), "trajectory": traj}
                yield {"type": "finish", "trajectory": traj}
                return

            # 3. Check for Final Answer in output
            if "Final Answer:" in raw_output:
                parts = raw_output.split("Final Answer:", 1)
                thought = parts[0].replace("Thought:", "").strip()
                final_answer = parts[1].strip()

                step = AgentStep(
                    step_number=step_idx,
                    thought=thought,
                    is_final=True,
                    execution_time_ms=round((time.perf_counter() - step_start) * 1000, 2),
                )
                steps.append(step)

                yield {"type": "thought", "step": step_idx, "content": thought}
                yield {"type": "final_answer", "answer": final_answer}

                traj = AgentTrajectory(
                    prompt=prompt,
                    status=AgentStatus.COMPLETED,
                    steps=steps,
                    final_answer=final_answer,
                    total_duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                )
                yield {"type": "finish", "trajectory": traj}
                return

            # 4. Parse Tool Calls from output
            tool_calls = ToolCallParser.parse(raw_output)

            if tool_calls:
                call = tool_calls[0]
                thought = (
                    ToolCallParser.strip_tool_calls(raw_output)
                    .replace("Thought:", "")
                    .replace("Action:", "")
                    .strip()
                )

                yield {"type": "thought", "step": step_idx, "content": thought}
                yield {
                    "type": "action",
                    "step": step_idx,
                    "tool": call.name,
                    "arguments": call.arguments,
                }

                # Execute tool via Registry
                tool_res = await self.registry.execute_tool_async(call.name, call.arguments)
                obs_text = str(tool_res.output) if tool_res.success else f"Error: {tool_res.error}"

                # Circuit breaker tracking
                if not tool_res.success:
                    consecutive_tool_failures += 1
                else:
                    consecutive_tool_failures = 0

                step = AgentStep(
                    step_number=step_idx,
                    thought=thought,
                    action_name=call.name,
                    action_input=call.arguments,
                    observation=obs_text,
                    is_final=False,
                    execution_time_ms=round((time.perf_counter() - step_start) * 1000, 2),
                )
                steps.append(step)
                yield {"type": "observation", "step": step_idx, "observation": obs_text}

                # Circuit breaker trip
                if consecutive_tool_failures >= self.max_tool_failures:
                    traj = AgentTrajectory(
                        prompt=prompt,
                        status=AgentStatus.FAILED,
                        steps=steps,
                        error=f"Circuit breaker tripped: tool failed {self.max_tool_failures} consecutive times",
                        total_duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                    )
                    yield {"type": "error", "error": traj.error, "trajectory": traj}
                    yield {"type": "finish", "trajectory": traj}
                    return

                # Append history for next cycle
                messages.append({"role": "assistant", "content": raw_output})
                messages.append({"role": "user", "content": f"Observation: {obs_text}"})

            else:
                # Direct conversational response without tool call or formal 'Final Answer:' marker
                thought = raw_output.replace("Thought:", "").strip()
                step = AgentStep(
                    step_number=step_idx,
                    thought=thought,
                    is_final=True,
                    execution_time_ms=round((time.perf_counter() - step_start) * 1000, 2),
                )
                steps.append(step)

                yield {"type": "final_answer", "answer": thought}

                traj = AgentTrajectory(
                    prompt=prompt,
                    status=AgentStatus.COMPLETED,
                    steps=steps,
                    final_answer=thought,
                    total_duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                )
                yield {"type": "finish", "trajectory": traj}
                return

        # Exceeded max_steps budget
        traj = AgentTrajectory(
            prompt=prompt,
            status=AgentStatus.FAILED,
            steps=steps,
            error=f"Agent exceeded maximum step limit of {self.max_steps} without reaching final answer",
            total_duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        yield {"type": "max_steps_exceeded", "trajectory": traj}
        yield {"type": "finish", "trajectory": traj}
