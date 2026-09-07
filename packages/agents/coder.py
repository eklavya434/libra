"""
Libra Agents Package - Code Generation & Test-Driven Auto-Debugging (Self-Correction)

Implements the iterative Code Generation and Self-Debugging loop:
1. Synthesizes initial Python code from natural language task specifications.
2. Executes the code and optional unit assertions inside SafePythonSandbox.
3. Captures runtime exceptions (SyntaxError, IndexError, ZeroDivisionError, AssertionError).
4. Employs reflective self-correction prompts to diagnose the root cause and repair the code.
5. Iterates until all assertions pass or max debug budget is exhausted.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from packages.agents.base import AgentStatus, AgentStep, AgentTrajectory, BaseAgent
from packages.agents.pal import extract_code
from packages.providers.router import ProviderRouter, get_router
from packages.tools.sandbox import SafePythonSandbox, SandboxSecurityError


class DebugIteration(BaseModel):
    """Encapsulates a single iteration of code execution, diagnostics, and repair."""

    iteration: int = Field(..., description="1-indexed attempt number")
    code: str = Field(..., description="Code executed in this iteration")
    success: bool = Field(..., description="Whether execution and assertions succeeded")
    stdout: str = Field("", description="Standard output captured from execution")
    result: Any | None = Field(None, description="Returned evaluation result if any")
    error: str | None = Field(None, description="Exception name and message if failed")
    traceback: str | None = Field(None, description="Formatted traceback with line numbers")
    reflection: str | None = Field(None, description="Agent diagnostic reflection on the bug")
    repair_explanation: str | None = Field(None, description="Agent explanation of the fix")


class CodeTrajectory(BaseModel):
    """Complete history of an iterative code synthesis and debugging session."""

    prompt: str = Field(..., description="Original user task prompt")
    assertions: str | None = Field(
        None, description="Optional unit test assertions verified against"
    )
    initial_code: str = Field("", description="First generated candidate code")
    final_code: str = Field("", description="Final working or best-effort code")
    iterations: list[DebugIteration] = Field(
        default_factory=list, description="Sequence of repair iterations"
    )
    success: bool = Field(False, description="Whether the final code passed all tests")
    total_attempts: int = Field(0, description="Total number of execution/repair attempts")
    final_output: Any | None = Field(None, description="Output from the successful run")
    error: str | None = Field(None, description="Final error if code could not be repaired")
    total_duration_ms: float = Field(0.0, description="Total wall-clock duration in milliseconds")


def format_code_with_lines(code: str) -> str:
    """Formats code with line numbers (e.g. '  1 | def foo():') for LLM diagnostic readability."""
    lines = code.splitlines()
    return "\n".join(f"{i + 1:3d} | {line}" for i, line in enumerate(lines))


def combine_code_and_assertions(code: str, assertions: str | None = None) -> str:
    """Combines user code with unit test assertions for sandboxed execution."""
    code = code.strip()
    if not assertions or not assertions.strip():
        return code
    return f"{code}\n\n# --- Libra Unit Assertions ---\n{assertions.strip()}"


class AutoDebugger:
    """
    Dedicated test-driven auto-debugging engine.
    Diagnoses runtime crashes, assertion failures, or syntax errors,
    and queries the model with precise line-level diagnostics to generate working repairs.
    """

    DEBUGGER_SYSTEM_PROMPT = (
        "You are an expert Python Auto-Debugger and software engineer.\n"
        "Your task is to fix broken Python code that produced a runtime crash or failed test assertions.\n\n"
        "Rules:\n"
        "1. Analyze the failing line number, error message, and traceback carefully.\n"
        "2. Identify the exact root cause (off-by-one, type mismatch, missing edge case, unhandled boundary, etc.).\n"
        "3. Provide a brief explanation of the bug and your fix.\n"
        "4. Output the COMPLETE, fully working corrected Python code inside a ```python ... ``` block.\n"
        "5. Do NOT use ellipsis (...) or omit any part of the implementation.\n"
        "6. Do NOT import os, sys, subprocess, or unapproved external packages."
    )

    def __init__(
        self,
        model_id: str = "mock-model",
        provider_name: str | None = None,
        router: ProviderRouter | None = None,
        sandbox: SafePythonSandbox | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model_id = model_id
        self.provider_name = provider_name
        self.router = router or get_router()
        self.sandbox = sandbox or SafePythonSandbox(default_timeout=5.0, max_memory_mb=128.0)
        self.temperature = temperature

    async def debug(
        self,
        task_description: str,
        initial_code: str,
        assertions: str | None = None,
        max_iterations: int = 3,
    ) -> CodeTrajectory:
        """
        Executes the iterative debug loop:
        Runs code -> detects error -> prompts LLM for repair -> repeats until success or max_iterations.
        """
        start_time = time.perf_counter()
        trajectory = CodeTrajectory(
            prompt=task_description,
            assertions=assertions,
            initial_code=initial_code,
            final_code=initial_code,
        )

        current_code = initial_code
        last_reflection: str | None = None
        last_explanation: str | None = None

        for attempt in range(1, max_iterations + 1):
            full_code_to_run = combine_code_and_assertions(current_code, assertions)

            # 1. Execute in sandbox
            try:
                exec_res = self.sandbox.execute(full_code_to_run)
            except (TimeoutError, SandboxSecurityError, Exception) as e:  # noqa: BLE001
                exec_res = {
                    "stdout": "",
                    "result": None,
                    "success": False,
                    "error": f"{type(e).__name__}: {e!s}",
                    "traceback": getattr(e, "__traceback__", None) and str(e),
                }

            is_success = exec_res.get("success", False)
            iteration_record = DebugIteration(
                iteration=attempt,
                code=current_code,
                success=is_success,
                stdout=exec_res.get("stdout") or "",
                result=exec_res.get("result"),
                error=exec_res.get("error"),
                traceback=exec_res.get("traceback"),
                reflection=last_reflection,
                repair_explanation=last_explanation,
            )
            trajectory.iterations.append(iteration_record)
            trajectory.final_code = current_code
            trajectory.total_attempts = attempt

            # 2. Check if working
            if is_success:
                trajectory.success = True
                trajectory.final_output = (
                    exec_res.get("result") or exec_res.get("stdout", "").strip()
                )
                trajectory.error = None
                break

            # If failed and reached budget, terminate
            if attempt >= max_iterations:
                trajectory.success = False
                trajectory.error = (
                    exec_res.get("error") or "Max debug iterations exceeded without resolution."
                )
                break

            # 3. Formulate reflective debugging prompt for next iteration
            error_details = exec_res.get("error") or "Unknown runtime error"
            tb_details = exec_res.get("traceback") or "No traceback available"
            numbered_code = format_code_with_lines(full_code_to_run)

            debug_prompt = (
                f"Task Description:\n{task_description}\n\n"
                f"Current Code (with line numbers):\n```python\n{numbered_code}\n```\n\n"
                f"Execution Failure Details:\n"
                f"Error: {error_details}\n"
                f"Traceback:\n{tb_details}\n\n"
            )

            if assertions:
                debug_prompt += (
                    f"Target Unit Assertions to Satisfy:\n```python\n{assertions}\n```\n\n"
                )

            debug_prompt += (
                "Please analyze what caused this error, explain your fix concisely, "
                "and provide the complete repaired Python code (without line numbers) in a ```python ... ``` block."
            )

            messages = [
                {"role": "system", "content": self.DEBUGGER_SYSTEM_PROMPT},
                {"role": "user", "content": debug_prompt},
            ]

            try:
                response = await self.router.chat(
                    messages=messages,
                    model_id=self.model_id,
                    provider_name=self.provider_name,
                    temperature=self.temperature,
                )
                raw_reply = response["choices"][0]["message"]["content"]
                repaired_code = extract_code(raw_reply)

                # Split explanation and code if possible
                last_reflection = (
                    raw_reply.split("```")[0].strip()
                    if "```" in raw_reply
                    else "Reflected on bug fix."
                )
                last_explanation = f"Repaired in iteration {attempt + 1}"

                if repaired_code.strip():
                    current_code = repaired_code
                else:
                    # Model failed to output code
                    trajectory.error = "Model did not provide executable code in repair step."
                    break

            except Exception as e:  # noqa: BLE001
                trajectory.error = f"Model debugging query failed: {e}"
                break

        trajectory.total_duration_ms = (time.perf_counter() - start_time) * 1000.0
        return trajectory


class CodeAgent(BaseAgent):
    """
    Autonomous Code Generation & Auto-Debugging Agent.
    Synthesizes code from task specifications, runs self-evaluating tests,
    and automatically debugs failures until completion.
    """

    CODER_SYSTEM_PROMPT = (
        "You are an expert Python software engineering agent.\n"
        "Your task is to write clean, correct, and self-contained Python code to solve the user's problem.\n\n"
        "Guidelines:\n"
        "1. Write clear, efficient Python with type annotations and docstrings.\n"
        "2. Ensure the code satisfies any requested function signatures and edge cases.\n"
        "3. You may use standard libraries: math, statistics, random, json, datetime, collections, itertools, re, string, decimal, fractions.\n"
        "4. Do NOT import os, sys, subprocess, or external dependencies.\n"
        "5. Enclose your complete Python script inside a ```python ... ``` block."
    )

    def __init__(
        self,
        model_id: str = "mock-model",
        provider_name: str | None = None,
        router: ProviderRouter | None = None,
        sandbox: SafePythonSandbox | None = None,
        max_steps: int = 5,
        max_debug_iterations: int = 3,
        timeout_sec: float = 60.0,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(max_steps=max_steps, timeout_sec=timeout_sec)
        self.model_id = model_id
        self.provider_name = provider_name
        self.router = router or get_router()
        self.sandbox = sandbox or SafePythonSandbox(default_timeout=5.0, max_memory_mb=128.0)
        self.max_debug_iterations = max_debug_iterations
        self.temperature = temperature
        self.debugger = AutoDebugger(
            model_id=model_id,
            provider_name=provider_name,
            router=self.router,
            sandbox=self.sandbox,
            temperature=temperature,
        )

    async def generate_and_test(
        self,
        prompt: str,
        assertions: str | None = None,
        max_iterations: int | None = None,
    ) -> CodeTrajectory:
        """
        Synthesizes initial code from specification, then runs the AutoDebugger loop.
        """
        budget = max_iterations or self.max_debug_iterations

        # 1. Generate initial code
        user_prompt = prompt
        if assertions:
            user_prompt += f"\n\nYour solution must pass these unit test assertions:\n```python\n{assertions}\n```"

        messages = [
            {"role": "system", "content": self.CODER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self.router.chat(
                messages=messages,
                model_id=self.model_id,
                provider_name=self.provider_name,
                temperature=self.temperature,
            )
            raw_code_reply = response["choices"][0]["message"]["content"]
            initial_code = extract_code(raw_code_reply)
        except Exception as e:  # noqa: BLE001
            return CodeTrajectory(
                prompt=prompt,
                assertions=assertions,
                initial_code="",
                final_code="",
                success=False,
                error=f"Initial code synthesis failed: {e}",
            )

        # 2. Run test-driven AutoDebugger
        return await self.debugger.debug(
            task_description=prompt,
            initial_code=initial_code,
            assertions=assertions,
            max_iterations=budget,
        )

    async def run(self, prompt: str) -> AgentTrajectory:
        """
        Executes the agent loop conforming to the BaseAgent abstraction.
        Records step-by-step synthesis, test runs, and reflections.
        """
        start_time = time.perf_counter()
        trajectory = AgentTrajectory(prompt=prompt, status=AgentStatus.PLANNING)

        code_traj = await self.generate_and_test(
            prompt=prompt, max_iterations=self.max_debug_iterations
        )

        # Convert CodeTrajectory iterations to AgentSteps
        for it in code_traj.iterations:
            thought = it.reflection or f"Attempt {it.iteration}: Evaluating code against sandbox."
            action_input = {"code": it.code}
            obs = f"Success={it.success} | stdout={it.stdout} | error={it.error}"
            step = AgentStep(
                step_number=it.iteration,
                thought=thought,
                action_name="python_interpreter",
                action_input=action_input,
                observation=obs,
                is_final=it.success or it.iteration >= code_traj.total_attempts,
                execution_time_ms=0.0,
            )
            trajectory.steps.append(step)

        if code_traj.success:
            trajectory.status = AgentStatus.COMPLETED
            trajectory.final_answer = (
                f"Successfully synthesized and verified working Python solution after {code_traj.total_attempts} attempt(s):\n\n"
                f"```python\n{code_traj.final_code}\n```\n\n"
                f"Output: {code_traj.final_output}"
            )
        else:
            trajectory.status = AgentStatus.FAILED
            trajectory.error = code_traj.error or "Code failed to pass all execution checks."
            trajectory.final_answer = (
                f"Best-effort code reached after {code_traj.total_attempts} attempt(s):\n\n"
                f"```python\n{code_traj.final_code}\n```\n\n"
                f"Remaining error: {code_traj.error}"
            )

        trajectory.total_duration_ms = (time.perf_counter() - start_time) * 1000.0
        return trajectory
