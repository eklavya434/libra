"""
Libra Agents Package - Program-Aided Language Models (PAL)

Implements the PAL paradigm (Gao et al., 2022):
Offloads algorithmic, arithmetic, and symbolic reasoning from LLM token prediction
to Python code executed inside the process-isolated SafePythonSandbox.
"""

from __future__ import annotations

import ast
import re
import time

from packages.agents.base import AgentStatus, AgentStep, AgentTrajectory, BaseAgent
from packages.providers.router import ProviderRouter, get_router
from packages.tools.sandbox import SafePythonSandbox, SandboxSecurityError


def extract_code(text: str) -> str:
    """
    Extracts executable Python code from model responses.
    Handles ```python ... ``` fences, generic ``` ... ``` fences,
    or raw Python text.
    """
    # 1. Look for ```python ... ``` code block
    py_match = re.search(r"```(?:python|py)\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if py_match:
        return py_match.group(1).strip()

    # 2. Look for any generic ``` ... ``` block
    generic_match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if generic_match:
        return generic_match.group(1).strip()

    # 3. Fallback: If text itself looks like code lines, filter prose
    lines = text.splitlines()
    code_lines: list[str] = []
    in_code = False

    for line in lines:
        stripped = line.strip()
        if (
            stripped.startswith(
                ("def ", "import ", "from ", "class ", "return ", "print(", "for ", "while ", "if ")
            )
            or "=" in stripped
        ):
            in_code = True
            code_lines.append(line)
        elif (
            in_code
            and (
                stripped.startswith(("#", "else:", "elif:", "except", "finally:", "try:", "with "))
                or line.startswith(("    ", "\t"))
            )
            or in_code
            and stripped == ""
        ):
            code_lines.append(line)
        elif in_code and not line.startswith(("    ", "\t")):
            # Check if this line is still code
            try:
                ast.parse(stripped)
                code_lines.append(line)
            except SyntaxError:
                # Likely returned to natural language prose
                pass

    if code_lines:
        return "\n".join(code_lines).strip()

    return text.strip()


def normalize_pal_code(code: str) -> str:
    """
    Ensures that if the code defines `solution()` but never calls it,
    a call to `solution()` is placed at the end so runner.py evaluates it.
    """
    code = code.strip()
    if not code:
        return code

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    # Check if a function named 'solution' is defined
    has_solution_def = any(
        isinstance(node, ast.FunctionDef) and node.name == "solution" for node in tree.body
    )

    if has_solution_def:
        # Check if solution is already called at the top level
        calls_solution = any(
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "solution"
            for node in tree.body
        )
        if not calls_solution:
            code = f"{code}\n\nsolution()"

    return code


class PALAgent(BaseAgent):
    """
    Program-Aided Language Model Agent.
    Converts arithmetic, combinatorics, date/time, and symbolic word problems
    into executable Python scripts and executes them inside SafePythonSandbox.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert Program-Aided Language (PAL) assistant.\n"
        "To solve math, algorithmic, combinatorics, or logic problems accurately, do NOT calculate in your head.\n"
        "Instead, write clean, modular, and self-contained Python code that calculates the exact answer.\n\n"
        "Guidelines:\n"
        "1. Write a function `solution()` that returns the final result, or print the result.\n"
        "2. You can use standard libraries: math, statistics, random, json, datetime, collections, itertools, re, string, decimal, fractions.\n"
        "3. Do NOT import os, sys, subprocess, or external packages.\n"
        "4. Enclose your Python script inside ```python ... ``` markdown blocks.\n\n"
        "Example:\n"
        "Question: If a store has 45 apples and sells 3/5 of them, then receives 28 more, how many apples are there?\n"
        "```python\n"
        "def solution():\n"
        "    initial = 45\n"
        "    sold = initial * (3 / 5)\n"
        "    remaining = initial - sold\n"
        "    new_shipment = 28\n"
        "    return int(remaining + new_shipment)\n"
        "```"
    )

    def __init__(
        self,
        model_id: str = "mock-model",
        provider_name: str | None = None,
        router: ProviderRouter | None = None,
        sandbox: SafePythonSandbox | None = None,
        max_steps: int = 5,
        timeout_sec: float = 30.0,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(max_steps=max_steps, timeout_sec=timeout_sec)
        self.model_id = model_id
        self.provider_name = provider_name
        self.router = router or get_router()
        self.sandbox = sandbox or SafePythonSandbox(default_timeout=5.0, max_memory_mb=128.0)
        self.temperature = temperature

    async def run(self, prompt: str) -> AgentTrajectory:
        """
        Executes the PAL pipeline:
        1. Formulates prompt requesting executable Python program.
        2. Queries the model provider.
        3. Extracts and normalizes the code block.
        4. Executes code inside SafePythonSandbox.
        5. Formulates deterministic final answer from execution results.
        """
        start_time = time.perf_counter()
        trajectory = AgentTrajectory(prompt=prompt, status=AgentStatus.PLANNING)

        # 1. Query LLM to generate PAL code
        messages = [
            {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.router.chat(
                messages=messages,
                model_id=self.model_id,
                provider_name=self.provider_name,
                temperature=self.temperature,
            )
            raw_reply = response["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            trajectory.status = AgentStatus.FAILED
            trajectory.error = f"Model generation failed: {e}"
            trajectory.total_duration_ms = (time.perf_counter() - start_time) * 1000.0
            return trajectory

        # 2. Extract code
        code = extract_code(raw_reply)
        normalized_code = normalize_pal_code(code)

        step1 = AgentStep(
            step_number=1,
            thought=raw_reply,
            action_name="python_interpreter",
            action_input={"code": normalized_code},
            execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
        )
        trajectory.steps.append(step1)

        if not normalized_code.strip():
            trajectory.status = AgentStatus.FAILED
            trajectory.error = (
                "No executable Python code could be extracted from the model response."
            )
            trajectory.total_duration_ms = (time.perf_counter() - start_time) * 1000.0
            return trajectory

        # 3. Execute in SafePythonSandbox
        t_exec_start = time.perf_counter()
        try:
            exec_result = self.sandbox.execute(
                normalized_code, timeout_sec=min(self.timeout_sec, 10.0)
            )
            exec_time_ms = (time.perf_counter() - t_exec_start) * 1000.0

            if exec_result["success"]:
                # Result can be in result or stdout
                val = exec_result.get("result")
                stdout_text = (exec_result.get("stdout") or "").strip()
                answer_val = val if val is not None else stdout_text

                step2 = AgentStep(
                    step_number=2,
                    thought="Code executed successfully in sandbox.",
                    observation=f"Result: {answer_val}",
                    is_final=True,
                    execution_time_ms=exec_time_ms,
                )
                trajectory.steps.append(step2)
                trajectory.final_answer = str(answer_val)
                trajectory.status = AgentStatus.COMPLETED
            else:
                error_msg = exec_result.get("error") or "Unknown runtime error"
                step2 = AgentStep(
                    step_number=2,
                    thought="Execution failed in sandbox.",
                    observation=f"Error: {error_msg}",
                    is_final=True,
                    execution_time_ms=exec_time_ms,
                )
                trajectory.steps.append(step2)
                trajectory.status = AgentStatus.FAILED
                trajectory.error = error_msg

        except (TimeoutError, SandboxSecurityError, Exception) as e:  # noqa: BLE001
            exec_time_ms = (time.perf_counter() - t_exec_start) * 1000.0
            step2 = AgentStep(
                step_number=2,
                thought="Execution exception raised.",
                observation=f"Exception: {e}",
                is_final=True,
                execution_time_ms=exec_time_ms,
            )
            trajectory.steps.append(step2)
            trajectory.status = AgentStatus.FAILED
            trajectory.error = str(e)

        trajectory.total_duration_ms = (time.perf_counter() - start_time) * 1000.0
        return trajectory
