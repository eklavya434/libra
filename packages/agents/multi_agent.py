"""
Libra Agents Package - Multi-Agent Collaboration & Orchestration

Implements role-specialized collaborative ensembles:
- Architect: Problem decomposition and functional specification
- Coder: Implementation and revision
- Reviewer: Adversarial code review and quality audit
- Tester: Unit test synthesis and sandbox runtime execution
- Coordinator: Consensus management and iteration control
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from packages.agents.base import AgentStatus, AgentTrajectory, BaseAgent
from packages.agents.coder import combine_code_and_assertions, format_code_with_lines
from packages.agents.message_bus import (
    AgentMessage,
    AgentRole,
    MessageType,
    SharedBlackboard,
)
from packages.agents.pal import extract_code
from packages.providers.router import ProviderRouter, get_router
from packages.tools.sandbox import SafePythonSandbox


class TeamRound(BaseModel):
    """Snapshot of a single collaborative iteration across team members."""

    round_number: int = Field(..., description="1-indexed iteration round")
    candidate_code: str = Field("", description="Code proposed by Coder in this round")
    reviewer_approved: bool = Field(False, description="Whether Reviewer approved the code")
    review_notes: str = Field("", description="Critique from the Reviewer")
    tests_passed: bool = Field(False, description="Whether sandbox test assertions passed")
    test_output: str = Field("", description="Sandbox stdout or error trace")
    messages: list[AgentMessage] = Field(
        default_factory=list, description="Messages emitted in this round"
    )


class TeamTrajectory(BaseModel):
    """Complete execution history and outcome of a multi-agent team session."""

    task: str = Field(..., description="Original user prompt")
    status: AgentStatus = Field(AgentStatus.PLANNING, description="Outcome status")
    consensus_reached: bool = Field(
        False, description="Whether all validation criteria were satisfied"
    )
    total_rounds: int = Field(0, description="Total collaboration rounds executed")
    rounds: list[TeamRound] = Field(default_factory=list, description="Per-round snapshots")
    final_code: str = Field("", description="Final approved or best-effort Python code")
    final_answer: str = Field("", description="Synthesized team answer delivered to user")
    messages: list[AgentMessage] = Field(default_factory=list, description="All messages exchanged")
    total_duration_ms: float = Field(0.0, description="Total wall-clock duration in milliseconds")


class SpecializedAgent:
    """An autonomous agent with a dedicated persona, system instructions, and role."""

    def __init__(
        self,
        name: str,
        role: AgentRole,
        system_prompt: str,
        model_id: str = "mock-model",
        provider_name: str | None = None,
        router: ProviderRouter | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model_id = model_id
        self.provider_name = provider_name
        self.router = router or get_router()
        self.temperature = temperature

    async def generate_reply(self, prompt: str) -> str:
        """Invokes the model provider with this agent's system prompt."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = await self.router.chat(
            messages=messages,
            model_id=self.model_id,
            provider_name=self.provider_name,
            temperature=self.temperature,
        )
        return response["choices"][0]["message"]["content"]


class CollaborativeTeam(BaseAgent):
    """
    Orchestrates a collaborative ensemble of specialized agents:
    Architect -> Coder -> Reviewer -> Tester -> Coordinator.
    """

    ARCHITECT_PROMPT = (
        "You are the Lead Software Architect in a collaborative engineering team.\n"
        "Your role: Decompose user tasks into a clear, concise technical specification.\n"
        "Specify:\n"
        "1. Required function/class names and signatures with type hints.\n"
        "2. Input parameters and expected return values.\n"
        "3. Critical boundary conditions and edge cases to handle.\n"
        "Be concise, technical, and precise."
    )

    CODER_PROMPT = (
        "You are the Senior Implementation Engineer in a collaborative team.\n"
        "Your role: Write clean, self-contained, and bug-free Python code conforming to the specification.\n"
        "If the Reviewer or Tester has provided critique/failures from a previous round, address every issue.\n"
        "Rules:\n"
        "1. Output your complete solution in a ```python ... ``` block.\n"
        "2. Only use standard libraries (math, datetime, time, statistics, collections, itertools, json, re).\n"
        "3. Do NOT import os, sys, subprocess, or external dependencies."
    )

    REVIEWER_PROMPT = (
        "You are the Adversarial Senior Code Reviewer in a collaborative team.\n"
        "Your role: Critically audit the candidate code against the Architect's specification.\n"
        "Check for:\n"
        "1. Functional correctness and logic bugs.\n"
        "2. Off-by-one errors and unhandled boundary cases.\n"
        "3. Time/space complexity efficiency.\n"
        "Conclude your review with EXACTLY ONE of the following verdict lines:\n"
        "VERDICT: APPROVED\n"
        "or\n"
        "VERDICT: REVISION_REQUESTED\n"
        "If requesting revisions, provide specific bulleted instructions for the Coder."
    )

    TESTER_PROMPT = (
        "You are the Automated QA and Testing Specialist in a collaborative team.\n"
        "Your role: Write comprehensive Python `assert` test statements to verify the candidate code.\n"
        "Rules:\n"
        "1. Write 3-5 unit test assertions testing normal cases, edge cases, and boundaries.\n"
        "2. Format your assertions in a ```python ... ``` block (e.g. `assert func(2) == 4`).\n"
        "3. Assertions will be executed alongside candidate code in a sandbox."
    )

    def __init__(
        self,
        model_id: str = "mock-model",
        provider_name: str | None = None,
        router: ProviderRouter | None = None,
        sandbox: SafePythonSandbox | None = None,
        max_rounds: int = 3,
        timeout_sec: float = 120.0,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(max_steps=max_rounds * 4, timeout_sec=timeout_sec)
        self.model_id = model_id
        self.provider_name = provider_name
        self.router = router or get_router()
        self.sandbox = sandbox or SafePythonSandbox(default_timeout=5.0, max_memory_mb=128.0)
        self.max_rounds = max_rounds
        self.temperature = temperature

        # Instantiate specialized team members
        self.architect = SpecializedAgent(
            name="Architect",
            role=AgentRole.ARCHITECT,
            system_prompt=self.ARCHITECT_PROMPT,
            model_id=model_id,
            provider_name=provider_name,
            router=self.router,
            temperature=temperature,
        )
        self.coder = SpecializedAgent(
            name="Coder",
            role=AgentRole.CODER,
            system_prompt=self.CODER_PROMPT,
            model_id=model_id,
            provider_name=provider_name,
            router=self.router,
            temperature=temperature,
        )
        self.reviewer = SpecializedAgent(
            name="Reviewer",
            role=AgentRole.REVIEWER,
            system_prompt=self.REVIEWER_PROMPT,
            model_id=model_id,
            provider_name=provider_name,
            router=self.router,
            temperature=temperature,
        )
        self.tester = SpecializedAgent(
            name="Tester",
            role=AgentRole.TESTER,
            system_prompt=self.TESTER_PROMPT,
            model_id=model_id,
            provider_name=provider_name,
            router=self.router,
            temperature=temperature,
        )

    async def collaborate(self, prompt: str) -> TeamTrajectory:
        """Executes the complete multi-agent collaboration loop to completion."""
        trajectory: TeamTrajectory | None = None
        async for event in self.stream_collaboration(prompt):
            if event.get("type") == "finish":
                trajectory = event.get("trajectory")

        if trajectory is not None:
            return trajectory

        return TeamTrajectory(
            task=prompt,
            status=AgentStatus.FAILED,
            final_answer="Collaboration terminated unexpectedly without trajectory.",
        )

    async def stream_collaboration(self, prompt: str) -> AsyncIterator[dict[str, Any]]:
        """
        Asynchronously executes the multi-agent collaboration loop,
        streaming real-time message events and round milestones.
        """
        start_time = time.perf_counter()
        blackboard = SharedBlackboard(task=prompt)
        trajectory = TeamTrajectory(task=prompt, status=AgentStatus.PLANNING)

        yield {"type": "start", "task": prompt}

        # ---------------------------------------------------------------------
        # Stage 1: Initial Architecture & Specification
        # ---------------------------------------------------------------------
        arch_prompt = (
            f"User Task Requirement:\n{prompt}\n\nPlease formulate the technical specification."
        )
        arch_reply = await self.architect.generate_reply(arch_prompt)
        blackboard.specification = arch_reply

        arch_msg = AgentMessage(
            sender_name=self.architect.name,
            sender_role=self.architect.role,
            recipient="broadcast",
            message_type=MessageType.SPECIFICATION,
            content=arch_reply,
        )
        blackboard.post_message(arch_msg)
        trajectory.messages.append(arch_msg)
        yield {"type": "message", "message": arch_msg.model_dump()}

        # ---------------------------------------------------------------------
        # Stages 2-5: Iterative Collaborative Rounds (Coder -> Reviewer -> Tester)
        # ---------------------------------------------------------------------
        for round_idx in range(1, self.max_rounds + 1):
            round_record = TeamRound(round_number=round_idx)
            yield {"type": "round_start", "round": round_idx}

            # 1. Coder Step
            if round_idx == 1:
                coder_prompt = (
                    f"Task: {prompt}\n\n"
                    f"Architect Specification:\n{blackboard.specification}\n\n"
                    "Write the complete Python implementation."
                )
            else:
                numbered_code = format_code_with_lines(blackboard.candidate_code)
                coder_prompt = (
                    f"Task: {prompt}\n\n"
                    f"Previous Candidate Code:\n```python\n{numbered_code}\n```\n\n"
                    f"Reviewer Critique:\n{blackboard.review_notes}\n\n"
                    f"Sandbox Test Failures:\n{blackboard.test_output}\n\n"
                    "Please revise the implementation to address all feedback and pass tests."
                )

            coder_reply = await self.coder.generate_reply(coder_prompt)
            candidate_code = extract_code(coder_reply)
            blackboard.candidate_code = candidate_code
            round_record.candidate_code = candidate_code

            coder_msg = AgentMessage(
                sender_name=self.coder.name,
                sender_role=self.coder.role,
                recipient="broadcast",
                message_type=MessageType.CODE_PROPOSAL,
                content=coder_reply,
                metadata={"code": candidate_code},
            )
            blackboard.post_message(coder_msg)
            trajectory.messages.append(coder_msg)
            round_record.messages.append(coder_msg)
            yield {"type": "message", "message": coder_msg.model_dump()}

            # 2. Reviewer Step (Adversarial Audit)
            reviewer_prompt = (
                f"Specification:\n{blackboard.specification}\n\n"
                f"Candidate Implementation:\n```python\n{blackboard.candidate_code}\n```\n\n"
                "Audit the code and conclude with VERDICT: APPROVED or VERDICT: REVISION_REQUESTED."
            )
            reviewer_reply = await self.reviewer.generate_reply(reviewer_prompt)
            blackboard.review_notes = reviewer_reply

            is_approved = "VERDICT: APPROVED" in reviewer_reply or (
                "APPROVED" in reviewer_reply.upper()
                and "REVISION_REQUESTED" not in reviewer_reply.upper()
            )
            blackboard.is_approved_by_reviewer = is_approved
            round_record.reviewer_approved = is_approved
            round_record.review_notes = reviewer_reply

            rev_msg = AgentMessage(
                sender_name=self.reviewer.name,
                sender_role=self.reviewer.role,
                recipient="broadcast",
                message_type=MessageType.CONSENSUS_APPROVAL
                if is_approved
                else MessageType.CRITIQUE,
                content=reviewer_reply,
                metadata={"approved": is_approved},
            )
            blackboard.post_message(rev_msg)
            trajectory.messages.append(rev_msg)
            round_record.messages.append(rev_msg)
            yield {"type": "message", "message": rev_msg.model_dump()}

            # 3. Tester Step (Sandbox Verification)
            if round_idx == 1 or not blackboard.test_assertions:
                tester_prompt = (
                    f"Specification:\n{blackboard.specification}\n\n"
                    f"Candidate Code:\n```python\n{blackboard.candidate_code}\n```\n\n"
                    "Formulate 3-5 unit test assertions verifying this code."
                )
                tester_reply = await self.tester.generate_reply(tester_prompt)
                test_assertions = extract_code(tester_reply)
                blackboard.test_assertions = test_assertions
            else:
                test_assertions = blackboard.test_assertions
                tester_reply = f"Re-evaluating assertions:\n```python\n{test_assertions}\n```"

            # Execute in SafePythonSandbox
            test_bundle = combine_code_and_assertions(blackboard.candidate_code, test_assertions)
            try:
                exec_res = self.sandbox.execute(test_bundle)
                tests_passed = exec_res.get("success", False)
                test_out = exec_res.get("stdout") or exec_res.get("error") or "Execution complete."
            except Exception as e:  # noqa: BLE001
                tests_passed = False
                test_out = f"Sandbox Exception: {e}"

            blackboard.tests_passed = tests_passed
            blackboard.test_output = test_out
            round_record.tests_passed = tests_passed
            round_record.test_output = test_out

            tester_msg = AgentMessage(
                sender_name=self.tester.name,
                sender_role=self.tester.role,
                recipient="broadcast",
                message_type=MessageType.TEST_RESULTS,
                content=f"{tester_reply}\n\nSandbox Test Result: {'PASSED' if tests_passed else 'FAILED'}\nOutput: {test_out}",
                metadata={"tests_passed": tests_passed, "output": test_out},
            )
            blackboard.post_message(tester_msg)
            trajectory.messages.append(tester_msg)
            round_record.messages.append(tester_msg)
            yield {"type": "message", "message": tester_msg.model_dump()}

            trajectory.rounds.append(round_record)
            trajectory.total_rounds = round_idx
            trajectory.final_code = blackboard.candidate_code

            # 4. Check Consensus
            if is_approved and tests_passed:
                trajectory.consensus_reached = True
                trajectory.status = AgentStatus.COMPLETED
                final_answer = (
                    f"### Collaborative Multi-Agent Solution (Approved in Round {round_idx})\n\n"
                    f"**Architect Specification**: Satisfied\n"
                    f"**Reviewer Audit**: APPROVED\n"
                    f"**Sandbox Tests**: PASSED\n\n"
                    f"```python\n{blackboard.candidate_code}\n```"
                )
                trajectory.final_answer = final_answer
                break

            yield {"type": "round_end", "round": round_idx, "consensus": False}

        # If loop finishes without consensus
        if not trajectory.consensus_reached:
            trajectory.status = AgentStatus.FAILED
            trajectory.final_answer = (
                f"### Best-Effort Multi-Agent Solution (Max rounds {self.max_rounds} reached)\n\n"
                f"**Reviewer Status**: {'APPROVED' if blackboard.is_approved_by_reviewer else 'REVISION_REQUESTED'}\n"
                f"**Sandbox Tests**: {'PASSED' if blackboard.tests_passed else 'FAILED'}\n\n"
                f"```python\n{blackboard.candidate_code}\n```\n\n"
                f"Remaining Feedback: {blackboard.review_notes}\n"
                f"Test Output: {blackboard.test_output}"
            )

        trajectory.total_duration_ms = (time.perf_counter() - start_time) * 1000.0
        yield {"type": "finish", "trajectory": trajectory}

    async def run(self, prompt: str) -> AgentTrajectory:
        """Conforms to BaseAgent interface, mapping TeamTrajectory to AgentTrajectory."""
        team_traj = await self.collaborate(prompt)
        return AgentTrajectory(
            prompt=prompt,
            status=team_traj.status,
            final_answer=team_traj.final_answer,
            total_duration_ms=team_traj.total_duration_ms,
        )
