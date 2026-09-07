"""
Phase 20 Interactive Demo: Multi-Agent Collaboration & Orchestration

Demonstrates:
1. Role-specialized agents (Architect, Coder, Reviewer, Tester) collaborating on a task.
2. Inter-agent communication via SharedBlackboard.
3. Adversarial code review catching a boundary bug in Round 1.
4. Coder revising the implementation in Round 2.
5. Sandbox unit test verification and team consensus convergence.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from packages.agents import (
    CollaborativeTeam,
)
from packages.providers.base import BaseProvider, ModelMetadata
from packages.providers.router import ProviderRouter
from packages.tools.sandbox import SafePythonSandbox


class DemoTeamProvider(BaseProvider):
    """Provides realistic scripted responses for the multi-agent team demonstration."""

    def __init__(self) -> None:
        self.coder_round = 0
        self.reviewer_round = 0

    @property
    def name(self) -> str:
        return "demo-team-provider"

    async def list_models(self) -> list[ModelMetadata]:
        return []

    async def chat(self, messages, model=None, **kwargs):
        sys_prompt = messages[0]["content"] if messages else ""

        if "Adversarial Senior Code Reviewer" in sys_prompt:
            self.reviewer_round += 1
            if self.reviewer_round == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "### Adversarial Code Review (Round 1):\n"
                                    "1. Functionality: Computes rolling average, but fails when the window size k > len(values).\n"
                                    "2. Edge Case: If values is empty, returns an empty list, but raises ZeroDivisionError if k <= 0.\n\n"
                                    "VERDICT: REVISION_REQUESTED\n"
                                    "- Guard against k <= 0 by raising ValueError.\n"
                                    "- If k > len(values), return an empty list."
                                )
                            }
                        }
                    ]
                }
            else:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "### Adversarial Code Review (Round 2):\n"
                                    "1. Checked ValueError validation for k <= 0: Passed.\n"
                                    "2. Checked boundary handling for k > len(values): Passed.\n"
                                    "3. Sliding window calculation logic is clean and O(N) efficient.\n\n"
                                    "VERDICT: APPROVED"
                                )
                            }
                        }
                    ]
                }

        elif "Lead Software Architect" in sys_prompt:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "### Technical Specification:\n"
                                "- Function: `def rolling_average(values: list[float], k: int) -> list[float]`\n"
                                "- Parameters:\n"
                                "  - `values`: Sequence of numerical values.\n"
                                "  - `k`: Size of sliding average window.\n"
                                "- Return: List of rolling averages of length `max(0, len(values) - k + 1)`.\n"
                                "- Edge Cases: k <= 0 must raise ValueError; k > len(values) returns []."
                            )
                        }
                    }
                ]
            }

        elif "Senior Implementation Engineer" in sys_prompt:
            self.coder_round += 1
            if self.coder_round == 1:
                # Round 1 code (missing k <= 0 check)
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "Here is the initial implementation:\n\n"
                                    "```python\n"
                                    "def rolling_average(values: list[float], k: int) -> list[float]:\n"
                                    "    if k > len(values):\n"
                                    "        return []\n"
                                    "    result = []\n"
                                    "    for i in range(len(values) - k + 1):\n"
                                    "        window = values[i : i + k]\n"
                                    "        result.append(sum(window) / k)\n"
                                    "    return result\n"
                                    "```"
                                )
                            }
                        }
                    ]
                }
            else:
                # Round 2 code (fixed with ValueError check)
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "Revised implementation addressing the Reviewer's critique:\n\n"
                                    "```python\n"
                                    "def rolling_average(values: list[float], k: int) -> list[float]:\n"
                                    "    if k <= 0:\n"
                                    "        raise ValueError('Window size k must be positive.')\n"
                                    "    if k > len(values):\n"
                                    "        return []\n"
                                    "    result = []\n"
                                    "    window_sum = sum(values[:k])\n"
                                    "    result.append(window_sum / k)\n"
                                    "    for i in range(k, len(values)):\n"
                                    "        window_sum += values[i] - values[i - k]\n"
                                    "        result.append(window_sum / k)\n"
                                    "    return result\n"
                                    "```"
                                )
                            }
                        }
                    ]
                }

        elif (
            "Automated QA and Testing Specialist" in sys_prompt
            or "Testing Specialist" in sys_prompt
        ):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "```python\n"
                                "assert rolling_average([10, 20, 30, 40], 2) == [15.0, 25.0, 35.0]\n"
                                "assert rolling_average([1, 2], 5) == []\n"
                                "try:\n"
                                "    rolling_average([1, 2], 0)\n"
                                "    raise AssertionError('Expected ValueError for k <= 0')\n"
                                "except ValueError:\n"
                                "    pass\n"
                                "```"
                            )
                        }
                    }
                ]
            }

        return {"choices": [{"message": {"content": "Default response."}}]}

    async def stream(self, messages, model=None, **kwargs):
        yield ""

    async def health(self):
        return {"status": "healthy"}

    async def embeddings(self, texts, model=None):
        return [[0.0] * 16]

    def capabilities(self):
        return {"supports_text": True}


async def main():
    print("=" * 75)
    print(" Project Libra -- Phase 20: Multi-Agent Collaboration & Orchestration Demo")
    print("=" * 75)

    router = ProviderRouter()
    provider = DemoTeamProvider()
    router._providers["demo-team-provider"] = provider
    sandbox = SafePythonSandbox(default_timeout=5.0)

    team = CollaborativeTeam(
        model_id="demo-team-provider",
        provider_name="demo-team-provider",
        router=router,
        sandbox=sandbox,
        max_rounds=3,
    )

    task_prompt = "Implement a rolling average function over numerical data."
    print(f"\nUser Objective: {task_prompt}\n")

    t0 = time.perf_counter()
    async for event in team.stream_collaboration(task_prompt):
        ev_type = event.get("type")

        if ev_type == "message":
            msg = event["message"]
            sender = msg["sender_name"]
            role = msg["sender_role"].upper()
            m_type = msg["message_type"].upper()
            print(f"\n{'-' * 50}")
            print(f"[{sender} ({role}) -- {m_type}]")
            print(f"{'-' * 50}")
            print(msg["content"].strip())

        elif ev_type == "round_start":
            round_num = event["round"]
            print(f"\n{'#' * 75}")
            print(f" >>> STARTING COLLABORATION ROUND {round_num} <<<")
            print(f"{'#' * 75}")

        elif ev_type == "round_end":
            round_num = event["round"]
            print(
                f"\n[Round {round_num} Completed: Consensus Not Reached -> Advancing to Next Round]"
            )

        elif ev_type == "finish":
            traj = event["trajectory"]
            elapsed = (time.perf_counter() - t0) * 1000.0
            print(f"\n{'=' * 75}")
            print(" COLLABORATION COMPLETED")
            print(f"{'=' * 75}")
            print(f"Status: {traj.status.value.upper()}")
            print(f"Consensus Reached: {traj.consensus_reached}")
            print(f"Total Rounds: {traj.total_rounds}")
            print(f"Total Duration: {elapsed:.2f} ms")
            print(f"\n{traj.final_answer}")


if __name__ == "__main__":
    asyncio.run(main())
