"""
Libra Models - Self-Consistency Reasoning Engine (Phase 29)
Implements multi-trajectory Chain-of-Thought rollouts with majority voting consensus
(Wang et al., 2022). Generates diverse reasoning paths and selects the modal answer.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from packages.models.reasoning.trace_parser import ReasoningTrace, parse_reasoning_trace
from packages.providers.router import ProviderRouter, get_router


class SelfConsistencyResult(BaseModel):
    """Aggregate result of multi-path self-consistency evaluation."""

    consensus_answer: str = Field(..., description="Modal answer selected by majority vote")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Agreement confidence (vote fraction)"
    )
    total_paths: int = Field(..., ge=1, description="Number of sampled reasoning rollouts")
    vote_distribution: dict[str, int] = Field(
        default_factory=dict, description="Frequency breakdown per normalized answer"
    )
    trajectories: list[ReasoningTrace] = Field(
        default_factory=list, description="All sampled reasoning trajectories"
    )
    winning_trajectory: ReasoningTrace | None = Field(
        None, description="Representative trajectory yielding the consensus answer"
    )


def normalize_answer(raw_answer: str) -> str:
    """Normalizes candidate answer strings for canonical voting equivalence."""
    if not raw_answer:
        return ""

    text = raw_answer.strip().lower()

    # 1. Look for explicit "the answer is X" or "result: X"
    match_explicit = re.search(
        r"(?:the\s+answer\s+is|result\s*(?:is|:)|therefore\s*,?\s*|final\s+answer\s*:)\s*([^\n.]+)",
        text,
    )
    if match_explicit:
        text = match_explicit.group(1).strip()

    # 2. Extract final standalone number or percentage if present
    num_match = re.search(r"(?:^|\s)(-?\d+(?:\.\d+)?%?)(?:\s|[.,]|$)", text)
    if num_match and len(text.split()) <= 4:
        return num_match.group(1)

    # 3. Clean punctuation and normalize spacing
    text = re.sub(r"[\s\-_]+", " ", text)
    text = re.sub(r"[^\w\s$%.,]", "", text).strip()
    text = text.rstrip(".,;:")
    return text.strip()


class SelfConsistencyEngine:
    """Orchestrates parallel stochastic rollouts and aggregates consensus votes."""

    def __init__(self, router: ProviderRouter | None = None) -> None:
        self.router = router or get_router()

    async def evaluate(
        self,
        prompt: str,
        model: str = "qwen3:4b",
        num_paths: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 256,
        system_prompt: str | None = None,
        provider_name: str | None = None,
    ) -> SelfConsistencyResult:
        """Executes num_paths parallel rollouts, parses reasoning traces, and votes on consensus."""
        if num_paths < 1:
            raise ValueError("num_paths must be at least 1.")

        effective_system = system_prompt or (
            "Think step by step before providing your answer. "
            "Enclose your thinking process in <think>...</think> tags, then state the final answer clearly."
        )

        messages = [
            {"role": "system", "content": effective_system},
            {"role": "user", "content": prompt},
        ]

        # Generate rollouts concurrently
        tasks = [
            self.router.chat(
                messages=messages,
                model_id=model,
                provider_name=provider_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for _ in range(num_paths)
        ]

        responses: list[dict[str, Any]] = await asyncio.gather(*tasks, return_exceptions=True)

        trajectories: list[ReasoningTrace] = []
        normalized_answers: list[str] = []

        for resp in responses:
            if isinstance(resp, Exception):
                continue
            choices = resp.get("choices", [])
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            trace = parse_reasoning_trace(content)
            trajectories.append(trace)
            norm = normalize_answer(trace.final_answer)
            normalized_answers.append(norm if norm else trace.final_answer.strip())

        if not trajectories:
            return SelfConsistencyResult(
                consensus_answer="No successful responses generated.",
                confidence=0.0,
                total_paths=num_paths,
                vote_distribution={},
                trajectories=[],
                winning_trajectory=None,
            )

        # Count votes
        counter = Counter(normalized_answers)
        most_common = counter.most_common()
        winner_norm, winner_votes = most_common[0]
        confidence = round(winner_votes / len(trajectories), 2)

        # Find representative winning trajectory
        winning_trace = next(
            (t for t, n in zip(trajectories, normalized_answers, strict=False) if n == winner_norm),
            trajectories[0],
        )

        return SelfConsistencyResult(
            consensus_answer=winning_trace.final_answer,
            confidence=confidence,
            total_paths=len(trajectories),
            vote_distribution=dict(counter),
            trajectories=trajectories,
            winning_trajectory=winning_trace,
        )
