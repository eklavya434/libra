"""
Libra Models - Best-of-N Test-Time Compute & Search Verifier (Phase 29)
Implements test-time compute scaling: generates N candidate reasoning trajectories,
evaluates each candidate with scalar reward scoring, and selects the optimal solution.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from packages.evaluation.automated_judge import AutomatedJudge
from packages.models.reasoning.trace_parser import ReasoningTrace, parse_reasoning_trace
from packages.providers.router import ProviderRouter, get_router


class CandidateScore(BaseModel):
    """Detailed score for a single candidate reasoning rollout."""

    index: int
    score: float = Field(..., description="Calculated reward/quality score (0.0 to 10.0)")
    trace: ReasoningTrace
    rationale: str = Field(default="", description="Scoring explanation")


class BestOfNResult(BaseModel):
    """Result of Best-of-N candidate search and verification."""

    best_candidate: ReasoningTrace
    best_score: float
    best_index: int
    total_candidates: int
    candidates: list[CandidateScore]
    selection_method: str = Field(
        default="automated_judge", description="'reward_model' or 'automated_judge'"
    )


class BestOfNVerifier:
    """Samples N candidate trajectories and verifies them with reward scoring."""

    def __init__(
        self,
        router: ProviderRouter | None = None,
        judge: AutomatedJudge | None = None,
    ) -> None:
        self.router = router or get_router()
        self.judge = judge or AutomatedJudge()

    async def search(
        self,
        prompt: str,
        model: str = "qwen3:4b",
        n_candidates: int = 3,
        temperature: float = 0.8,
        max_tokens: int = 256,
        provider_name: str | None = None,
    ) -> BestOfNResult:
        """Executes N stochastic rollouts, scores each candidate, and returns the top-ranked trajectory."""
        if n_candidates < 1:
            raise ValueError("n_candidates must be at least 1.")

        messages = [
            {
                "role": "system",
                "content": (
                    "Think step by step in detail before answering. "
                    "Enclose your reasoning trace in <think>...</think> tags, followed by your final answer."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        # Generate N rollouts concurrently
        tasks = [
            self.router.chat(
                messages=messages,
                model_id=model,
                provider_name=provider_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for _ in range(n_candidates)
        ]

        responses: list[dict[str, Any]] = await asyncio.gather(*tasks, return_exceptions=True)

        candidate_scores: list[CandidateScore] = []

        for idx, resp in enumerate(responses):
            if isinstance(resp, Exception):
                continue

            choices = resp.get("choices", [])
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            trace = parse_reasoning_trace(content)

            # Score using automated referee multi-criteria engine
            scores_dict = self.judge._score_single(prompt, trace.final_answer)
            base_score = scores_dict.get("total", 5.0)

            # Bonus for structured reasoning steps and reflection
            step_bonus = min(1.5, len(trace.steps) * 0.3)
            thought_bonus = 1.0 if trace.has_thought else 0.0

            final_score = round(min(10.0, base_score + step_bonus + thought_bonus), 2)
            rationale = (
                f"Base score: {base_score:.2f}, steps identified: {len(trace.steps)}, "
                f"internal reflection present: {trace.has_thought}"
            )

            candidate_scores.append(
                CandidateScore(
                    index=idx,
                    score=final_score,
                    trace=trace,
                    rationale=rationale,
                )
            )

        if not candidate_scores:
            fallback_trace = ReasoningTrace(
                raw_thought="",
                steps=[],
                final_answer="No valid candidates generated.",
                has_thought=False,
            )
            return BestOfNResult(
                best_candidate=fallback_trace,
                best_score=0.0,
                best_index=0,
                total_candidates=n_candidates,
                candidates=[],
                selection_method="automated_judge",
            )

        # Select highest-scoring candidate
        best = max(candidate_scores, key=lambda c: c.score)

        return BestOfNResult(
            best_candidate=best.trace,
            best_score=best.score,
            best_index=best.index,
            total_candidates=len(candidate_scores),
            candidates=candidate_scores,
            selection_method="automated_judge",
        )
