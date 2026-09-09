"""
Libra Models - Self-Play Trajectory Generator & Preference Pair Synthesizer (Phase 43)

Generates synthetic self-play reasoning games, explores branching trajectories,
evaluates intermediate validity with Process Reward Models (PRM), and synthesizes
step-level DPO preference pairs (prompt, chosen, rejected) for continuous self-improvement.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from packages.models.reasoning.prm import ProcessRewardModel, TraceEvaluationResult


@dataclass
class DPOPreferencePair:
    """Pair of chosen and rejected reasoning trajectories for DPO alignment."""

    prompt: str
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float
    reward_margin: float
    failing_step_index: Optional[int]


class ReasoningSelfPlay:
    """
    Self-play generator producing multi-step reasoning trajectories,
    identifying critical reasoning failures with PRMs, and outputting DPO pairs.
    """

    SAMPLE_PROBLEMS = [
        {
            "prompt": "Using numbers [4, 4, 7, 7] and basic arithmetic (+, -, *, /), make exactly 24.",
            "correct_steps": [
                "Step 1: Compute 7 / 7 = 1.",
                "Step 2: Subtract from the other 7: 7 - 1 = 6.",
                "Step 3: Multiply by 4: 4 * 6 = 24. Exact match!",
            ],
            "flawed_steps": [
                "Step 1: Compute 7 * 7 = 49.",
                "Step 2: Subtract 4: 49 - 4 = 45.",
                "Step 3: 45 - 4 = 24. This contradicts arithmetic (45 - 4 = 41).",
            ],
        },
        {
            "prompt": "Solve for x: 3 * x + 15 = 42.",
            "correct_steps": [
                "Step 1: Subtract 15 from both sides: 3 * x = 42 - 15 = 27.",
                "Step 2: Divide by 3: x = 27 / 3 = 9.",
                "Step 3: Check: 3 * 9 = 27, and 27 + 15 = 42. Verified.",
            ],
            "flawed_steps": [
                "Step 1: Add 15 to both sides: 3 * x = 42 + 15 = 57.",
                "Step 2: Divide by 3: x = 57 / 3 = 19.",
                "Step 3: Check: 3 * 19 + 15 = 72 (not 42). Result is incorrect.",
            ],
        },
        {
            "prompt": "Find the area of a right triangle with legs of length 8 and 15.",
            "correct_steps": [
                "Step 1: The area formula for a triangle is Area = (1/2) * base * height.",
                "Step 2: Substitute legs: Area = 0.5 * 8 * 15.",
                "Step 3: Compute: 0.5 * 8 = 4, then 4 * 15 = 60. Final Area = 60.",
            ],
            "flawed_steps": [
                "Step 1: The area is base * height without factor of 1/2.",
                "Step 2: Multiply 8 * 15 = 120.",
                "Step 3: Area = 120. (Error: omitted 1/2 factor).",
            ],
        },
    ]

    def __init__(self, prm: Optional[ProcessRewardModel] = None) -> None:
        self.prm = prm or ProcessRewardModel()

    def generate_pair(
        self,
        problem_idx: Optional[int] = None,
    ) -> DPOPreferencePair:
        """
        Rolls out a self-play scenario, scores both branches using the PRM,
        and assigns the superior trajectory as `chosen` and the defective one as `rejected`.
        """
        prob = (
            self.SAMPLE_PROBLEMS[problem_idx % len(self.SAMPLE_PROBLEMS)]
            if problem_idx is not None
            else random.choice(self.SAMPLE_PROBLEMS)
        )

        prompt = prob["prompt"]
        eval_corr: TraceEvaluationResult = self.prm.score_trace(prob["correct_steps"])
        eval_flaw: TraceEvaluationResult = self.prm.score_trace(prob["flawed_steps"])

        chosen_text = "\n".join(prob["correct_steps"])
        rejected_text = "\n".join(prob["flawed_steps"])
        chosen_score = eval_corr.mean_step_score
        rejected_score = eval_flaw.mean_step_score

        margin = round(chosen_score - rejected_score, 3)

        return DPOPreferencePair(
            prompt=prompt,
            chosen=chosen_text,
            rejected=rejected_text,
            chosen_score=chosen_score,
            rejected_score=rejected_score,
            reward_margin=margin,
            failing_step_index=eval_flaw.first_error_index,
        )

    def generate_batch(self, count: int = 3) -> List[Dict[str, Any]]:
        """Generates a batch of step-level preference datasets."""
        batch: List[Dict[str, Any]] = []
        for i in range(count):
            pair = self.generate_pair(problem_idx=i)
            batch.append(
                {
                    "prompt": pair.prompt,
                    "chosen": pair.chosen,
                    "rejected": pair.rejected,
                    "chosen_score": pair.chosen_score,
                    "rejected_score": pair.rejected_score,
                    "reward_margin": pair.reward_margin,
                    "failing_step_index": pair.failing_step_index,
                }
            )
        return batch
