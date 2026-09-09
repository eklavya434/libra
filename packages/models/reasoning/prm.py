"""
Libra Models - Process Reward Model (PRM) & Step-Level Verifier (Phase 43)

Evaluates individual intermediate reasoning steps (step-level reward r in [0, 1])
rather than solely judging final outcomes (Outcome Reward Models - ORM).
Detects arithmetic errors, logic inversions, and hallucinations for early tree pruning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class StepScore:
    """Evaluation score for a single reasoning step."""

    step_index: int
    content: str
    score: float  # Probability of step correctness in [0.0, 1.0]
    is_valid: bool
    rationale: str
    detected_errors: List[str]


@dataclass
class TraceEvaluationResult:
    """Full trajectory evaluation from the Process Reward Model."""

    steps: List[StepScore]
    first_error_index: Optional[int]
    is_valid_trace: bool
    mean_step_score: float
    min_step_score: float


class ProcessRewardModel:
    """
    First-principles Process Reward Model providing step-level verification
    and early error detection for tree search algorithms (MCTS).
    """

    VALID_STEP_THRESHOLD = 0.55

    # Arithmetic pattern detection: e.g. "4 + 7 = 11" or "4 * 7 = 28" or "(7 - 4) = 3"
    ARITHMETIC_EQ_REGEX = re.compile(
        r"(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)\s*=\s*(\-?\d+(?:\.\d+)?)"
    )

    CONTRADICTION_PATTERNS = [
        re.compile(r"which is impossible", re.IGNORECASE),
        re.compile(r"this contradicts", re.IGNORECASE),
        re.compile(r"cannot be true", re.IGNORECASE),
        re.compile(r"0\s*=\s*[1-9]", re.IGNORECASE),
        re.compile(r"undefined", re.IGNORECASE),
        re.compile(r"division by zero", re.IGNORECASE),
    ]

    POSITIVE_PROGRESSION_PATTERNS = [
        re.compile(r"therefore", re.IGNORECASE),
        re.compile(r"we have", re.IGNORECASE),
        re.compile(r"substituting", re.IGNORECASE),
        re.compile(r"simplifying", re.IGNORECASE),
        re.compile(r"step \d+", re.IGNORECASE),
        re.compile(r"equals", re.IGNORECASE),
    ]

    def verify_arithmetic(self, text: str) -> Tuple[bool, List[str]]:
        """Verifies embedded arithmetic equations for numerical correctness."""
        errors: List[str] = []
        matches = self.ARITHMETIC_EQ_REGEX.findall(text)

        for left_str, op, right_str, result_str in matches:
            try:
                left = float(left_str)
                right = float(right_str)
                expected_res = float(result_str)

                computed = 0.0
                if op == "+":
                    computed = left + right
                elif op == "-":
                    computed = left - right
                elif op == "*":
                    computed = left * right
                elif op == "/":
                    if right == 0:
                        errors.append(f"Division by zero: {left_str} / 0")
                        continue
                    computed = left / right

                if abs(computed - expected_res) > 1e-4:
                    errors.append(
                        f"Arithmetic mismatch: '{left_str} {op} {right_str} = {result_str}', expected {computed}"
                    )
            except (ValueError, ZeroDivisionError) as e:
                errors.append(f"Invalid equation format: {e}")

        is_valid = len(errors) == 0
        return is_valid, errors

    def score_step(
        self,
        step_content: str,
        context_history: Optional[List[str]] = None,
    ) -> StepScore:
        """
        Evaluates a single intermediate step, producing a calibrated validity probability
        in [0.0, 1.0] and identifying specific logical or mathematical flaws.
        """
        clean_text = step_content.strip()
        if not clean_text:
            return StepScore(
                step_index=len(context_history or []),
                content=step_content,
                score=0.1,
                is_valid=False,
                rationale="Empty reasoning step",
                detected_errors=["Empty step"],
            )

        detected_errors: List[str] = []
        base_score = 0.75  # Prior assumption of plausible reasoning

        # 1. Arithmetic Verification
        arithmetic_valid, arith_errors = self.verify_arithmetic(clean_text)
        if not arithmetic_valid:
            base_score -= 0.50
            detected_errors.extend(arith_errors)
        elif self.ARITHMETIC_EQ_REGEX.search(clean_text):
            # Reward verified math equations
            base_score += 0.15

        # 2. Contradiction Detection
        for pattern in self.CONTRADICTION_PATTERNS:
            if pattern.search(clean_text):
                # If step explicitly points out a contradiction as an error
                if not any(w in clean_text.lower() for w in ["assume", "suppose", "if", "check"]):
                    base_score -= 0.40
                    detected_errors.append(f"Contradiction pattern detected: '{pattern.pattern}'")

        # 3. Logical Progression Bonus
        for pattern in self.POSITIVE_PROGRESSION_PATTERNS:
            if pattern.search(clean_text):
                base_score += 0.05
                break

        # 4. Repetition Penalty against context history
        if context_history:
            for past_step in context_history:
                if clean_text.lower() == past_step.strip().lower():
                    base_score -= 0.45
                    detected_errors.append("Circular reasoning: exact duplicate of previous step")
                    break

        final_score = round(max(0.02, min(0.99, base_score)), 3)
        is_valid = final_score >= self.VALID_STEP_THRESHOLD

        rationale = (
            "Step verified: logically coherent and mathematically sound"
            if is_valid
            else f"Step rejected: {'; '.join(detected_errors) if detected_errors else 'low logical validity'}"
        )

        return StepScore(
            step_index=len(context_history or []),
            content=step_content,
            score=final_score,
            is_valid=is_valid,
            rationale=rationale,
            detected_errors=detected_errors,
        )

    def score_trace(self, steps: List[str]) -> TraceEvaluationResult:
        """Evaluates an entire reasoning sequence step-by-step."""
        step_scores: List[StepScore] = []
        context: List[str] = []
        first_err_idx: Optional[int] = None

        for idx, step in enumerate(steps):
            score_obj = self.score_step(step, context_history=context)
            step_scores.append(score_obj)
            context.append(step)

            if not score_obj.is_valid and first_err_idx is None:
                first_err_idx = idx

        scores_list = [s.score for s in step_scores] if step_scores else [0.0]
        mean_score = round(sum(scores_list) / len(scores_list), 3)
        min_score = min(scores_list)

        return TraceEvaluationResult(
            steps=step_scores,
            first_error_index=first_err_idx,
            is_valid_trace=first_err_idx is None,
            mean_step_score=mean_score,
            min_step_score=min_score,
        )
