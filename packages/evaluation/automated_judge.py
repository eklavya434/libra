"""
Libra Evaluation - Automated Multi-Criteria Referee & Position-Bias Mitigation (Phase 28)
Implements position-bias-mitigated pairwise judging between candidate completions.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    """Structured verdict from the automated referee."""

    winner: str = Field(..., description="'model_a', 'model_b', or 'tie'")
    score_a: float = Field(..., description="Calculated quality score for Model A (0.0 to 10.0)")
    score_b: float = Field(..., description="Calculated quality score for Model B (0.0 to 10.0)")
    confidence: float = Field(default=0.9, description="Confidence in the judgment (0.0 to 1.0)")
    rationale: str = Field(..., description="Detailed explanation for the verdict")
    position_swapped: bool = Field(
        default=True, description="Whether dual-pass position swapping was applied"
    )
    criteria_breakdown: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Sub-scores per criterion (instruction, conciseness, coherence, structure)",
    )


class AutomatedJudge:
    """Evaluates pairs of model completions with dual-pass position bias mitigation."""

    def __init__(self, criteria_weights: dict[str, float] | None = None) -> None:
        self.weights = criteria_weights or {
            "instruction_following": 0.35,
            "coherence_and_clarity": 0.25,
            "conciseness_efficiency": 0.20,
            "formatting_structure": 0.20,
        }

    def evaluate_pair(
        self,
        prompt: str,
        completion_a: str,
        completion_b: str,
        model_a_id: str = "model_a",
        model_b_id: str = "model_b",
    ) -> JudgeVerdict:
        """Executes dual-pass evaluation to eliminate LLM position bias:

        Pass 1: Order (A, B) -> evaluates Candidate 1 vs Candidate 2.
        Pass 2: Order (B, A) -> evaluates Candidate 1 vs Candidate 2 with order swapped.
        Reconciles both passes to compute final objective verdict.
        """
        # Pass 1: Order A, B
        scores_pass1, rationale_1 = self._score_candidates(prompt, completion_a, completion_b)
        # Pass 2: Order B, A (swapped positions)
        scores_pass2, rationale_2 = self._score_candidates(prompt, completion_b, completion_a)

        # Reconcile scores
        # Model A score is average of its Candidate 1 score in Pass 1 and Candidate 2 score in Pass 2
        score_a = round(
            (scores_pass1["cand_1"]["total"] + scores_pass2["cand_2"]["total"]) / 2.0, 2
        )
        score_b = round(
            (scores_pass1["cand_2"]["total"] + scores_pass2["cand_1"]["total"]) / 2.0, 2
        )

        # Determine winner
        margin = abs(score_a - score_b)
        if margin < 0.35:
            winner = "tie"
            rationale = (
                f"Both models performed comparably (Score A: {score_a:.2f}, Score B: {score_b:.2f}, "
                f"delta: {margin:.2f} < 0.35 threshold). {rationale_1}"
            )
        elif score_a > score_b:
            winner = "model_a"
            rationale = (
                f"{model_a_id} prevailed with score {score_a:.2f} vs {score_b:.2f}. {rationale_1}"
            )
        else:
            winner = "model_b"
            rationale = (
                f"{model_b_id} prevailed with score {score_b:.2f} vs {score_a:.2f}. {rationale_2}"
            )

        criteria_breakdown = {
            model_a_id: {
                c: round((scores_pass1["cand_1"][c] + scores_pass2["cand_2"][c]) / 2.0, 2)
                for c in self.weights
            },
            model_b_id: {
                c: round((scores_pass1["cand_2"][c] + scores_pass2["cand_1"][c]) / 2.0, 2)
                for c in self.weights
            },
        }

        return JudgeVerdict(
            winner=winner,
            score_a=score_a,
            score_b=score_b,
            confidence=round(min(1.0, 0.7 + margin * 0.1), 2),
            rationale=rationale,
            position_swapped=True,
            criteria_breakdown=criteria_breakdown,
        )

    def _score_candidates(
        self,
        prompt: str,
        cand_1: str,
        cand_2: str,
    ) -> tuple[dict[str, dict[str, float]], str]:
        """Calculates multi-dimensional criteria scores for two candidates."""
        s1 = self._score_single(prompt, cand_1)
        s2 = self._score_single(prompt, cand_2)

        scores = {"cand_1": s1, "cand_2": s2}

        # Compose brief rationale
        reasons = []
        if s1["instruction_following"] != s2["instruction_following"]:
            adv = (
                "Candidate 1"
                if s1["instruction_following"] > s2["instruction_following"]
                else "Candidate 2"
            )
            reasons.append(f"{adv} followed constraints more strictly.")
        if s1["coherence_and_clarity"] != s2["coherence_and_clarity"]:
            adv = (
                "Candidate 1"
                if s1["coherence_and_clarity"] > s2["coherence_and_clarity"]
                else "Candidate 2"
            )
            reasons.append(f"{adv} showed higher coherence.")
        if s1["conciseness_efficiency"] != s2["conciseness_efficiency"]:
            adv = (
                "Candidate 1"
                if s1["conciseness_efficiency"] > s2["conciseness_efficiency"]
                else "Candidate 2"
            )
            reasons.append(f"{adv} was more direct and concise.")

        rationale = (
            " ".join(reasons)
            if reasons
            else "Responses were evenly matched in criteria evaluation."
        )
        return scores, rationale

    def _score_single(self, prompt: str, text: str) -> dict[str, float]:
        """Scores a single completion across criteria out of 10.0."""
        if not text or not text.strip():
            return {c: 0.0 for c in self.weights} | {"total": 0.0}

        cleaned = text.strip()
        words = cleaned.split()
        word_count = len(words)

        # 1. Instruction following
        inst_score = 7.0
        prompt_lower = prompt.lower()

        # Check for sentence count constraint (e.g. "in 2 sentences")
        sent_match = re.search(r"in (\d+) sentences?", prompt_lower)
        if sent_match:
            target_sentences = int(sent_match.group(1))
            actual_sentences = len(re.findall(r"[.!?]+", cleaned))
            if abs(actual_sentences - target_sentences) == 0:
                inst_score += 2.5
            elif abs(actual_sentences - target_sentences) == 1:
                inst_score += 1.0
            else:
                inst_score -= 1.5

        # Check for code constraint
        if "code" in prompt_lower or "python" in prompt_lower or "function" in prompt_lower:
            if "```" in cleaned or "def " in cleaned:
                inst_score += 2.0
            else:
                inst_score -= 2.0

        # Check for non-empty meaningful answer
        if word_count >= 5:
            inst_score = min(10.0, inst_score + 0.5)

        # 2. Coherence and clarity
        coherence_score = 7.5
        # Penalize excessive repetition (e.g. loops)
        if word_count > 10:
            unique_ratio = len(set(words)) / word_count
            if unique_ratio < 0.4:
                coherence_score -= 3.5
            elif unique_ratio > 0.7:
                coherence_score += 1.5

        # Penalize broken artifacts
        if "⚠️" in cleaned or "error" in cleaned.lower():
            coherence_score -= 2.5

        # 3. Conciseness and efficiency
        conciseness_score = 7.5
        if 15 <= word_count <= 250:
            conciseness_score = 9.0
        elif word_count > 400:
            conciseness_score = 6.0
        elif word_count < 5:
            conciseness_score = 4.0

        # 4. Formatting and structure
        format_score = 7.0
        if any(marker in cleaned for marker in ["- ", "1. ", "• ", "**", "```"]):
            format_score += 2.0
        if cleaned[0].isupper() and cleaned[-1] in ".!?`\"'":
            format_score += 1.0

        # Clamp all criteria to [1.0, 10.0]
        scores = {
            "instruction_following": round(max(1.0, min(10.0, inst_score)), 2),
            "coherence_and_clarity": round(max(1.0, min(10.0, coherence_score)), 2),
            "conciseness_efficiency": round(max(1.0, min(10.0, conciseness_score)), 2),
            "formatting_structure": round(max(1.0, min(10.0, format_score)), 2),
        }

        # Weighted composite total
        total = sum(scores[k] * self.weights[k] for k in self.weights)
        scores["total"] = round(total, 2)
        return scores
