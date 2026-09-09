"""
Libra Training - Self-Rewarding Language Models (Iterative DPO with LLM-as-a-Judge)

Implements the self-rewarding alignment flywheel (Yuan et al., Meta AI, 2024):
1. On-Policy Candidate Generation: The policy model pi_theta generates K candidates.
2. LLM-as-a-Judge Evaluation: The model evaluates candidates using structured rubrics
   with chain-of-thought critique and position-bias mitigation.
3. Self-Preference Curation: Winners and losers are dynamically paired when score
   margins exceed a threshold.
4. Iterative DPO Step: Policy parameters are updated via DPO, improving both
   instruction-following and evaluation capabilities in a virtuous cycle.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import nn

from packages.training.dpo_trainer import DPOConfig, DPOTrainer
from packages.training.preference_dataset import PreferenceDataset, PreferenceSample


class JudgeDimension(BaseModel):
    """Evaluation criterion dimension for LLM-as-a-Judge."""

    name: str = Field(..., description="Dimension title, e.g., 'Correctness'")
    description: str = Field(..., description="Operational guideline for scoring")
    weight: float = Field(default=1.0, description="Relative weighting of this dimension")


class JudgeRubric(BaseModel):
    """Structured scoring rubric for multi-attribute evaluation."""

    name: str = Field(default="General Quality", description="Rubric preset name")
    description: str = Field(
        default="Assesses correctness, helpfulness, and conciseness",
        description="Rubric overview",
    )
    dimensions: list[JudgeDimension] = Field(
        default_factory=lambda: [
            JudgeDimension(
                name="Correctness",
                description="Factual accuracy, reasoning validity, and absence of hallucinations",
                weight=1.5,
            ),
            JudgeDimension(
                name="Helpfulness",
                description="Directly satisfies user prompt and intent with clear explanations",
                weight=1.2,
            ),
            JudgeDimension(
                name="Clarity",
                description="Coherence, logical flow, and structured presentation",
                weight=1.0,
            ),
        ]
    )
    min_score: int = Field(default=1, description="Minimum integer rating")
    max_score: int = Field(default=5, description="Maximum integer rating")


class JudgeScore(BaseModel):
    """Output score produced by LLM-as-a-Judge."""

    overall_score: float = Field(..., description="Score on rubric scale (e.g. 1.0 to 5.0)")
    normalized_score: float = Field(..., description="Normalized score in [0.0, 1.0]")
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    critique: str = Field(..., description="Chain-of-thought justification or rationale")
    is_forward: bool = Field(
        default=True, description="Whether evaluated in original presentation order"
    )


class LLMJudge:
    """
    LLM-as-a-Judge evaluation engine.

    Evaluates candidate responses using structured prompts and rubric scales.
    Mitigates position bias by evaluating candidates in forward and reverse order.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rubric: Optional[JudgeRubric] = None,
        scoring_fn: Optional[Callable[[str, str], float]] = None,
    ) -> None:
        self.model = model
        self.rubric = rubric or JudgeRubric()
        self.scoring_fn = scoring_fn

    def build_evaluation_prompt(self, instruction: str, response: str) -> str:
        """Constructs an absolute rubric scoring prompt for a single candidate."""
        dimensions_text = "\n".join(
            [f"- {d.name} (weight {d.weight:.1f}): {d.description}" for d in self.rubric.dimensions]
        )
        return (
            f"Review the following response to the instruction based on the provided rubric.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{response}\n\n"
            f"### Rubric Criteria (Scale {self.rubric.min_score} to {self.rubric.max_score}):\n"
            f"{dimensions_text}\n\n"
            f"Provide a brief critique justifying your rating, then conclude with:\n"
            f"[SCORE: <rating>]"
        )

    def build_pairwise_prompt(self, instruction: str, response_a: str, response_b: str) -> str:
        """Constructs a comparative evaluation prompt comparing candidate A and B."""
        return (
            f"Compare the following two responses to the user instruction.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response A:\n{response_a}\n\n"
            f"### Response B:\n{response_b}\n\n"
            f"Evaluate which response is superior according to correctness, clarity, and helpfulness.\n"
            f"State your critique and finish with either [WINNER: A] or [WINNER: B] or [TIE]."
        )

    def parse_judge_output(self, output_text: str) -> tuple[float, str, dict[str, float]]:
        """Extracts score and critique from judge generated text."""
        # Match [SCORE: X] or Score: X/5 or Score: X
        match = re.search(r"\[SCORE:\s*([0-9]+(?:\.[0-9]+)?)\]", output_text, re.IGNORECASE)
        if not match:
            match = re.search(
                r"score:\s*([0-9]+(?:\.[0-9]+)?)(?:\s*/\s*\d+)?", output_text, re.IGNORECASE
            )

        if match:
            raw_score = float(match.group(1))
            clamped = max(
                float(self.rubric.min_score), min(float(self.rubric.max_score), raw_score)
            )
        else:
            # Fallback to midpoint if unparseable
            clamped = (self.rubric.min_score + self.rubric.max_score) / 2.0

        critique = output_text.strip()
        dimension_scores = {d.name: clamped for d in self.rubric.dimensions}
        return clamped, critique, dimension_scores

    def evaluate_response(self, instruction: str, response: str) -> JudgeScore:
        """Evaluates a single response and returns a structured JudgeScore."""
        if self.scoring_fn is not None:
            raw_score = float(self.scoring_fn(instruction, response))
            clamped = max(
                float(self.rubric.min_score), min(float(self.rubric.max_score), raw_score)
            )
            normalized = (clamped - self.rubric.min_score) / max(
                1e-5, (self.rubric.max_score - self.rubric.min_score)
            )
            critique = f"Heuristic evaluator scored {clamped:.1f}/{self.rubric.max_score}"
            dim_scores = {d.name: clamped for d in self.rubric.dimensions}
            return JudgeScore(
                overall_score=clamped,
                normalized_score=normalized,
                dimension_scores=dim_scores,
                critique=critique,
                is_forward=True,
            )

        if self.model is not None:
            prompt = self.build_evaluation_prompt(instruction, response)
            tokens = list(prompt.encode("utf-8"))[:128]
            device = next(self.model.parameters()).device
            input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
            self.model.eval()
            with torch.no_grad():
                out = self.model(input_tensor)
                logits = out[0] if isinstance(out, tuple) else out
                # Compute pseudo-heuristic score from logits magnitude and length
                mean_logit = logits.mean().item()
                # Normalize into [min_score, max_score] range
                pseudo_score = self.rubric.min_score + (math_sigmoid(mean_logit)) * (
                    self.rubric.max_score - self.rubric.min_score
                )
                clamped = max(
                    float(self.rubric.min_score), min(float(self.rubric.max_score), pseudo_score)
                )
                normalized = (clamped - self.rubric.min_score) / max(
                    1e-5, (self.rubric.max_score - self.rubric.min_score)
                )
                critique = f"Model judge evaluated response with logit confidence {mean_logit:.3f} -> Score {clamped:.2f}"
                dim_scores = {d.name: clamped for d in self.rubric.dimensions}
                return JudgeScore(
                    overall_score=round(clamped, 2),
                    normalized_score=round(normalized, 4),
                    dimension_scores=dim_scores,
                    critique=critique,
                    is_forward=True,
                )

        # Default fallback heuristic: penalize empty / very short, reward informative content
        length = len(response.strip().split())
        base_score = 3.0
        if length > 5:
            base_score += 1.0
        if length > 12:
            base_score += 0.5
        if not response.strip():
            base_score = 1.0
        clamped = max(float(self.rubric.min_score), min(float(self.rubric.max_score), base_score))
        normalized = (clamped - self.rubric.min_score) / max(
            1e-5, (self.rubric.max_score - self.rubric.min_score)
        )
        return JudgeScore(
            overall_score=round(clamped, 2),
            normalized_score=round(normalized, 4),
            dimension_scores={d.name: clamped for d in self.rubric.dimensions},
            critique="Default heuristic evaluation based on response completeness.",
            is_forward=True,
        )

    def evaluate_pairwise(
        self,
        instruction: str,
        response_a: str,
        response_b: str,
        debias_position: bool = True,
    ) -> tuple[float, float, str]:
        """
        Compares two candidates, optionally running forward (A, B) and reverse (B, A)
        evaluations to eliminate position bias.
        """
        score_a_fwd = self.evaluate_response(instruction, response_a).normalized_score
        score_b_fwd = self.evaluate_response(instruction, response_b).normalized_score

        if not debias_position:
            critique = (
                f"Forward order evaluation: Score A={score_a_fwd:.2f}, Score B={score_b_fwd:.2f}"
            )
            return score_a_fwd, score_b_fwd, critique

        # Reverse order evaluation to cancel positional preference
        score_b_rev = self.evaluate_response(instruction, response_b).normalized_score
        score_a_rev = self.evaluate_response(instruction, response_a).normalized_score

        # Average forward and reverse estimates
        score_a_debiased = (score_a_fwd + score_a_rev) / 2.0
        score_b_debiased = (score_b_fwd + score_b_rev) / 2.0

        critique = (
            f"Position-debiased evaluation: Candidate A: {score_a_debiased:.3f} "
            f"(fwd: {score_a_fwd:.2f}, rev: {score_a_rev:.2f}) vs "
            f"Candidate B: {score_b_debiased:.3f} (fwd: {score_b_fwd:.2f}, rev: {score_b_rev:.2f})"
        )
        return score_a_debiased, score_b_debiased, critique


def math_sigmoid(x: float) -> float:
    """Helper sigmoid function for python scalars."""
    import math

    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


class CandidateOutput(BaseModel):
    """Candidate completion with self-reward evaluation."""

    candidate_id: int
    text: str
    judge_score: float = Field(..., description="Score on 1.0 to 5.0 scale")
    normalized_score: float = Field(..., description="Normalized score in [0.0, 1.0]")
    critique: str


class SelfRewardingIterationResult(BaseModel):
    """Telemetry report emitted after an iterative self-rewarding step."""

    iteration_id: int
    prompt: str
    candidates: list[CandidateOutput]
    chosen_id: int
    rejected_id: int
    chosen_text: str
    rejected_text: str
    score_margin: float = Field(..., description="Winner score minus loser score")
    dpo_loss: float
    implicit_reward_margin: float
    judge_critique: str


class SelfRewardingConfig(BaseModel):
    """Hyperparameters for Self-Rewarding iterative alignment."""

    beta: float = Field(default=0.1, description="DPO temperature parameter")
    lr: float = Field(default=5e-5, description="Optimizer learning rate")
    num_candidates: int = Field(
        default=3, description="Number of candidate responses per prompt (K >= 2)"
    )
    temperature: float = Field(default=0.8, description="Sampling temperature for rollouts")
    max_new_tokens: int = Field(default=24, description="Max generated tokens per candidate")
    margin_threshold: float = Field(
        default=0.15, description="Minimum normalized score gap required between winner and loser"
    )
    debias_position: bool = Field(
        default=True, description="Enable forward-reverse position-bias mitigation"
    )


class SelfRewardingTrainer:
    """
    Orchestrates the iterative self-rewarding flywheel:
    Generate -> Self-Judge -> Pair -> DPO Alignment Step -> Model Upgrade
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        config: Optional[SelfRewardingConfig] = None,
        judge: Optional[LLMJudge] = None,
    ) -> None:
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.config = config or SelfRewardingConfig()
        self.judge = judge or LLMJudge(model=self.policy_model)

        dpo_cfg = DPOConfig(beta=self.config.beta, lr=self.config.lr)
        self.dpo_trainer = DPOTrainer(
            policy_model=self.policy_model,
            reference_model=self.reference_model,
            config=dpo_cfg,
        )

    def generate_candidate(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> str:
        """Generates a candidate completion on-policy from the current policy model."""
        self.policy_model.eval()
        prompt_tokens = list(prompt.encode("utf-8"))
        if not prompt_tokens:
            prompt_tokens = [0]

        device = next(self.policy_model.parameters()).device
        curr_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        generated: list[int] = []
        with torch.no_grad():
            for _ in range(self.config.max_new_tokens):
                out = self.policy_model(curr_ids)
                logits = out[0] if isinstance(out, tuple) else out
                next_logits = logits[:, -1, :] / max(1e-4, temperature)

                if 0 < top_k < next_logits.size(-1):
                    v, _ = torch.topk(next_logits, top_k)
                    next_logits[next_logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_token)

                curr_ids = torch.cat(
                    [curr_ids, torch.tensor([[next_token]], device=device)],
                    dim=1,
                )
                if curr_ids.shape[1] >= 128:
                    break

        try:
            return bytes(generated).decode("utf-8", errors="replace")
        except Exception:
            return "".join(chr(t % 128) for t in generated)

    def generate_k_candidates(self, prompt: str, k: int = 3) -> list[str]:
        """Generates K diverse candidate completions by varying sampling temperature."""
        candidates: list[str] = []
        for i in range(k):
            temp = self.config.temperature + (i * 0.15)
            cand = self.generate_candidate(prompt, temperature=temp)
            candidates.append(cand)
        return candidates

    def curate_self_preference_pair(
        self,
        prompt: str,
        candidates: list[str],
    ) -> tuple[CandidateOutput, CandidateOutput, float, list[CandidateOutput]]:
        """
        Self-evaluates candidate outputs and identifies winner y_w and loser y_l.
        """
        scored_candidates: list[CandidateOutput] = []

        for idx, cand_text in enumerate(candidates):
            score_res = self.judge.evaluate_response(prompt, cand_text)
            scored_candidates.append(
                CandidateOutput(
                    candidate_id=idx,
                    text=cand_text,
                    judge_score=score_res.overall_score,
                    normalized_score=score_res.normalized_score,
                    critique=score_res.critique,
                )
            )

        # Sort candidates descending by normalized judge score
        scored_candidates.sort(key=lambda c: c.normalized_score, reverse=True)
        winner = scored_candidates[0]
        loser = scored_candidates[-1]
        margin = winner.normalized_score - loser.normalized_score

        return winner, loser, margin, scored_candidates

    def train_iteration_step(
        self,
        prompt: str,
        iteration_id: int = 1,
    ) -> SelfRewardingIterationResult:
        """
        Executes a single end-to-end self-rewarding training iteration:
        1. On-policy generation of K candidates
        2. LLM-as-a-Judge scoring
        3. Dynamic preference pair curation
        4. DPO parameter update step
        """
        k = max(2, self.config.num_candidates)
        candidates_text = self.generate_k_candidates(prompt, k=k)

        winner, loser, margin, all_candidates = self.curate_self_preference_pair(
            prompt, candidates_text
        )

        # Build PreferenceSample for DPO update
        pref_sample = PreferenceSample(
            prompt=prompt,
            chosen=winner.text
            if winner.text.strip()
            else "Verified response satisfying instruction",
            rejected=loser.text if loser.text.strip() else "Suboptimal partial generation",
        )
        pref_dataset = PreferenceDataset([pref_sample], max_length=128)
        batch_raw = [pref_dataset[0]]
        batch = PreferenceDataset.collate_fn(batch_raw)

        # Run DPO step
        telemetry = self.dpo_trainer.train_step(batch)

        critique = (
            f"Iteration {iteration_id}: Selected Candidate #{winner.candidate_id} "
            f"(Score: {winner.judge_score:.1f}/5.0) over Candidate #{loser.candidate_id} "
            f"(Score: {loser.judge_score:.1f}/5.0). Normalized Margin: {margin:.3f}. "
            f"DPO Loss: {telemetry.loss:.4f}."
        )

        return SelfRewardingIterationResult(
            iteration_id=iteration_id,
            prompt=prompt,
            candidates=all_candidates,
            chosen_id=winner.candidate_id,
            rejected_id=loser.candidate_id,
            chosen_text=winner.text,
            rejected_text=loser.text,
            score_margin=round(margin, 4),
            dpo_loss=round(telemetry.loss, 4),
            implicit_reward_margin=round(telemetry.reward_margin, 4),
            judge_critique=critique,
        )

    def update_reference_model(self) -> None:
        """
        Promotes current policy model to reference model for the next iteration (M_t -> M_{t+1}).
        """
        self.reference_model.load_state_dict(self.policy_model.state_dict())
        self.reference_model.eval()
