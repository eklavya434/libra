"""
Libra API v1 - Self-Rewarding Language Models & LLM-as-a-Judge Endpoints (Phase 49)

Provides REST endpoints for:
- LLM-as-a-Judge evaluation with rubric scoring & position bias analysis
- Iterative Self-Rewarding alignment steps (Flywheel: Rollout -> Self-Judge -> Pair -> DPO)
- Multi-iteration benchmarking (M_0 -> M_1 -> M_2)
- Educational prompt & rubric presets
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from packages.evaluation.self_rewarding_eval import (
    PositionBiasAnalysis,
    SelfRewardingBenchmarkResult,
    SelfRewardingEvaluator,
)
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.training.self_rewarding import (
    JudgeRubric,
    LLMJudge,
    SelfRewardingConfig,
    SelfRewardingIterationResult,
    SelfRewardingTrainer,
)

router = APIRouter(prefix="/self-rewarding", tags=["Self-Rewarding Language Models"])


class JudgeRequest(BaseModel):
    """Request payload for evaluating responses using LLM-as-a-Judge."""

    instruction: str = Field(..., min_length=3, description="Instruction prompt")
    response: str = Field(..., min_length=1, description="Primary candidate response")
    response_b: Optional[str] = Field(
        default=None, description="Optional secondary response for comparative pairwise judging"
    )
    rubric_name: str = Field(default="General Helpfulness", description="Name of the rubric preset")
    debias_position: bool = Field(
        default=True, description="Mitigate position bias via order-swap averaging"
    )


class JudgeResponse(BaseModel):
    """Evaluation response from LLM-as-a-Judge."""

    overall_score: float
    normalized_score: float
    dimension_scores: dict[str, float]
    critique: str
    is_pairwise: bool = False
    pairwise_winner: Optional[str] = None
    position_bias_analysis: Optional[PositionBiasAnalysis] = None


class IterateRequest(BaseModel):
    """Request payload for executing an iterative self-rewarding step."""

    prompt: str = Field(..., min_length=3, description="Training prompt")
    iteration_id: int = Field(
        default=1, ge=1, le=10, description="Current iteration index (e.g. 1 for M_1)"
    )
    num_candidates: int = Field(
        default=3, ge=2, le=5, description="Candidates to generate per prompt"
    )
    temperature: float = Field(default=0.8, ge=0.1, le=2.0, description="Sampling temperature")
    beta: float = Field(default=0.1, ge=0.01, le=1.0, description="DPO temperature beta")
    debias_position: bool = Field(default=True, description="Enable order-swap position debiasing")


class BenchmarkRequest(BaseModel):
    """Request payload for multi-iteration benchmarking."""

    prompts: list[str] = Field(
        default=[
            "Explain what attention computes in a transformer in simple terms.",
            "Write a concise Python function to check if a string is a palindrome.",
            "Solve: A shopkeeper sells 3 items for $15 each and 2 items for $20 each. What is the total?",
        ],
        description="Test prompts to evaluate across iterations",
    )
    num_iterations: int = Field(
        default=3, ge=2, le=5, description="Number of iterative checkpoints to benchmark"
    )


# Global model singletons for educational demonstration
_policy_model: Optional[ModernTransformerLM] = None
_ref_model: Optional[ModernTransformerLM] = None
_judge: Optional[LLMJudge] = None
_trainer: Optional[SelfRewardingTrainer] = None


def _get_self_rewarding_engine() -> tuple[SelfRewardingTrainer, LLMJudge, ModernTransformerLM]:
    global _policy_model, _ref_model, _judge, _trainer
    if _policy_model is None or _trainer is None:
        cfg = ModernTransformerConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=2,
            max_context_length=128,
            hidden_dim=128,
        )
        _policy_model = ModernTransformerLM(cfg)
        _ref_model = ModernTransformerLM(cfg)
        _ref_model.load_state_dict(_policy_model.state_dict())

        _judge = LLMJudge(model=_policy_model)
        trainer_cfg = SelfRewardingConfig(
            beta=0.1,
            lr=5e-5,
            num_candidates=3,
            temperature=0.8,
            max_new_tokens=24,
            margin_threshold=0.1,
            debias_position=True,
        )
        _trainer = SelfRewardingTrainer(
            policy_model=_policy_model,
            reference_model=_ref_model,
            config=trainer_cfg,
            judge=_judge,
        )
    return _trainer, _judge, _policy_model


@router.get("/presets")
async def get_presets() -> dict[str, Any]:
    """Returns educational prompt presets and evaluation rubrics."""
    rubrics = [
        {
            "name": "General Helpfulness",
            "description": "Assesses completeness, helpfulness, and tone",
            "dimensions": [
                {
                    "name": "Correctness",
                    "description": "Factual accuracy and validity",
                    "weight": 1.5,
                },
                {
                    "name": "Helpfulness",
                    "description": "Solves the user's intent clearly",
                    "weight": 1.2,
                },
                {
                    "name": "Clarity",
                    "description": "Structured, readable formatting",
                    "weight": 1.0,
                },
            ],
        },
        {
            "name": "Code Quality",
            "description": "Evaluates algorithmic correctness, time complexity, and clean code conventions",
            "dimensions": [
                {
                    "name": "Syntax & Correctness",
                    "description": "Runs without runtime errors",
                    "weight": 2.0,
                },
                {
                    "name": "Idiomatic Style",
                    "description": "Follows standard PEP 8 / conventions",
                    "weight": 1.0,
                },
                {
                    "name": "Efficiency",
                    "description": "Optimal time and memory utilization",
                    "weight": 1.2,
                },
            ],
        },
        {
            "name": "Mathematical Reasoning",
            "description": "Evaluates step-by-step arithmetic and logical soundness",
            "dimensions": [
                {
                    "name": "Step Soundness",
                    "description": "Each deduction follows logically",
                    "weight": 2.0,
                },
                {
                    "name": "Final Answer",
                    "description": "Exact answer matches problem statement",
                    "weight": 1.5,
                },
            ],
        },
    ]

    prompts = [
        {
            "title": "Transformer Attention Explained",
            "prompt": "Explain what multi-head self-attention computes in a transformer from first principles.",
            "rubric": "General Helpfulness",
        },
        {
            "title": "Two-Sum Algorithm",
            "prompt": "Implement two-sum in Python using a hash map with O(n) time complexity.",
            "rubric": "Code Quality",
        },
        {
            "title": "Multi-Step Logic Puzzle",
            "prompt": "If 5 workers build 5 chairs in 5 days, how many days does it take 100 workers to build 100 chairs?",
            "rubric": "Mathematical Reasoning",
        },
    ]

    return {"rubrics": rubrics, "prompts": prompts}


@router.post("/judge", response_model=JudgeResponse)
async def evaluate_with_judge(payload: JudgeRequest) -> JudgeResponse:
    """Evaluates candidate response(s) using LLM-as-a-Judge."""
    _, judge, _ = _get_self_rewarding_engine()

    # Configure rubric dimensions
    rubric = JudgeRubric(name=payload.rubric_name)
    evaluator_judge = LLMJudge(model=judge.model, rubric=rubric)

    if payload.response_b is not None:
        # Pairwise comparative judging
        evaluator = SelfRewardingEvaluator(judge=evaluator_judge)
        bias_analysis = evaluator.analyze_position_bias(
            payload.instruction, payload.response, payload.response_b
        )

        s_a, s_b, critique = evaluator_judge.evaluate_pairwise(
            payload.instruction,
            payload.response,
            payload.response_b,
            debias_position=payload.debias_position,
        )
        winner = "Response A" if s_a > s_b else ("Response B" if s_b > s_a else "Tie")

        return JudgeResponse(
            overall_score=round(s_a * 5.0, 2),
            normalized_score=round(s_a, 4),
            dimension_scores={"Response A": round(s_a, 3), "Response B": round(s_b, 3)},
            critique=critique,
            is_pairwise=True,
            pairwise_winner=winner,
            position_bias_analysis=bias_analysis,
        )
    else:
        # Single response absolute scoring
        score_res = evaluator_judge.evaluate_response(payload.instruction, payload.response)
        return JudgeResponse(
            overall_score=score_res.overall_score,
            normalized_score=score_res.normalized_score,
            dimension_scores=score_res.dimension_scores,
            critique=score_res.critique,
            is_pairwise=False,
            pairwise_winner=None,
            position_bias_analysis=None,
        )


@router.post("/iterate", response_model=SelfRewardingIterationResult)
async def run_iteration_step(payload: IterateRequest) -> SelfRewardingIterationResult:
    """Runs a complete self-rewarding step: rollout -> self-judge -> pair -> DPO update."""
    trainer, judge, _ = _get_self_rewarding_engine()
    trainer.config.num_candidates = payload.num_candidates
    trainer.config.temperature = payload.temperature
    trainer.config.beta = payload.beta
    trainer.config.debias_position = payload.debias_position

    result = trainer.train_iteration_step(payload.prompt, iteration_id=payload.iteration_id)

    # Automatically advance reference model if advancing iteration
    if payload.iteration_id > 1:
        trainer.update_reference_model()

    return result


@router.post("/benchmark", response_model=SelfRewardingBenchmarkResult)
async def benchmark_iterations_endpoint(payload: BenchmarkRequest) -> SelfRewardingBenchmarkResult:
    """Benchmarks progressive model iterations (M_0 -> M_1 -> M_2)."""
    _, judge, base_model = _get_self_rewarding_engine()
    evaluator = SelfRewardingEvaluator(judge=judge)

    # Synthesize iteration models
    iterations: list[tuple[int, ModernTransformerLM]] = []
    for i in range(payload.num_iterations):
        # Create checkpoint snapshot
        cfg = ModernTransformerConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=2,
            max_context_length=128,
            hidden_dim=128,
        )
        mod = ModernTransformerLM(cfg)
        mod.load_state_dict(base_model.state_dict())
        iterations.append((i, mod))

    return evaluator.benchmark_iterations(iterations, payload.prompts)
