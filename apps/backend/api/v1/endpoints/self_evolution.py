"""
Libra API v1 - Autonomous Self-Evolution & Grand Capstone Endpoints (Phase 50)

Provides REST endpoints for:
- Executing autonomous self-evolution cycles (Generate -> Self-Judge -> Pair -> DPO -> PRM Test)
- Running multi-cycle curriculum self-evolution
- 10-Pillar Grand Capstone Architecture Audit across all 50 milestones
- Educational seed curricula & hyperparameter presets
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from packages.evaluation.grand_capstone import (
    GrandCapstoneAudit,
    GrandCapstoneReport,
)
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.models.reasoning.prm import ProcessRewardModel
from packages.training.self_evolution import (
    EvolutionStepMetric,
    SelfEvolutionConfig,
    SelfEvolutionEngine,
    SelfEvolutionReport,
)
from packages.training.self_rewarding import LLMJudge

router = APIRouter(prefix="/self-evolution", tags=["Autonomous Self-Evolution & Grand Capstone"])


class EvolutionCycleRequest(BaseModel):
    """Request payload for executing a single self-evolution cycle."""

    seed_prompt: str = Field(
        default="Explain how gradient descent optimizes neural network parameters with momentum.",
        description="Curriculum task seed prompt",
    )
    cycle_id: int = Field(default=1, ge=1, le=10, description="Cycle iteration counter")
    candidates_per_prompt: int = Field(default=3, ge=2, le=5, description="Number of rollouts")
    dpo_beta: float = Field(default=0.1, ge=0.01, le=1.0, description="DPO temperature")
    prm_prune_threshold: float = Field(
        default=0.5, ge=0.1, le=0.9, description="PRM correctness threshold"
    )


class EvolutionRunRequest(BaseModel):
    """Request payload for running autonomous multi-cycle evolution."""

    seed_prompts: list[str] = Field(
        default=[
            "Explain what multi-head self-attention computes in a transformer from first principles.",
            "Write a Python function to find the longest palindromic substring with O(n^2) complexity.",
            "Solve: If a car travels at 60 mph for 2.5 hours and 45 mph for 1.5 hours, what is total distance?",
        ],
        description="Curriculum seed prompts",
    )
    max_cycles: int = Field(
        default=2, ge=1, le=5, description="Number of iterative self-evolution cycles"
    )
    candidates_per_prompt: int = Field(default=3, ge=2, le=4, description="Rollouts per prompt")
    dpo_beta: float = Field(default=0.1, ge=0.01, le=1.0, description="DPO temperature")


# Global engine singleton for educational demonstration
_policy_model: Optional[ModernTransformerLM] = None
_ref_model: Optional[ModernTransformerLM] = None
_prm: Optional[ProcessRewardModel] = None
_evolution_engine: Optional[SelfEvolutionEngine] = None


def _get_evolution_engine() -> SelfEvolutionEngine:
    global _policy_model, _ref_model, _prm, _evolution_engine
    if _policy_model is None or _evolution_engine is None:
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

        _prm = ProcessRewardModel()
        judge = LLMJudge(model=_policy_model)

        evo_cfg = SelfEvolutionConfig(
            num_cycles=2,
            candidates_per_prompt=3,
            dpo_beta=0.1,
            learning_rate=5e-5,
            margin_threshold=0.1,
            prm_prune_threshold=0.5,
            max_new_tokens=24,
        )
        _evolution_engine = SelfEvolutionEngine(
            policy_model=_policy_model,
            reference_model=_ref_model,
            config=evo_cfg,
            judge=judge,
            prm=_prm,
        )
    return _evolution_engine


@router.get("/presets")
async def get_presets() -> dict[str, Any]:
    """Returns educational self-evolution curricula and parameter presets."""
    curricula = [
        {
            "name": "STEM & Reasoning Curriculum",
            "description": "Combines arithmetic deduction, algorithmic puzzles, and physical reasoning",
            "prompts": [
                "Calculate: (15 * 4) + (36 / 3) - 18 with step-by-step verification.",
                "Explain the difference between PagedAttention and standard multi-head attention.",
                "Implement binary search in Python and analyze its worst-case space complexity.",
            ],
        },
        {
            "name": "Instruction & Helpfulness Alignment",
            "description": "Aligns model responses toward clarity, conciseness, and zero-hallucination",
            "prompts": [
                "Summarize the key mathematical properties of Rotary Position Embeddings (RoPE).",
                "Explain how Direct Preference Optimization (DPO) derives its loss without a reward model.",
                "How does speculative decoding achieve exact output parity with standard autoregression?",
            ],
        },
    ]

    stages = [
        {
            "stage": "1. Data Synthesis",
            "description": "Generates on-policy synthetic candidates for instruction tasks",
        },
        {
            "stage": "2. LLM-as-a-Judge",
            "description": "Scores candidates across correctness and helpfulness rubrics",
        },
        {
            "stage": "3. Preference Pairing",
            "description": "Curates winning and losing pairs (y_w, y_l) with margin filter",
        },
        {
            "stage": "4. DPO Alignment",
            "description": "Applies reference-regularized DPO gradient step to policy weights",
        },
        {
            "stage": "5. Speculative Acceleration",
            "description": "Verifies speedup via multi-head drafting & dynamic KV caching",
        },
        {
            "stage": "6. Verifiable Reasoning",
            "description": "Tests mathematical step accuracy with PRM tree search",
        },
    ]

    return {"curricula": curricula, "stages": stages}


@router.post("/cycle", response_model=EvolutionStepMetric)
async def run_single_cycle(payload: EvolutionCycleRequest) -> EvolutionStepMetric:
    """Executes a single 6-stage autonomous self-evolution cycle."""
    engine = _get_evolution_engine()
    engine.config.candidates_per_prompt = payload.candidates_per_prompt
    engine.config.dpo_beta = payload.dpo_beta
    engine.config.prm_prune_threshold = payload.prm_prune_threshold

    return engine.run_evolution_cycle(payload.seed_prompt, cycle_id=payload.cycle_id)


@router.post("/run", response_model=SelfEvolutionReport)
async def run_autonomous_evolution_endpoint(payload: EvolutionRunRequest) -> SelfEvolutionReport:
    """Executes multi-cycle autonomous self-evolution over curriculum prompts."""
    engine = _get_evolution_engine()
    engine.config.candidates_per_prompt = payload.candidates_per_prompt
    engine.config.dpo_beta = payload.dpo_beta

    return engine.run_autonomous_evolution(payload.seed_prompts, max_cycles=payload.max_cycles)


@router.get("/grand-audit", response_model=GrandCapstoneReport)
async def run_grand_capstone_audit() -> GrandCapstoneReport:
    """Executes the complete 10-Pillar 50-Phase Grand Capstone system audit."""
    auditor = GrandCapstoneAudit()
    return auditor.run_full_grand_audit()
