"""
Libra API v1 - RLHF & Direct Preference Optimization (DPO) Alignment Endpoints
"""

from __future__ import annotations

from typing import Any

import torch
from fastapi import APIRouter
from pydantic import BaseModel, Field

from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.models.reward_model import TransformerRewardModel
from packages.training.dpo_trainer import DPOConfig, DPOTelemetry, DPOTrainer
from packages.training.preference_dataset import PreferenceDataset, PreferenceSample

router = APIRouter(prefix="/alignment", tags=["Alignment & RLHF"])


class RewardScoreRequest(BaseModel):
    prompt: str = Field(..., description="Prompt string")
    completion: str = Field(..., description="Candidate completion string")


class RewardRankRequest(BaseModel):
    prompt: str = Field(..., description="Prompt string")
    completions: list[str] = Field(..., min_length=2, description="Candidate completions to rank")


class DPOPairInput(BaseModel):
    prompt: str = Field(..., description="Prompt string")
    chosen: str = Field(..., description="Preferred completion")
    rejected: str = Field(..., description="Dispreferred completion")


class DPOStepRequest(BaseModel):
    pairs: list[DPOPairInput] = Field(..., min_length=1, description="Batch of preference pairs")
    beta: float = Field(default=0.1, ge=0.01, le=1.0, description="KL penalty temperature beta")
    lr: float = Field(default=1e-4, ge=1e-6, le=1e-2, description="Learning rate")


# Cached lightweight models for educational alignment endpoints
_reward_model: TransformerRewardModel | None = None
_policy_model: ModernTransformerLM | None = None
_ref_model: ModernTransformerLM | None = None
_dpo_trainer: DPOTrainer | None = None


def _get_alignment_models() -> tuple[TransformerRewardModel, DPOTrainer]:
    global _reward_model, _policy_model, _ref_model, _dpo_trainer
    if _reward_model is None or _dpo_trainer is None:
        cfg = ModernTransformerConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=2,
            max_context_length=128,
            hidden_dim=128,
        )
        _reward_model = TransformerRewardModel(cfg)
        _reward_model.eval()

        _policy_model = ModernTransformerLM(cfg)
        _ref_model = ModernTransformerLM(cfg)
        _ref_model.load_state_dict(_policy_model.state_dict())

        dpo_cfg = DPOConfig(beta=0.1, lr=1e-4)
        _dpo_trainer = DPOTrainer(
            policy_model=_policy_model,
            reference_model=_ref_model,
            config=dpo_cfg,
        )

    return _reward_model, _dpo_trainer


@router.post("/reward")
async def score_completion_reward(request: RewardScoreRequest) -> dict[str, Any]:
    """Scores a single prompt-completion pair using the Transformer Reward Model."""
    reward_model, _ = _get_alignment_models()

    full_text = request.prompt + request.completion
    tokens = list(full_text.encode("utf-8"))[:128]
    if not tokens:
        tokens = [0]

    input_ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        scalar = reward_model(input_ids).item()

    return {
        "prompt": request.prompt,
        "completion": request.completion,
        "reward_score": round(scalar, 4),
        "total_tokens": len(tokens),
    }


@router.post("/reward/rank")
async def rank_completions_reward(request: RewardRankRequest) -> dict[str, Any]:
    """Ranks multiple candidate completions for a prompt by reward score."""
    reward_model, _ = _get_alignment_models()

    results = []
    for comp in request.completions:
        full_text = request.prompt + comp
        tokens = list(full_text.encode("utf-8"))[:128]
        if not tokens:
            tokens = [0]
        input_ids = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            score = reward_model(input_ids).item()
        results.append({"completion": comp, "reward_score": round(score, 4)})

    # Sort descending by reward score
    results.sort(key=lambda x: x["reward_score"], reverse=True)

    return {
        "prompt": request.prompt,
        "ranked_completions": results,
        "winner": results[0]["completion"],
    }


@router.post("/dpo/step", response_model=DPOTelemetry)
async def execute_dpo_step(request: DPOStepRequest) -> DPOTelemetry:
    """Executes a single Direct Preference Optimization training step on a batch of pairs."""
    _, trainer = _get_alignment_models()
    trainer.config.beta = request.beta

    samples = [
        PreferenceSample(prompt=p.prompt, chosen=p.chosen, rejected=p.rejected)
        for p in request.pairs
    ]
    ds = PreferenceDataset(samples=samples, max_length=128)
    batch_raw = [ds[i] for i in range(len(ds))]
    batch = PreferenceDataset.collate_fn(batch_raw)

    telemetry = trainer.train_step(batch)
    return telemetry
