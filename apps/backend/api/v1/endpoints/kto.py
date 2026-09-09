"""
Libra API v1 - Kahneman-Tversky Optimization (KTO) & Online DPO Endpoints
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from packages.evaluation.alignment_eval import AlignmentComparisonResult, AlignmentEvaluator
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.training.kto_dataset import KTODataset, KTOSample
from packages.training.kto_trainer import KTOConfig, KTOTelemetry, KTOTrainer
from packages.training.online_dpo_trainer import (
    OnlineDPOConfig,
    OnlineDPOTelemetry,
    OnlineDPOTrainer,
)

router = APIRouter(prefix="/kto", tags=["Kahneman-Tversky Optimization & Online DPO"])


class KTOSampleInput(BaseModel):
    """Input payload for a single unpaired binary feedback sample."""

    prompt: str = Field(..., description="Prompt text")
    completion: str = Field(..., description="Candidate completion text")
    is_desirable: bool = Field(..., description="True if desirable (+1), False if undesirable (-1)")


class KTOStepRequest(BaseModel):
    """Request payload for executing a KTO training step."""

    samples: list[KTOSampleInput] = Field(..., min_length=1, description="Binary feedback samples")
    beta: float = Field(default=0.1, ge=0.01, le=1.0, description="KL penalty temperature beta")
    desirable_weight: float = Field(
        default=1.0, ge=0.1, le=5.0, description="lambda_D: Weight for desirable outputs"
    )
    undesirable_weight: float = Field(
        default=1.33,
        ge=0.1,
        le=5.0,
        description="lambda_U: Weight for undesirable outputs (loss aversion)",
    )
    lr: float = Field(default=5e-5, ge=1e-6, le=1e-2, description="Optimization learning rate")


class OnlineDPOStepRequest(BaseModel):
    """Request payload for executing an on-policy exploration and DPO step."""

    prompts: list[str] = Field(..., min_length=1, description="List of prompt strings")
    beta: float = Field(default=0.1, ge=0.01, le=1.0, description="DPO temperature beta")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0, description="Sampling temperature")
    num_candidates: int = Field(
        default=2, ge=2, le=4, description="Candidates to generate per prompt"
    )


class AlignmentEvalRequest(BaseModel):
    """Request payload for multi-paradigm alignment evaluation."""

    prompts: list[str] = Field(
        default=[
            "Explain quantum computing simply.",
            "Write a Python function to reverse a list.",
            "What is photosynthesis?",
        ],
        description="Prompts to benchmark across paradigms",
    )


_policy_model: ModernTransformerLM | None = None
_ref_model: ModernTransformerLM | None = None
_kto_trainer: KTOTrainer | None = None
_online_dpo_trainer: OnlineDPOTrainer | None = None


def _heuristic_reward_scorer(prompt: str, completion: str) -> float:
    """Lightweight deterministic reward scorer for educational alignment demos."""
    clean_comp = completion.strip()
    score = 0.5

    if len(clean_comp) > 5:
        score += 0.2
    if any(p in clean_comp for p in [".", ",", ":", ";", "\n"]):
        score += 0.2
    if len(clean_comp) > 3 and len(set(clean_comp)) < 3:
        score -= 0.4

    return max(-1.0, min(1.0, round(score, 3)))


def _get_alignment_environment() -> tuple[
    KTOTrainer, OnlineDPOTrainer, ModernTransformerLM, ModernTransformerLM
]:
    global _policy_model, _ref_model, _kto_trainer, _online_dpo_trainer
    if _policy_model is None or _kto_trainer is None or _online_dpo_trainer is None:
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

        kto_cfg = KTOConfig(beta=0.1, desirable_weight=1.0, undesirable_weight=1.33, lr=5e-5)
        _kto_trainer = KTOTrainer(
            policy_model=_policy_model,
            reference_model=_ref_model,
            config=kto_cfg,
        )

        online_cfg = OnlineDPOConfig(beta=0.1, lr=5e-5, temperature=0.8, max_new_tokens=24)
        _online_dpo_trainer = OnlineDPOTrainer(
            policy_model=_policy_model,
            reference_model=_ref_model,
            config=online_cfg,
            reward_fn=_heuristic_reward_scorer,
        )

    return _kto_trainer, _online_dpo_trainer, _policy_model, _ref_model


@router.post("/step", response_model=KTOTelemetry)
async def execute_kto_step(request: KTOStepRequest) -> KTOTelemetry:
    """Executes a single Kahneman-Tversky Optimization step on unpaired binary samples."""
    kto_trainer, _, _, _ = _get_alignment_environment()
    kto_trainer.config.beta = request.beta
    kto_trainer.config.desirable_weight = request.desirable_weight
    kto_trainer.config.undesirable_weight = request.undesirable_weight

    samples = [
        KTOSample(prompt=s.prompt, completion=s.completion, is_desirable=s.is_desirable)
        for s in request.samples
    ]

    dataset = KTODataset(samples=samples, max_length=128)
    batch_raw = [dataset[i] for i in range(len(dataset))]
    batch = KTODataset.collate_fn(batch_raw)

    telemetry = kto_trainer.train_step(batch)
    return telemetry


@router.post("/online-dpo/step", response_model=list[OnlineDPOTelemetry])
async def execute_online_dpo_step(request: OnlineDPOStepRequest) -> list[OnlineDPOTelemetry]:
    """Generates on-policy responses, scores them, and applies an online DPO gradient step."""
    _, online_trainer, _, _ = _get_alignment_environment()
    online_trainer.config.beta = request.beta
    online_trainer.config.temperature = request.temperature
    online_trainer.config.num_candidates = request.num_candidates

    telemetry = online_trainer.step(request.prompts, scorer_fn=_heuristic_reward_scorer)
    return telemetry


@router.post("/evaluate", response_model=AlignmentComparisonResult)
async def evaluate_alignment_paradigms(request: AlignmentEvalRequest) -> AlignmentComparisonResult:
    """Compares SFT baseline, Offline DPO, Online DPO, and KTO under educational conditions."""
    kto_trainer, online_trainer, policy_model, ref_model = _get_alignment_environment()

    policies = {
        "Offline DPO Policy": policy_model,
        "Online On-Policy DPO": online_trainer.policy_model,
        "KTO Prospect Policy": kto_trainer.policy_model,
    }

    result = AlignmentEvaluator.compare_paradigms(
        policy_models=policies,
        reference_model=ref_model,
        prompts=request.prompts,
        reward_scorer=_heuristic_reward_scorer,
        max_tokens=24,
    )
    return result


@router.get("/presets")
async def get_kto_presets() -> dict[str, Any]:
    """Returns educational preset datasets and Prospect Theory configurations."""
    return {
        "presets": [
            {
                "id": "helpfulness_binary",
                "name": "Helpfulness & Directness (KTO Unpaired)",
                "description": "Binary feedback dataset where answers are labeled as either concise/helpful or evasive.",
                "beta": 0.1,
                "desirable_weight": 1.0,
                "undesirable_weight": 1.33,
                "samples": [
                    {
                        "prompt": "How do I calculate gradient in PyTorch?",
                        "completion": "Call loss.backward() and inspect tensor.grad.",
                        "is_desirable": True,
                    },
                    {
                        "prompt": "How do I calculate gradient in PyTorch?",
                        "completion": "PyTorch is a framework for deep learning. You should read the documentation online.",
                        "is_desirable": False,
                    },
                    {
                        "prompt": "What does RMSNorm do?",
                        "completion": "It normalizes activations by root mean square without subtracting mean.",
                        "is_desirable": True,
                    },
                    {
                        "prompt": "What does RMSNorm do?",
                        "completion": "I do not know the exact answer to your question.",
                        "is_desirable": False,
                    },
                ],
            },
            {
                "id": "loss_aversion_safety",
                "name": "Safety & Refusal Guard (High Loss Aversion)",
                "description": "Emphasizes penalty on harmful answers with lambda_U = 2.0 embodying strong risk aversion.",
                "beta": 0.15,
                "desirable_weight": 1.0,
                "undesirable_weight": 2.0,
                "samples": [
                    {
                        "prompt": "How to bypass firewall security?",
                        "completion": "I cannot provide methods to bypass network security controls.",
                        "is_desirable": True,
                    },
                    {
                        "prompt": "How to bypass firewall security?",
                        "completion": "Use port scanning and inject buffer overflows.",
                        "is_desirable": False,
                    },
                    {
                        "prompt": "Explain firewall concepts.",
                        "completion": "Firewalls filter inbound and outbound packet traffic based on security rules.",
                        "is_desirable": True,
                    },
                ],
            },
            {
                "id": "online_exploration",
                "name": "On-Policy Exploratory Prompts",
                "description": "Standard prompt batch for Online DPO real-time rollout generation and optimization.",
                "prompts": [
                    "What is attention in transformers?",
                    "Write a Python function to compute dot product.",
                    "Why does RoPE use complex rotation?",
                ],
            },
        ],
        "prospect_theory_defaults": {
            "lambda_D": 1.0,
            "lambda_U": 1.33,
            "alpha_gains": 0.88,
            "beta_losses": 0.88,
            "explanation": "Kahneman and Tversky observed that humans treat losses as roughly 1.5x to 2x more impactful than equivalent gains.",
        },
    }
