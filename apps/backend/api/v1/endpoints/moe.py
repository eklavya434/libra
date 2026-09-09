"""
Libra API v1 - Mixture of Experts (MoE) Routing Endpoints
Supports sparse expert gating, token-by-token routing analysis, load balancing, and expert utilization.
"""

from __future__ import annotations

from typing import Any

import torch
from fastapi import APIRouter
from pydantic import BaseModel, Field

from packages.evaluation.moe_eval import MoEEvaluator
from packages.models.moe_transformer import MoETransformerConfig, MoETransformerLM

router = APIRouter(prefix="/moe", tags=["Mixture of Experts (MoE)"])


class MoEForwardRequest(BaseModel):
    prompt: str = Field(
        default="Mixture of experts dynamically routes tokens to specialized feed-forward sub-networks.",
        description="Text prompt to route across experts",
    )


class MoEForwardResponse(BaseModel):
    prompt: str
    prompt_tokens_count: int
    num_experts: int
    top_k: int
    aux_loss: float
    tokens: list[dict[str, Any]]
    layer_stats: list[dict[str, Any]]
    parameter_efficiency: dict[str, Any]


class MoETrainRequest(BaseModel):
    steps: int = Field(default=10, ge=1, le=30, description="Training iterations")
    aux_loss_coef: float = Field(
        default=0.02, ge=0.0, le=0.2, description="Auxiliary balance loss coefficient"
    )
    lr: float = Field(default=0.005, ge=1e-4, le=0.05, description="Learning rate")
    custom_text: str | None = Field(default=None, description="Optional custom training text")


class MoETrainResponse(BaseModel):
    history: list[dict[str, Any]]
    initial_total_loss: float
    final_total_loss: float
    initial_cv: float
    final_cv: float
    balance_improved: bool
    parameter_efficiency: dict[str, Any]


# Cached educational MoE model
_moe_model: MoETransformerLM | None = None
_cached_vocab: list[str] = [chr(i) for i in range(256)]


def _get_or_create_moe_model() -> MoETransformerLM:
    """Retrieve or initialize the cached educational MoE model."""
    global _moe_model
    if _moe_model is None:
        torch.manual_seed(42)
        cfg = MoETransformerConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            max_context_length=128,
            hidden_dim=128,
            num_experts=4,
            num_experts_per_tok=2,
            aux_loss_coef=0.01,
            noisy_gating=True,
        )
        _moe_model = MoETransformerLM(cfg)

        # Pre-train slightly on educational text so experts begin specializing
        corpus = "Mixture of experts routes tokens conditionally to specialized feed-forward expert layers."
        tokens = torch.tensor([[ord(c) % 256 for c in corpus]], dtype=torch.long)
        opt = torch.optim.AdamW(_moe_model.parameters(), lr=0.01)
        for _ in range(15):
            _, loss, _, _, _ = _moe_model(tokens[:, :-1], targets=tokens[:, 1:])
            if loss is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()

    return _moe_model


def _encode_text(text: str) -> torch.Tensor:
    """Byte-encode text into tensor of shape (1, T)."""
    clean = text.encode("utf-8")
    return torch.tensor([[b % 256 for b in clean]], dtype=torch.long)


@router.post("/forward", response_model=MoEForwardResponse)
async def forward_moe(req: MoEForwardRequest) -> MoEForwardResponse:
    """Execute forward pass and trace token-to-expert routing assignments."""
    model = _get_or_create_moe_model()
    tokens = _encode_text(req.prompt)

    analysis = MoEEvaluator.analyze_token_routing(
        model=model,
        input_ids=tokens,
        vocab_tokens=_cached_vocab,
    )

    param_stats = model.count_parameters()

    return MoEForwardResponse(
        prompt=req.prompt,
        prompt_tokens_count=analysis["prompt_tokens_count"],
        num_experts=analysis["num_experts"],
        top_k=analysis["top_k"],
        aux_loss=analysis["aux_loss"],
        tokens=analysis["tokens"],
        layer_stats=analysis["layer_stats"],
        parameter_efficiency=param_stats,
    )


@router.post("/train", response_model=MoETrainResponse)
async def train_moe_step(req: MoETrainRequest) -> MoETrainResponse:
    """Run an educational MoE training loop demonstrating load-balancing auxiliary loss."""
    model = _get_or_create_moe_model()
    model.train()
    model.config.aux_loss_coef = req.aux_loss_coef

    text = (
        req.custom_text
        if req.custom_text
        else "Sparse mixture of experts scales parameter capacity while keeping per-token inference FLOPs constant."
    )
    tokens = _encode_text(text)
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=req.lr)

    history = []
    for step in range(1, req.steps + 1):
        optimizer.zero_grad()
        logits, total_loss, task_loss, aux_loss, routing = model(inputs, targets=targets)
        assert total_loss is not None
        assert task_loss is not None
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Compute imbalance CV on first layer
        indices = routing[0]["routing"]["topk_indices"]
        num_exp = model.config.num_experts
        counts = [(indices == e).sum().item() for e in range(num_exp)]
        total_counts = max(1, sum(counts))
        fractions = [c / total_counts for c in counts]
        mean_f = sum(fractions) / num_exp
        std_f = (sum((f - mean_f) ** 2 for f in fractions) / num_exp) ** 0.5
        cv = round(std_f / max(1e-5, mean_f), 4)

        history.append(
            {
                "step": step,
                "total_loss": round(float(total_loss.item()), 4),
                "task_loss": round(float(task_loss.item()), 4),
                "aux_loss": round(float(aux_loss.item()), 5),
                "cv_imbalance": cv,
                "expert_fractions": [round(f, 3) for f in fractions],
            }
        )

    init_loss = history[0]["total_loss"]
    final_loss = history[-1]["total_loss"]
    init_cv = history[0]["cv_imbalance"]
    final_cv = history[-1]["cv_imbalance"]

    return MoETrainResponse(
        history=history,
        initial_total_loss=init_loss,
        final_total_loss=final_loss,
        initial_cv=init_cv,
        final_cv=final_cv,
        balance_improved=final_cv <= init_cv,
        parameter_efficiency=model.count_parameters(),
    )


@router.get("/utilization")
async def get_moe_utilization() -> dict[str, Any]:
    """Inspect expert load balance and starvation across validation benchmarks."""
    model = _get_or_create_moe_model()
    sentences = [
        "Language models predict tokens autoregressively.",
        "Mathematics relies on formal proofs and exact equations.",
        "Code execution requires precise syntax and runtime bounds.",
        "Mixture of experts routes tokens to specialized feed-forward blocks.",
    ]
    sequences = [_encode_text(s) for s in sentences]
    utilization = MoEEvaluator.compute_expert_utilization(model, sequences)
    return utilization


@router.get("/presets")
async def get_moe_presets() -> dict[str, Any]:
    """Retrieve pre-configured educational MoE routing setups."""
    return {
        "presets": [
            {
                "id": "mixtral_top2",
                "name": "Mixtral Style (Top-2 of 4 Experts)",
                "description": "Routes each token to 2 of 4 experts with auxiliary balance loss. 50% active feed-forward compute per token.",
                "num_experts": 4,
                "top_k": 2,
                "aux_loss_coef": 0.02,
                "noisy_gating": True,
            },
            {
                "id": "switch_top1",
                "name": "Switch Transformer (Top-1 of 4 Experts)",
                "description": "Extreme sparsity routing each token to exactly 1 expert. 75% compute savings per token.",
                "num_experts": 4,
                "top_k": 1,
                "aux_loss_coef": 0.05,
                "noisy_gating": True,
            },
            {
                "id": "starvation_ablation",
                "name": "Starvation Ablation (Aux Loss = 0.0)",
                "description": "Disables auxiliary load-balancing loss to observe routing collapse where 1-2 experts dominate.",
                "num_experts": 4,
                "top_k": 2,
                "aux_loss_coef": 0.0,
                "noisy_gating": False,
            },
        ]
    }
