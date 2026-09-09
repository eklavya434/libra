"""
Libra API v1 - Medusa Multi-Head Speculative Decoding Endpoints
Supports parallel speculative drafting, single-pass prefix verification, and multi-token acceleration.
"""

from __future__ import annotations

from typing import Any

import torch
from fastapi import APIRouter
from pydantic import BaseModel, Field

from packages.evaluation.medusa_eval import MedusaEvaluator
from packages.models.components.medusa import MedusaModel
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM

router = APIRouter(prefix="/medusa", tags=["Medusa Speculative Decoding"])


class MedusaGenerateRequest(BaseModel):
    prompt: str = Field(
        default="Speculative decoding accelerates language models by verifying multiple candidates.",
        description="Prompt text to continue with Medusa",
    )
    max_new_tokens: int = Field(default=25, ge=5, le=60, description="Tokens to generate")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")


class MedusaGenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    generated_tokens_count: int
    total_forward_passes: int
    speedup_ratio: float
    avg_accepted_per_step: float
    tokens_per_second: float
    elapsed_seconds: float
    steps: list[dict[str, Any]]


class MedusaBenchmarkRequest(BaseModel):
    prompts: list[str] | None = Field(
        default=None,
        description="List of benchmark prompts",
    )
    max_new_tokens: int = Field(default=15, ge=5, le=30, description="Tokens per prompt")


class MedusaBenchmarkResponse(BaseModel):
    benchmark: dict[str, Any]
    head_accuracies: dict[str, Any]


class MedusaTrainRequest(BaseModel):
    steps: int = Field(default=15, ge=1, le=40, description="Training steps")
    lr: float = Field(default=0.005, ge=1e-4, le=0.05, description="Learning rate for Medusa heads")
    decay: float = Field(default=0.8, ge=0.1, le=1.0, description="Head loss discount factor")
    custom_text: str | None = Field(default=None, description="Optional custom training corpus")


class MedusaTrainResponse(BaseModel):
    history: list[dict[str, Any]]
    initial_loss: float
    final_loss: float
    loss_reduction_pct: float
    num_heads: int


# Cached educational Medusa model
_medusa_model: MedusaModel | None = None
_cached_vocab: list[str] = [chr(i) for i in range(256)]


def _get_or_create_medusa_model() -> MedusaModel:
    """Retrieve or initialize cached MedusaModel."""
    global _medusa_model
    if _medusa_model is None:
        torch.manual_seed(42)
        cfg = ModernTransformerConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            max_context_length=128,
            hidden_dim=128,
        )
        base = ModernTransformerLM(cfg)

        # Pre-train base model slightly on educational corpus
        corpus = "Speculative decoding accelerates language models by verifying multiple candidates in parallel."
        tokens = torch.tensor([[ord(c) % 256 for c in corpus]], dtype=torch.long)
        opt = torch.optim.AdamW(base.parameters(), lr=0.01)
        for _ in range(15):
            _, loss = base(tokens[:, :-1], targets=tokens[:, 1:])
            if loss is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()

        # Wrap in Medusa with 3 speculative heads
        _medusa_model = MedusaModel(base_model=base, num_heads=3)

        # Freeze base model parameters
        for p in _medusa_model.base_model.parameters():
            p.requires_grad = False

        # Train Medusa heads for a few steps so they predict meaningful next tokens
        opt_heads = torch.optim.AdamW(_medusa_model.medusa_heads.parameters(), lr=0.01)
        for _ in range(25):
            opt_heads.zero_grad()
            _, medusa_logits, _ = _medusa_model.forward_with_medusa(tokens)
            loss, _ = _medusa_model.compute_medusa_loss(medusa_logits, targets=tokens, decay=0.8)
            loss.backward()
            opt_heads.step()

    return _medusa_model


def _encode_text(text: str) -> torch.Tensor:
    clean = text.encode("utf-8")
    return torch.tensor([[b % 256 for b in clean]], dtype=torch.long)


def _decode_tokens(tokens: list[int]) -> str:
    chars = [chr(t) for t in tokens if 0 <= t < 256]
    return "".join(chars)


@router.post("/generate", response_model=MedusaGenerateResponse)
async def generate_medusa(req: MedusaGenerateRequest) -> MedusaGenerateResponse:
    """Execute speculative multi-token decoding and emit verification timeline."""
    model = _get_or_create_medusa_model()
    prompt_tensor = _encode_text(req.prompt)

    res = model.medusa_generate(
        prompt_tokens=prompt_tensor,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )

    generated_text = _decode_tokens(res["generated_tokens"])

    # Decorate step telemetry with readable character strings
    steps_decorated = []
    for s in res["steps"]:
        steps_decorated.append(
            {
                "step": s["step"],
                "accepted_count": s["accepted_count"],
                "drafted_tokens_str": [
                    repr(chr(t)) if t < 256 else str(t) for t in s["drafted_tokens"]
                ],
                "accepted_tokens_str": [
                    repr(chr(t)) if t < 256 else str(t) for t in s["accepted_tokens"]
                ],
                "acceptance_mask": s["acceptance_mask"],
            }
        )

    return MedusaGenerateResponse(
        prompt=req.prompt,
        generated_text=generated_text,
        generated_tokens_count=res["generated_tokens_count"],
        total_forward_passes=res["total_forward_passes"],
        speedup_ratio=res["speedup_ratio"],
        avg_accepted_per_step=res["avg_accepted_per_step"],
        tokens_per_second=res["tokens_per_second"],
        elapsed_seconds=res["elapsed_seconds"],
        steps=steps_decorated,
    )


@router.post("/benchmark", response_model=MedusaBenchmarkResponse)
async def benchmark_medusa(req: MedusaBenchmarkRequest) -> MedusaBenchmarkResponse:
    """Compare standard autoregressive vs Medusa speculative decoding performance."""
    model = _get_or_create_medusa_model()

    prompts = req.prompts or [
        "Speculative decoding accelerates language models.",
        "Parallel verification emits multiple tokens per step.",
        "Medusa heads eliminate secondary draft models.",
    ]
    prompt_tensors = [_encode_text(p) for p in prompts]

    bench = MedusaEvaluator.benchmark_speculative_speedup(
        medusa_model=model,
        test_prompts=prompt_tensors,
        max_new_tokens=req.max_new_tokens,
    )

    accuracies = MedusaEvaluator.analyze_head_accuracies(
        medusa_model=model,
        sequences=prompt_tensors,
    )

    return MedusaBenchmarkResponse(
        benchmark=bench,
        head_accuracies=accuracies,
    )


@router.post("/train", response_model=MedusaTrainResponse)
async def train_medusa_heads(req: MedusaTrainRequest) -> MedusaTrainResponse:
    """Run an educational training loop for Medusa speculative heads on CPU."""
    model = _get_or_create_medusa_model()
    model.train()

    # Freeze base model
    for p in model.base_model.parameters():
        p.requires_grad = False
    for p in model.medusa_heads.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.medusa_heads.parameters(), lr=req.lr)

    text = (
        req.custom_text
        if req.custom_text
        else "Speculative decoding accelerates language models by verifying multiple candidate tokens simultaneously."
    )
    tokens = _encode_text(text)

    history = []
    for step in range(1, req.steps + 1):
        optimizer.zero_grad()
        _, medusa_logits, _ = model.forward_with_medusa(tokens)
        total_loss, head_losses = model.compute_medusa_loss(
            medusa_logits, targets=tokens, decay=req.decay
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.medusa_heads.parameters(), 1.0)
        optimizer.step()

        history.append(
            {
                "step": step,
                "total_loss": round(float(total_loss.item()), 4),
                "per_head_losses": head_losses,
            }
        )

    init_loss = history[0]["total_loss"]
    final_loss = history[-1]["total_loss"]
    reduction = (
        round(((init_loss - final_loss) / max(1e-4, init_loss)) * 100.0, 2)
        if init_loss > 0
        else 0.0
    )

    return MedusaTrainResponse(
        history=history,
        initial_loss=init_loss,
        final_loss=final_loss,
        loss_reduction_pct=reduction,
        num_heads=model.num_heads,
    )


@router.get("/presets")
async def get_medusa_presets() -> dict[str, Any]:
    """Retrieve educational Medusa speculative configurations."""
    return {
        "presets": [
            {
                "id": "medusa_3heads",
                "name": "Medusa 3-Head Drafting (Standard)",
                "description": "3 parallel speculative heads predicting t+2, t+3, t+4 with decay=0.8. Up to 4x peak speedup.",
                "num_heads": 3,
                "decay": 0.8,
                "temperature": 0.0,
            },
            {
                "id": "medusa_2heads",
                "name": "Medusa 2-Head Lightweight",
                "description": "2 speculative heads for minimal CPU parameter overhead with up to 3x peak speedup.",
                "num_heads": 2,
                "decay": 0.9,
                "temperature": 0.0,
            },
            {
                "id": "greedy_verification",
                "name": "Greedy Prefix Verification (temperature=0.0)",
                "description": "Deterministic argmax verification maximizing acceptance rates.",
                "num_heads": 3,
                "decay": 0.8,
                "temperature": 0.0,
            },
        ]
    }
