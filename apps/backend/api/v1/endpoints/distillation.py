"""
Libra API v1 - Knowledge Distillation & Model Shrinking Endpoints
Supports teacher-student logit transfer, temperature scaling, layer dropping, and dark knowledge analysis.
"""

from __future__ import annotations

from typing import Any

import torch
from fastapi import APIRouter
from pydantic import BaseModel, Field

from packages.evaluation.distillation_eval import DistillationEvaluator
from packages.models.components.model_shrinking import ModelShrinker
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.training.distillation_trainer import (
    DistillationConfig,
    DistillationTelemetry,
    DistillationTrainer,
)

router = APIRouter(prefix="/distillation", tags=["Knowledge Distillation"])


class DistillationTrainRequest(BaseModel):
    temperature: float = Field(default=2.0, ge=0.5, le=20.0, description="Softmax temperature tau")
    alpha: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Weight for soft loss vs hard loss"
    )
    lr: float = Field(default=1e-3, ge=1e-5, le=0.1, description="Student learning rate")
    steps: int = Field(default=15, ge=1, le=50, description="Number of CPU training steps")
    hidden_loss_weight: float = Field(
        default=0.0, ge=0.0, le=5.0, description="Intermediate layer alignment loss weight"
    )
    custom_text: str | None = Field(default=None, description="Optional custom training text")


class DistillationTrainResponse(BaseModel):
    telemetry: list[DistillationTelemetry]
    compression_stats: dict[str, Any]
    initial_total_loss: float
    final_total_loss: float
    initial_agreement: float
    final_agreement: float
    agreement_improvement_pct: float


class SoftLabelsRequest(BaseModel):
    prompt: str = Field(default="The capital of France is Paris", description="Input prompt text")
    temperatures: list[float] = Field(
        default=[1.0, 2.0, 5.0, 10.0],
        description="Temperatures to inspect",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top tokens to inspect")


class SoftLabelsResponse(BaseModel):
    prompt: str
    vocab_size: int
    temperature_analysis: list[dict[str, Any]]


class EvaluateDistillationRequest(BaseModel):
    prompts: list[str] | None = Field(
        default=None,
        description="List of validation prompts to evaluate",
    )
    runs: int = Field(default=3, ge=1, le=10, description="Benchmark timing iterations")


class EvaluateDistillationResponse(BaseModel):
    benchmark: dict[str, Any]
    compression_stats: dict[str, Any]
    sample_predictions: list[dict[str, Any]]


# Cached models for instant educational inference
_teacher_model: ModernTransformerLM | None = None
_student_model: ModernTransformerLM | None = None
_cached_vocab: list[str] = [chr(i) for i in range(256)]


def _get_or_create_models() -> tuple[ModernTransformerLM, ModernTransformerLM]:
    """Provide initialized teacher and student models."""
    global _teacher_model, _student_model
    if _teacher_model is None or _student_model is None:
        torch.manual_seed(42)
        # Teacher: 4 layers, 64 hidden dim
        teacher_cfg = ModernTransformerConfig(
            vocab_size=256,
            d_model=64,
            n_layers=4,
            n_heads=4,
            max_context_length=128,
            hidden_dim=128,
        )
        _teacher_model = ModernTransformerLM(teacher_cfg)

        # Pre-train teacher slightly with synthetic pattern so it has meaningful logits
        toy_corpus = "Knowledge distillation transfers dark knowledge from a large teacher to a compact student network."
        toy_bytes = torch.tensor([[ord(c) % 256 for c in toy_corpus]], dtype=torch.long)
        opt = torch.optim.AdamW(_teacher_model.parameters(), lr=0.01)
        for _ in range(20):
            logits, loss = _teacher_model(toy_bytes[:, :-1], targets=toy_bytes[:, 1:])
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Student: 2 layers, initialized via layer dropping
        _student_model = ModelShrinker.shrink_layers(
            _teacher_model,
            target_layer_indices=[0, 2],
        )

    return _teacher_model, _student_model


def _encode_text(text: str) -> torch.Tensor:
    """Byte-encode a text prompt into a 2D tensor (1, T)."""
    clean = text.encode("utf-8")
    return torch.tensor([[b % 256 for b in clean]], dtype=torch.long)


@router.post("/train", response_model=DistillationTrainResponse)
async def train_distillation(req: DistillationTrainRequest) -> DistillationTrainResponse:
    """Execute a lightweight educational distillation loop on CPU."""
    teacher, student = _get_or_create_models()

    train_text = (
        req.custom_text
        if req.custom_text
        else "Knowledge distillation transfers dark knowledge from a large teacher to a compact student model."
    )
    tokens = _encode_text(train_text)
    if tokens.shape[1] < 2:
        tokens = _encode_text("distillation transfer teacher student")

    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    config = DistillationConfig(
        temperature=req.temperature,
        alpha=req.alpha,
        lr=req.lr,
        hidden_loss_weight=req.hidden_loss_weight,
    )
    trainer = DistillationTrainer(student_model=student, teacher_model=teacher, config=config)

    telemetry = trainer.train_sequence(inputs, targets, steps=req.steps)
    compression = ModelShrinker.compute_compression_stats(teacher, student)

    initial_loss = telemetry[0].total_loss if telemetry else 0.0
    final_loss = telemetry[-1].total_loss if telemetry else 0.0
    initial_agree = telemetry[0].top1_agreement if telemetry else 0.0
    final_agree = telemetry[-1].top1_agreement if telemetry else 0.0

    improvement = (
        round(((final_agree - initial_agree) / max(1e-4, initial_agree)) * 100.0, 2)
        if initial_agree > 0
        else 0.0
    )

    return DistillationTrainResponse(
        telemetry=telemetry,
        compression_stats=compression,
        initial_total_loss=initial_loss,
        final_total_loss=final_loss,
        initial_agreement=initial_agree,
        final_agreement=final_agree,
        agreement_improvement_pct=improvement,
    )


@router.post("/soft_labels", response_model=SoftLabelsResponse)
async def inspect_soft_labels(req: SoftLabelsRequest) -> SoftLabelsResponse:
    """Analyze temperature softening and dark knowledge revelation across temperatures."""
    teacher, student = _get_or_create_models()
    input_ids = _encode_text(req.prompt)

    analysis = DistillationEvaluator.analyze_dark_knowledge(
        teacher_model=teacher,
        student_model=student,
        input_ids=input_ids,
        temperatures=req.temperatures,
        top_k=req.top_k,
        vocab_tokens=_cached_vocab,
    )

    return SoftLabelsResponse(
        prompt=req.prompt,
        vocab_size=analysis["vocab_size"],
        temperature_analysis=analysis["temperature_analysis"],
    )


@router.post("/evaluate", response_model=EvaluateDistillationResponse)
async def evaluate_distillation(req: EvaluateDistillationRequest) -> EvaluateDistillationResponse:
    """Benchmark inference latency, CPU throughput, and prediction agreement."""
    teacher, student = _get_or_create_models()

    prompts = req.prompts or [
        "Language models compress intelligence into weights.",
        "Deep learning optimizes continuous loss surfaces.",
        "Knowledge transfer retains teacher generalizability.",
        "Libra models demonstrate first principles in PyTorch.",
    ]

    sequences = [_encode_text(p) for p in prompts]
    benchmark = DistillationEvaluator.benchmark_speed_and_agreement(
        teacher_model=teacher,
        student_model=student,
        test_sequences=sequences,
        runs=req.runs,
    )
    compression = ModelShrinker.compute_compression_stats(teacher, student)

    sample_preds = []
    teacher.eval()
    student.eval()
    with torch.no_grad():
        for p in prompts[:4]:
            t_ids = _encode_text(p)
            t_out, _ = teacher(t_ids)
            s_out, _ = student(t_ids)

            t_next_id = int(torch.argmax(t_out[0, -1, :]).item())
            s_next_id = int(torch.argmax(s_out[0, -1, :]).item())

            t_char = repr(chr(t_next_id)) if t_next_id < 256 else str(t_next_id)
            s_char = repr(chr(s_next_id)) if s_next_id < 256 else str(s_next_id)

            sample_preds.append(
                {
                    "prompt": p,
                    "teacher_next_token": t_char,
                    "student_next_token": s_char,
                    "tokens_match": t_next_id == s_next_id,
                }
            )

    return EvaluateDistillationResponse(
        benchmark=benchmark,
        compression_stats=compression,
        sample_predictions=sample_preds,
    )


@router.get("/presets")
async def get_distillation_presets() -> dict[str, Any]:
    """Retrieve pre-configured educational distillation templates."""
    return {
        "presets": [
            {
                "id": "standard_balanced",
                "name": "Standard Distillation (Balanced)",
                "description": "Standard Hinton-style distillation with balanced soft and hard losses (alpha=0.5, tau=2.0).",
                "temperature": 2.0,
                "alpha": 0.5,
                "hidden_loss_weight": 0.0,
                "learning_rate": 0.001,
                "steps": 15,
            },
            {
                "id": "dark_knowledge",
                "name": "High Temperature Dark Knowledge (tau=5.0)",
                "description": "Exaggerates secondary token probabilities to reveal fine-grained inter-class semantics.",
                "temperature": 5.0,
                "alpha": 0.7,
                "hidden_loss_weight": 0.0,
                "learning_rate": 0.001,
                "steps": 20,
            },
            {
                "id": "hard_labels_only",
                "name": "Hard Labels Only Ablation (alpha=0.0)",
                "description": "Ablation study training student on ground-truth targets without teacher guidance.",
                "temperature": 1.0,
                "alpha": 0.0,
                "hidden_loss_weight": 0.0,
                "learning_rate": 0.001,
                "steps": 15,
            },
            {
                "id": "hidden_alignment",
                "name": "Logit + Intermediate Representation Alignment",
                "description": "Combines output logit distillation with intermediate transformer hidden state MSE loss.",
                "temperature": 2.0,
                "alpha": 0.5,
                "hidden_loss_weight": 0.2,
                "learning_rate": 0.001,
                "steps": 15,
            },
        ]
    }
