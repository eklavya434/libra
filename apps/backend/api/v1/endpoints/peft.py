"""
Libra API v1 - PEFT & LoRA Endpoints
Phase 25: Parameter-Efficient Fine-Tuning (PEFT & LoRA)
"""

import time

import torch
from fastapi import APIRouter
from pydantic import BaseModel, Field

from packages.models.lora import (
    apply_lora,
    get_lora_parameter_summary,
    merge_lora_weights,
    unmerge_lora_weights,
)
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.training.lora_trainer import LoRATrainer

router = APIRouter(prefix="/peft", tags=["PEFT & LoRA"])


class LoRAApplyRequest(BaseModel):
    vocab_size: int = Field(default=100, ge=10, le=50000)
    d_model: int = Field(default=64, ge=16, le=1024)
    n_heads: int = Field(default=4, ge=1, le=32)
    n_layers: int = Field(default=2, ge=1, le=16)
    rank: int = Field(default=4, ge=1, le=64)
    alpha: float = Field(default=8.0, ge=1.0, le=128.0)
    target_modules: list[str] = Field(default=["q_proj", "v_proj"])


class LoRATrainStepRequest(BaseModel):
    vocab_size: int = Field(default=100, ge=10, le=50000)
    d_model: int = Field(default=64, ge=16, le=1024)
    n_heads: int = Field(default=4, ge=1, le=32)
    n_layers: int = Field(default=2, ge=1, le=16)
    rank: int = Field(default=4, ge=1, le=64)
    alpha: float = Field(default=8.0, ge=1.0, le=128.0)
    learning_rate: float = Field(default=1e-3, ge=1e-6, le=1.0)
    seq_len: int = Field(default=8, ge=2, le=64)


class LoRAMergeRequest(BaseModel):
    vocab_size: int = Field(default=100, ge=10, le=50000)
    d_model: int = Field(default=64, ge=16, le=1024)
    n_heads: int = Field(default=4, ge=1, le=32)
    n_layers: int = Field(default=2, ge=1, le=16)
    rank: int = Field(default=4, ge=1, le=64)
    alpha: float = Field(default=8.0, ge=1.0, le=128.0)


@router.post("/apply")
def apply_lora_endpoint(request: LoRAApplyRequest):
    """
    Applies LoRA to a model configuration and returns parameter audit.
    Shows reduction in trainable parameters (<5% typical).
    """
    cfg = ModernTransformerConfig(
        vocab_size=request.vocab_size,
        d_model=request.d_model,
        n_heads=request.n_heads,
        n_layers=request.n_layers,
    )
    base_model = ModernTransformerLM(cfg)
    base_params = sum(p.numel() for p in base_model.parameters())

    lora_model = apply_lora(
        base_model,
        rank=request.rank,
        alpha=request.alpha,
        target_modules=request.target_modules,
    )

    summary = get_lora_parameter_summary(lora_model)

    return {
        "status": "success",
        "rank": request.rank,
        "alpha": request.alpha,
        "scaling_factor": round(request.alpha / request.rank, 4),
        "target_modules": request.target_modules,
        "base_parameters": base_params,
        "parameter_summary": summary,
        "parameter_savings_ratio": f"{round(base_params / max(summary['trainable_parameters'], 1), 1)}x fewer trainable parameters",
    }


@router.post("/train_step")
def train_step_endpoint(request: LoRATrainStepRequest):
    """
    Executes a single LoRA fine-tuning step on synthetic data and returns loss and gradient norm.
    """
    cfg = ModernTransformerConfig(
        vocab_size=request.vocab_size,
        d_model=request.d_model,
        n_heads=request.n_heads,
        n_layers=request.n_layers,
    )
    model = ModernTransformerLM(cfg)
    apply_lora(model, rank=request.rank, alpha=request.alpha)

    trainer = LoRATrainer(model, learning_rate=request.learning_rate)

    input_ids = torch.randint(0, request.vocab_size, (2, request.seq_len))
    target_ids = torch.randint(0, request.vocab_size, (2, request.seq_len))

    t0 = time.perf_counter()
    metrics = trainer.train_step(input_ids, target_ids)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "status": "success",
        "step_metrics": metrics,
        "elapsed_ms": round(elapsed_ms, 2),
    }


@router.post("/merge")
def merge_endpoint(request: LoRAMergeRequest):
    """
    Demonstrates zero-latency weight merging and mathematical equivalence.
    """
    cfg = ModernTransformerConfig(
        vocab_size=request.vocab_size,
        d_model=request.d_model,
        n_heads=request.n_heads,
        n_layers=request.n_layers,
    )
    model = ModernTransformerLM(cfg)
    apply_lora(model, rank=request.rank, alpha=request.alpha)

    sample = torch.randint(0, request.vocab_size, (1, 8))

    model.eval()
    with torch.no_grad():
        out_unmerged, _ = model(sample)

    # Merge weights into base
    merge_lora_weights(model)
    with torch.no_grad():
        out_merged, _ = model(sample)

    diff = (out_unmerged - out_merged).abs().max().item()

    # Unmerge weights
    unmerge_lora_weights(model)
    with torch.no_grad():
        _out_restored, _ = model(sample)

    return {
        "status": "success",
        "rank": request.rank,
        "alpha": request.alpha,
        "max_discrepancy_after_merge": diff,
        "is_numerically_identical": diff < 1e-5,
    }
