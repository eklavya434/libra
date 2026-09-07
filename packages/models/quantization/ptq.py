"""
Libra Models - Post-Training Quantization (PTQ) Engine
Phase 24: Quantization (INT8 / INT4 & Post-Training Quantization)

Provides:
1. quantize_model: Traverses a PyTorch transformer model and replaces Linear layers
   with QuantizedLinearINT8 or QuantizedLinearINT4.
2. compute_model_memory: Calculates precise memory footprint of all parameters/buffers.
3. audit_quantization_fidelity: Evaluates logits MSE and SQNR between FP32 and quantized models.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from packages.models.quantization.quant_core import compute_quantization_metrics
from packages.models.quantization.quant_linear import (
    QuantizedLinearINT4,
    QuantizedLinearINT8,
)


def compute_model_memory(model: nn.Module) -> dict[str, int | float]:
    """Calculates model parameter and buffer memory footprint in bytes and megabytes."""
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())

    # Add custom layer memory if quantized layers are present
    custom_bytes = 0
    for m in model.modules():
        if isinstance(m, (QuantizedLinearINT8, QuantizedLinearINT4)):
            # Buffers are already counted, but track if any uncounted custom fields exist
            pass

    total_bytes = param_bytes + buffer_bytes + custom_bytes
    return {
        "param_bytes": param_bytes,
        "buffer_bytes": buffer_bytes,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 3),
    }


def quantize_model(
    model: nn.Module,
    mode: str = "int8",
    per_channel: bool = True,
    inplace: bool = False,
    exclude_modules: list[str] | None = None,
) -> nn.Module:
    """
    Applies Post-Training Quantization to a PyTorch model by recursively replacing
    `torch.nn.Linear` layers with `QuantizedLinearINT8` or `QuantizedLinearINT4`.

    Args:
        model: PyTorch model (e.g. ModernTransformerLM or TinyTransformerLM).
        mode: Quantization mode ("int8" or "int4").
        per_channel: Whether to compute scales per output channel (True) or per tensor (False).
        inplace: If False, operates on a deepcopy of the model.
        exclude_modules: List of module names to keep in full precision (e.g. ["output_head"]).

    Returns:
        Quantized model.
    """
    if mode not in ("int8", "int4"):
        raise ValueError(f"Unsupported quantization mode '{mode}'. Choose 'int8' or 'int4'.")

    target_model = model if inplace else copy.deepcopy(model)
    exclude = set(exclude_modules or [])

    def _replace_layers(module: nn.Module, prefix: str = "") -> None:
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if full_name in exclude or name in exclude:
                continue

            if isinstance(child, nn.Linear):
                # Check for INT4 divisibility constraint
                if mode == "int4" and child.in_features % 2 != 0:
                    continue  # Keep as float if in_features is odd

                if mode == "int8":
                    qlinear = QuantizedLinearINT8.from_float(child, per_channel=per_channel)
                else:
                    qlinear = QuantizedLinearINT4.from_float(child, per_channel=per_channel)

                setattr(module, name, qlinear)
            else:
                _replace_layers(child, full_name)

    _replace_layers(target_model)
    return target_model


def audit_quantization_fidelity(
    float_model: nn.Module,
    quant_model: nn.Module,
    sample_input: torch.Tensor,
) -> dict[str, Any]:
    """
    Compares logits output of floating point model vs quantized model on a sample batch.
    Computes logits MSE, SQNR (dB), cosine similarity, and top-1 agreement.
    """
    float_model.eval()
    quant_model.eval()

    with torch.no_grad():
        float_out, _ = float_model(sample_input)
        quant_out, _ = quant_model(sample_input)

    metrics = compute_quantization_metrics(float_out, quant_out)

    # Top-1 token agreement rate
    float_preds = torch.argmax(float_out, dim=-1)
    quant_preds = torch.argmax(quant_out, dim=-1)
    agreement = (float_preds == quant_preds).float().mean().item()

    mem_float = compute_model_memory(float_model)
    mem_quant = compute_model_memory(quant_model)

    compression_ratio = round(mem_float["total_bytes"] / max(mem_quant["total_bytes"], 1), 2)
    memory_savings_pct = round(
        (1.0 - (mem_quant["total_bytes"] / max(mem_float["total_bytes"], 1))) * 100, 2
    )

    return {
        "quantization_mode": getattr(quant_model, "quant_mode", "unknown"),
        "fidelity": {
            "mse": metrics["mse"],
            "sqnr_db": metrics["sqnr_db"],
            "cosine_similarity": metrics["cosine_similarity"],
            "top1_agreement_pct": round(agreement * 100, 2),
        },
        "memory": {
            "fp32_bytes": mem_float["total_bytes"],
            "quantized_bytes": mem_quant["total_bytes"],
            "compression_ratio": f"{compression_ratio}x",
            "savings_pct": f"{memory_savings_pct}%",
        },
    }
