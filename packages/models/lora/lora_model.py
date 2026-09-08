"""
Libra Models - Model-Level PEFT and Adapter Management
Phase 25: Parameter-Efficient Fine-Tuning (PEFT & LoRA)

Provides:
1. apply_lora: Injects LoRALinear layers into target projection modules of a transformer.
2. merge_lora_weights: Folds all LoRA weights back into the base model.
3. unmerge_lora_weights: Unmerges all LoRA weights to resume training.
4. save_lora_adapter: Serializes only trainable LoRA tensors and config metadata.
5. load_lora_adapter: Loads LoRA adapter weights from checkpoint.
6. get_lora_parameter_summary: Audits trainable vs frozen parameter counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from packages.models.lora.lora_linear import LoRALinear


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: list[str] | None = None,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Applies LoRA to a transformer model by freezing all base weights and replacing
    targeted Linear layers with LoRALinear modules.

    Default target_modules for attention and MLP:
    ['q_proj', 'v_proj', 'k_proj', 'out_proj']
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    target_set = set(target_modules)

    # 1. Freeze all base parameters
    for param in model.parameters():
        param.requires_grad = False

    # 2. Recursively replace target nn.Linear layers
    def _replace_layers(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in target_set:
                lora_layer = LoRALinear(
                    base_linear=child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(module, name, lora_layer)
            else:
                _replace_layers(child)

    _replace_layers(model)
    return model


def merge_lora_weights(model: nn.Module) -> None:
    """Merges all LoRALinear layers in the model for zero-overhead inference."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()


def unmerge_lora_weights(model: nn.Module) -> None:
    """Unmerges all LoRALinear layers in the model to continue training."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge_weights()


def get_lora_parameter_summary(model: nn.Module) -> dict[str, Any]:
    """Computes total, trainable, and frozen parameter counts and percentages."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    trainable_pct = (trainable_params / max(total_params, 1)) * 100.0

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "trainable_percentage": round(trainable_pct, 2),
    }


def save_lora_adapter(
    model: nn.Module,
    save_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Saves only the lightweight LoRA adapter tensors (lora_A, lora_B) and config metadata.
    Avoids saving gigabytes of frozen base model parameters.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    adapter_state: dict[str, torch.Tensor] = {}
    adapter_configs: dict[str, dict[str, Any]] = {}

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and module.rank > 0:
            adapter_state[f"{name}.lora_A"] = module.lora_A.data.cpu()
            adapter_state[f"{name}.lora_B"] = module.lora_B.data.cpu()
            adapter_configs[name] = {
                "rank": module.rank,
                "alpha": module.alpha,
                "scaling": module.scaling,
            }

    payload = {
        "adapter_state": adapter_state,
        "adapter_configs": adapter_configs,
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    return path


def load_lora_adapter(
    model: nn.Module,
    load_path: str | Path,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Loads saved LoRA adapter tensors into matching LoRALinear modules.
    """
    path = Path(load_path)
    if not path.exists():
        raise FileNotFoundError(f"LoRA adapter not found at {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    adapter_state = payload["adapter_state"]

    loaded_keys = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            key_A = f"{name}.lora_A"
            key_B = f"{name}.lora_B"
            if key_A in adapter_state and key_B in adapter_state:
                module.lora_A.data.copy_(adapter_state[key_A])
                module.lora_B.data.copy_(adapter_state[key_B])
                loaded_keys.extend([key_A, key_B])
            elif strict:
                raise KeyError(f"Missing weights for LoRA module: {name}")

    return {
        "loaded_keys_count": len(loaded_keys),
        "metadata": payload.get("metadata", {}),
    }
