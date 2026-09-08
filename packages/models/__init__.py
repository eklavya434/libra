"""Libra Models Package."""

from packages.models.catalog import get_default_registry
from packages.models.components.kv_cache import KVCache
from packages.models.config import TinyTransformerConfig
from packages.models.generation import generate, generate_with_cache
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.models.lora import (
    LoRALinear,
    apply_lora,
    get_lora_parameter_summary,
    load_lora_adapter,
    merge_lora_weights,
    save_lora_adapter,
    unmerge_lora_weights,
)
from packages.models.quantization import (
    QuantizedLinearINT4,
    QuantizedLinearINT8,
    audit_quantization_fidelity,
    compute_model_memory,
    quantize_model,
)
from packages.models.registry import (
    HardwareClassifier,
    HardwareTier,
    LicenseType,
    ModelMetadata,
    ModelRegistry,
)
from packages.models.reward_model import RewardTelemetry, TransformerRewardModel
from packages.models.speculative import (
    SpeculativeDecoder,
    SpeculativeResult,
    standard_autoregressive_generate,
)
from packages.models.transformer import TinyTransformerLM

__all__ = [
    "TinyTransformerConfig",
    "TinyTransformerLM",
    "ModernTransformerConfig",
    "ModernTransformerLM",
    "generate",
    "generate_with_cache",
    "KVCache",
    "SpeculativeDecoder",
    "SpeculativeResult",
    "standard_autoregressive_generate",
    "TransformerRewardModel",
    "RewardTelemetry",
    "ModelMetadata",
    "HardwareTier",
    "LicenseType",
    "HardwareClassifier",
    "ModelRegistry",
    "get_default_registry",
    "quantize_model",
    "compute_model_memory",
    "audit_quantization_fidelity",
    "QuantizedLinearINT8",
    "QuantizedLinearINT4",
    "LoRALinear",
    "apply_lora",
    "merge_lora_weights",
    "unmerge_lora_weights",
    "get_lora_parameter_summary",
    "save_lora_adapter",
    "load_lora_adapter",
]
