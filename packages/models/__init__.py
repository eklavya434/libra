"""Libra Models Package."""

from packages.models.catalog import get_default_registry
from packages.models.components.kv_cache import KVCache
from packages.models.config import TinyTransformerConfig
from packages.models.generation import generate, generate_with_cache
from packages.models.lora import (
    LoRALinear,
    apply_lora,
    get_lora_parameter_summary,
    load_lora_adapter,
    merge_lora_weights,
    save_lora_adapter,
    unmerge_lora_weights,
)
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
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
from packages.models.telemetry import (
    CandidateToken,
    SequenceTelemetry,
    TokenTelemetry,
    aggregate_sequence_telemetry,
    analyze_sequence_telemetry,
    astream_generate_with_telemetry,
    compute_entropy_bits,
    compute_step_telemetry,
    compute_surprisal_bits,
    stream_generate_with_telemetry,
)
from packages.models.transformer import TinyTransformerLM

__all__ = [
    "CandidateToken",
    "HardwareClassifier",
    "HardwareTier",
    "KVCache",
    "LicenseType",
    "LoRALinear",
    "ModelMetadata",
    "ModelRegistry",
    "ModernTransformerConfig",
    "ModernTransformerLM",
    "QuantizedLinearINT4",
    "QuantizedLinearINT8",
    "RewardTelemetry",
    "SequenceTelemetry",
    "SpeculativeDecoder",
    "SpeculativeResult",
    "TinyTransformerConfig",
    "TinyTransformerLM",
    "TokenTelemetry",
    "TransformerRewardModel",
    "aggregate_sequence_telemetry",
    "analyze_sequence_telemetry",
    "apply_lora",
    "astream_generate_with_telemetry",
    "audit_quantization_fidelity",
    "compute_entropy_bits",
    "compute_model_memory",
    "compute_step_telemetry",
    "compute_surprisal_bits",
    "generate",
    "generate_with_cache",
    "get_default_registry",
    "get_lora_parameter_summary",
    "load_lora_adapter",
    "merge_lora_weights",
    "quantize_model",
    "save_lora_adapter",
    "standard_autoregressive_generate",
    "stream_generate_with_telemetry",
    "unmerge_lora_weights",
]
