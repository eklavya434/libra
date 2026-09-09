"""
Libra Models - Model Registry & Hardware Compatibility Engine

Provides:
  1. Standardized model metadata schema.
  2. Automated hardware tier classification (CPU-friendly, GPU-required, large, very-large).
  3. Dynamic compatibility check against active machine hardware.
  4. Queryable model registry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

from packages.core.hardware import HardwareReport, detect_hardware


class HardwareTier(str, Enum):
    CPU_FRIENDLY = "CPU-friendly"
    GPU_REQUIRED = "GPU-required"
    LARGE = "large"
    VERY_LARGE = "very-large"


class LicenseType(str, Enum):
    PERMISSIVE_OPEN_SOURCE = "permissive_open_source"  # MIT, Apache 2.0, BSD
    OPEN_WEIGHTS_COMMUNITY = "open_weights_community"  # Llama Community, Gemma Terms
    RESEARCH_ONLY = "research_only"  # Non-commercial research
    PROPRIETARY = "proprietary"  # Closed commercial APIs (OpenAI, Gemini)


@dataclass
class ModelMetadata:
    """Canonical metadata for an AI model."""

    model_id: str
    name: str
    organization: str
    provider: str  # "ollama", "huggingface", "libra_lab", "openai", "gemini", etc.
    architecture: str
    parameter_count: int
    context_length: int
    license: str
    license_type: LicenseType
    quantization: str = "FP16"  # e.g. "Q4_K_M", "Q8_0", "FP16", "BF16"
    recommended_ram_gb: float = 0.0
    recommended_vram_gb: float = 0.0
    capabilities: list[str] = field(default_factory=lambda: ["chat"])
    hardware_tier: HardwareTier = HardwareTier.CPU_FRIENDLY
    can_run_locally: bool = True
    hardware_verdict: str = "Compatible"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["id"] = self.model_id
        d["hardware_tier"] = self.hardware_tier.value
        d["license_type"] = self.license_type.value
        d["is_local"] = self.provider in ("ollama", "libra_lab", "huggingface", "mock-provider")
        d["requires_gpu"] = self.hardware_tier in (
            HardwareTier.GPU_REQUIRED,
            HardwareTier.VERY_LARGE,
        )
        return d


class HardwareClassifier:
    """Calculates memory requirements and assigns hardware tiers."""

    @staticmethod
    def estimate_ram_gb(param_count: int, quantization: str) -> float:
        """Estimate minimum system RAM needed to load and run model inference."""
        quant = quantization.upper()
        if "Q4" in quant or "4BIT" in quant:
            bytes_per_param = 0.55  # ~4.5 bits including quantization scale tables
        elif "Q8" in quant or "8BIT" in quant:
            bytes_per_param = 1.05
        elif "FP16" in quant or "BF16" in quant:
            bytes_per_param = 2.0
        elif "FP32" in quant:
            bytes_per_param = 4.0
        else:
            bytes_per_param = 0.6  # Default to ~4-bit quantized estimate

        # Base model weight footprint in GB (1 GB = 10^9 bytes for marketing / 1024^3 for binary)
        weight_gb = (param_count * bytes_per_param) / (1024**3)
        # Add 25% safety overhead for KV cache, context tensors, and runtime buffers
        total_ram_needed = weight_gb * 1.25
        return max(0.1, round(total_ram_needed, 2))

    @classmethod
    def classify(
        cls,
        param_count: int,
        quantization: str,
        system_info: Optional[HardwareReport] = None,
    ) -> tuple[HardwareTier, float, bool, str]:
        """Classifies hardware tier and checks compatibility against system specs."""
        if system_info is None:
            system_info = detect_hardware()

        ram_needed = cls.estimate_ram_gb(param_count, quantization)
        available_ram = system_info.ram_total_gb
        has_cuda = system_info.has_cuda

        # 1. Classify Tier
        if param_count > 50_000_000_000:  # > 50B (e.g. 70B, 405B, 671B)
            tier = HardwareTier.VERY_LARGE
        elif param_count > 10_000_000_000:  # 10B - 50B (e.g. 14B, 32B)
            tier = HardwareTier.LARGE
        elif ram_needed <= 8.0 and ("Q4" in quantization.upper() or param_count <= 3_500_000_000):
            tier = HardwareTier.CPU_FRIENDLY
        else:
            tier = HardwareTier.GPU_REQUIRED

        # 2. Local Execution Verdict
        if param_count < 10_000_000:  # Tiny educational models
            can_run = True
            verdict = "Runs instantly on CPU (< 50MB RAM)"
        elif ram_needed <= (available_ram - 3.0):  # Fits comfortably in RAM leaving 3GB for OS
            can_run = True
            verdict = f"Runs smoothly on CPU (Requires ~{ram_needed:.1f} GB of {available_ram:.0f} GB RAM)"
        elif ram_needed <= available_ram:
            can_run = True
            verdict = f"Tight fit on CPU (Requires ~{ram_needed:.1f} GB; close to {available_ram:.0f} GB limit)"
        else:
            can_run = False
            verdict = f"Exceeds system RAM (Requires ~{ram_needed:.1f} GB; system has {available_ram:.0f} GB)"

        # If it specifically needs GPU and system has no CUDA
        if tier == HardwareTier.GPU_REQUIRED and not has_cuda and not can_run:
            verdict = f"GPU Recommended (Requires CUDA with ~{ram_needed:.1f} GB VRAM)"

        return tier, ram_needed, can_run, verdict


class ModelRegistry:
    """In-memory model catalog and governance manager."""

    def __init__(self, system_info: Optional[HardwareReport] = None) -> None:
        self.system_info = system_info or detect_hardware()
        self._models: dict[str, ModelMetadata] = {}

    def register(self, metadata: ModelMetadata) -> ModelMetadata:
        """Register a model, automatically updating hardware classification if needed."""
        is_cloud = metadata.provider in (
            "gemini",
            "openai",
            "anthropic",
            "groq",
            "deepseek",
            "openrouter",
        )

        if is_cloud:
            metadata.hardware_tier = HardwareTier.CPU_FRIENDLY
            metadata.recommended_ram_gb = 0.0
            metadata.can_run_locally = True
            metadata.hardware_verdict = "Cloud API (Zero Local RAM)"
        else:
            tier, ram_needed, can_run, verdict = HardwareClassifier.classify(
                param_count=metadata.parameter_count,
                quantization=metadata.quantization,
                system_info=self.system_info,
            )
            metadata.hardware_tier = tier
            if metadata.recommended_ram_gb <= 0.0:
                metadata.recommended_ram_gb = ram_needed
            metadata.can_run_locally = can_run
            metadata.hardware_verdict = verdict

        self._models[metadata.model_id] = metadata
        return metadata

    def get(self, model_id: str) -> Optional[ModelMetadata]:
        """Retrieve model metadata by canonical ID."""
        return self._models.get(model_id)

    def list_models(
        self,
        provider: Optional[str] = None,
        tier: Optional[HardwareTier] = None,
        cpu_friendly_only: bool = False,
    ) -> list[ModelMetadata]:
        """List models with optional filters."""
        results = list(self._models.values())
        if provider:
            results = [m for m in results if m.provider.lower() == provider.lower()]
        if tier:
            results = [m for m in results if m.hardware_tier == tier]
        if cpu_friendly_only:
            results = [
                m
                for m in results
                if m.hardware_tier == HardwareTier.CPU_FRIENDLY and m.can_run_locally
            ]
        return results

    def get_cpu_friendly_models(self) -> list[ModelMetadata]:
        """Convenience method returning all models verified to run smoothly on the user CPU."""
        return self.list_models(cpu_friendly_only=True)

    def count(self) -> int:
        return len(self._models)
