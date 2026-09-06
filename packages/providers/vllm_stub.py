"""
Libra Providers - vLLM Architectural Specification (Future GPU-Only)

IMPORTANT OPERATIONAL PROTOCOL (Prompt 2 Directive 6):
  "Ollama is the default and primary local-inference path for this machine — it is
   the beginner-friendly option that actually works well on CPU with quantized open-weight models.
   vLLM should be documented as a future option only, requiring a dedicated NVIDIA GPU.
   Do not install or attempt to run vLLM on this machine — its CPU backend is limited
   and not worth the setup cost here. Revisit vLLM only when GPU hardware is added."
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from packages.core.hardware import detect_hardware
from packages.providers.base import BaseProvider, ModelMetadata


class VLLMProvider(BaseProvider):
    """Architectural placeholder and specification for future vLLM GPU inference."""

    def __init__(self, api_url: str = "http://127.0.0.1:8000/v1"):
        self.api_url = api_url

    @property
    def name(self) -> str:
        return "vllm"

    def capabilities(self) -> dict[str, bool]:
        return {
            "supports_text": True,
            "supports_vision": False,
            "supports_tools": True,
            "supports_reasoning": True,
            "supports_embeddings": False,
            "supports_streaming": True,
        }

    async def health(self) -> dict[str, Any]:
        hw = detect_hardware()
        return {
            "status": "disabled",
            "provider": "vllm",
            "has_cuda": hw.has_cuda,
            "hardware_tier": hw.device_tier,
            "reason": (
                "vLLM requires a dedicated NVIDIA GPU with CUDA (PagedAttention optimization). "
                "On this CPU machine (Intel Core i5-12450H), Ollama is the active local runtime. "
                "vLLM will be activated in future phases when dedicated GPU hardware is attached."
            ),
        }

    async def list_models(self) -> list[ModelMetadata]:
        return []

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise RuntimeError(
            "vLLM is disabled on this CPU-only machine. Use OllamaProvider or LocalTransformerProvider instead."
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        raise RuntimeError(
            "vLLM is disabled on this CPU-only machine. Use OllamaProvider or LocalTransformerProvider instead."
        )
        yield ""

    async def embeddings(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        raise NotImplementedError("vLLM embeddings not enabled on CPU.")
