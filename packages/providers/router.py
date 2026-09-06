"""
Libra Providers - Unified Model Provider Router

Routes inference requests across available providers (Ollama, Local Lab, Mock, vLLM, HuggingFace).
Handles graceful fallback and provider discovery.
"""

from __future__ import annotations

from typing import Any, Optional

from packages.providers.base import BaseProvider
from packages.providers.huggingface import HuggingFaceProvider
from packages.providers.local_transformer import LocalTransformerProvider
from packages.providers.mock import MockProvider
from packages.providers.ollama import OllamaProvider
from packages.providers.vllm_stub import VLLMProvider


class ProviderRouter:
    """Orchestrates model provider selection, routing, and fallbacks."""

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {
            "ollama": OllamaProvider(),
            "libra_lab": LocalTransformerProvider(),
            "mock-provider": MockProvider(),
            "vllm": VLLMProvider(),
            "huggingface": HuggingFaceProvider(),
        }

    def get_provider(self, name: str) -> BaseProvider:
        """Retrieve a specific provider by name."""
        name_clean = name.lower()
        if name_clean in self._providers:
            return self._providers[name_clean]
        # Check aliases
        if "hf" in name_clean or "hugging" in name_clean:
            return self._providers["huggingface"]
        if "mock" in name_clean:
            return self._providers["mock-provider"]
        if "lab" in name_clean or "libra" in name_clean:
            return self._providers["libra_lab"]
        raise ValueError(f"Unknown provider '{name}'. Available: {list(self._providers.keys())}")

    async def resolve_provider_for_model(self, model_id: str, requested_provider: Optional[str] = None) -> BaseProvider:
        """Determines the best active provider for a model request."""
        if requested_provider:
            return self.get_provider(requested_provider)

        model_lower = model_id.lower()

        # If it is a Hugging Face model
        if model_lower.startswith("hf/") or "gpt2" in model_lower:
            return self._providers["huggingface"]

        # If it is an educational Libra model
        if "libra" in model_lower and "mock" not in model_lower:
            return self._providers["libra_lab"]

        # If it is a mock model
        if "mock" in model_lower:
            return self._providers["mock-provider"]

        # Check if Ollama is running locally
        ollama_health = await self._providers["ollama"].health()
        if ollama_health.get("connected"):
            return self._providers["ollama"]

        # If Ollama is offline and mock is available, gracefully use mock
        return self._providers["mock-provider"]


_default_router: Optional[ProviderRouter] = None


def get_router() -> ProviderRouter:
    global _default_router
    if _default_router is None:
        _default_router = ProviderRouter()
    return _default_router
