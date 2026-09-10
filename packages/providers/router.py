"""
Libra Providers - Unified Model Provider Router

Orchestrates multi-provider routing across local runtimes (Ollama, HF, Libra Lab),
cloud services (OpenAI, Gemini, Claude, DeepSeek, Groq, OpenRouter), and offline fallbacks.
"""

from __future__ import annotations

from typing import Any, Optional

from packages.providers.anthropic import AnthropicProvider
from packages.providers.base import BaseProvider
from packages.providers.deepseek import DeepSeekProvider
from packages.providers.gemini import GeminiProvider
from packages.providers.generic_openai import OpenRouterProvider
from packages.providers.groq import GroqProvider
from packages.providers.huggingface import HuggingFaceProvider
from packages.providers.local_transformer import LocalTransformerProvider
from packages.providers.mock import MockProvider
from packages.providers.ollama import OllamaProvider
from packages.providers.openai import OpenAIProvider
from packages.providers.vllm_stub import VLLMProvider


class ProviderRouter:
    """Orchestrates model provider selection, routing, and fallbacks."""

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {
            "ollama": OllamaProvider(),
            "libra_lab": LocalTransformerProvider(),
            "huggingface": HuggingFaceProvider(),
            "mock-provider": MockProvider(),
            "openai": OpenAIProvider(),
            "gemini": GeminiProvider(),
            "anthropic": AnthropicProvider(),
            "deepseek": DeepSeekProvider(),
            "groq": GroqProvider(),
            "openrouter": OpenRouterProvider(),
            "vllm": VLLMProvider(),
        }

    @property
    def providers(self) -> dict[str, BaseProvider]:
        """Access registered providers dictionary."""
        return self._providers

    def get_provider(self, name: str) -> BaseProvider:
        """Retrieve a specific provider by name or alias."""
        name_clean = name.lower().strip()
        if name_clean in self._providers:
            return self._providers[name_clean]

        # Aliases
        if "mock" in name_clean:
            return self._providers["mock-provider"]
        if "claude" in name_clean:
            return self._providers["anthropic"]
        if "google" in name_clean:
            return self._providers["gemini"]
        if "hf" in name_clean or "hugging" in name_clean:
            return self._providers["huggingface"]
        if "lab" in name_clean or "libra" in name_clean:
            return self._providers["libra_lab"]

        raise ValueError(f"Unknown provider '{name}'. Available: {list(self._providers.keys())}")

    async def resolve_provider_for_model(
        self,
        model_id: str,
        requested_provider: Optional[str] = None,
    ) -> BaseProvider:
        """Determines the best active provider for a model request with safe fallbacks."""
        if requested_provider:
            return self.get_provider(requested_provider)

        model_lower = model_id.lower()

        # 1. Cloud routing heuristics by model prefix
        if (
            model_lower.startswith("gpt-")
            or model_lower.startswith("o1")
            or model_lower.startswith("text-embedding")
        ):
            prov = self._providers["openai"]
            if getattr(prov, "api_key", None):
                return prov
            raise ValueError(
                f"OpenAI model '{model_id}' requested, but OPENAI_API_KEY is not configured in .env. "
                "Please configure OPENAI_API_KEY or select a local/mock model."
            )

        if "gemini" in model_lower:
            prov = self._providers["gemini"]
            if getattr(prov, "api_key", None):
                return prov
            raise ValueError(
                f"Google Gemini model '{model_id}' requested, but GEMINI_API_KEY is not configured in .env. "
                "Please configure GEMINI_API_KEY or select a local/mock model."
            )

        if "claude" in model_lower:
            prov = self._providers["anthropic"]
            if getattr(prov, "api_key", None):
                return prov
            raise ValueError(
                f"Anthropic Claude model '{model_id}' requested, but ANTHROPIC_API_KEY is not configured in .env. "
                "Please configure ANTHROPIC_API_KEY or select a local/mock model."
            )

        if "deepseek" in model_lower and not model_lower.startswith("deepseek-r1:"):
            prov = self._providers["deepseek"]
            if getattr(prov, "api_key", None):
                return prov
            raise ValueError(
                f"DeepSeek model '{model_id}' requested, but DEEPSEEK_API_KEY is not configured in .env. "
                "Please configure DEEPSEEK_API_KEY or select a local/mock model."
            )

        if "groq" in model_lower:
            prov = self._providers["groq"]
            if getattr(prov, "api_key", None):
                return prov
            raise ValueError(
                f"Groq model '{model_id}' requested, but GROQ_API_KEY is not configured in .env. "
                "Please configure GROQ_API_KEY or select a local/mock model."
            )

        if "openrouter" in model_lower:
            prov = self._providers["openrouter"]
            if getattr(prov, "api_key", None):
                return prov
            raise ValueError(
                f"OpenRouter model '{model_id}' requested, but OPENROUTER_API_KEY is not configured in .env. "
                "Please configure OPENROUTER_API_KEY or select a local/mock model."
            )

        # 2. Local routing
        if model_lower.startswith("hf/") or "gpt2" in model_lower:
            return self._providers["huggingface"]

        if "libra" in model_lower and "mock" not in model_lower:
            return self._providers["libra_lab"]

        if "mock" in model_lower:
            return self._providers["mock-provider"]

        # 3. Check local Ollama daemon
        ollama_health = await self._providers["ollama"].health()
        if ollama_health.get("connected"):
            return self._providers["ollama"]

        # 4. Zero-cost fallback to MockProvider
        return self._providers["mock-provider"]

    def list_registered_providers(self) -> list[str]:
        return list(self._providers.keys())

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model_id: str = "mock-model",
        provider_name: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Routes a chat completion request to the resolved provider."""
        provider = await self.resolve_provider_for_model(model_id, provider_name)
        return await provider.chat(messages, model=model_id, **kwargs)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model_id: str = "mock-model",
        provider_name: Optional[str] = None,
        **kwargs: Any,
    ):
        """Routes a streaming chat request to the resolved provider."""
        provider = await self.resolve_provider_for_model(model_id, provider_name)
        async for chunk in provider.stream(messages, model=model_id, **kwargs):
            yield chunk


_default_router: Optional[ProviderRouter] = None


def get_router() -> ProviderRouter:
    global _default_router
    if _default_router is None:
        _default_router = ProviderRouter()
    return _default_router
