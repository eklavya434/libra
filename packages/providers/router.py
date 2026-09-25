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
from packages.providers.kimi import KimiProvider
from packages.providers.local_transformer import LocalTransformerProvider
from packages.providers.mock import MockProvider
from packages.providers.nvidia import NvidiaProvider
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
            "nvidia": NvidiaProvider(),
            "gemini": GeminiProvider(),
            "anthropic": AnthropicProvider(),
            "deepseek": DeepSeekProvider(),
            "groq": GroqProvider(),
            "kimi": KimiProvider(),
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
        if "kimi" in name_clean or "moonshot" in name_clean:
            return self._providers["kimi"]
        if "hf" in name_clean or "hugging" in name_clean:
            return self._providers["huggingface"]
        if "lab" in name_clean or "libra" in name_clean:
            return self._providers["libra_lab"]
        if "nvidia" in name_clean or "nim" in name_clean:
            return self._providers["nvidia"]

        raise ValueError(f"Unknown provider '{name}'. Available: {list(self._providers.keys())}")

    async def resolve_provider_for_model(
        self,
        model_id: str,
        requested_provider: Optional[str] = None,
    ) -> BaseProvider:
        """Determine the active provider for a model request without silently faking output."""
        if requested_provider:
            return self.get_provider(requested_provider)

        model_lower = model_id.lower()

        # 1. Cloud routing heuristics by model prefix
        if (
            model_lower.startswith("gpt-")
            or model_lower.startswith("o1")
            or model_lower.startswith("chatgpt")
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

        if "kimi" in model_lower or "moonshot" in model_lower:
            prov = self._providers["kimi"]
            if getattr(prov, "api_key", None):
                return prov
            nv_prov = self._providers["nvidia"]
            if getattr(nv_prov, "api_key", None):
                return nv_prov
            raise ValueError(
                f"Kimi/Moonshot model '{model_id}' requested, but KIMI_API_KEY (or NVIDIA_KIMI_API_KEY) is not configured in .env. "
                "Please configure KIMI_API_KEY / NVIDIA_KIMI_API_KEY or select a local/mock model."
            )

        if (
            "deepseek" in model_lower
            and not model_lower.startswith("deepseek-r1:")
            and "coder" not in model_lower
        ):
            prov = self._providers["deepseek"]
            if getattr(prov, "api_key", None):
                return prov
            nv_prov = self._providers["nvidia"]
            if getattr(nv_prov, "api_key", None):
                return nv_prov
            raise ValueError(
                f"DeepSeek model '{model_id}' requested, but DEEPSEEK_API_KEY (or NVIDIA_DEEPSEEK_API_KEY) is not configured in .env. "
                "Please configure DEEPSEEK_API_KEY / NVIDIA_DEEPSEEK_API_KEY or select a local/mock model."
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

        if (
            model_lower.startswith("nvidia/")
            or model_lower.startswith("mistralai/")
            or "mistral-large" in model_lower
            or model_lower.startswith("nv-")
            or "nvidia" in model_lower
        ):
            prov = self._providers["nvidia"]
            if getattr(prov, "api_key", None):
                return prov
            raise ValueError(
                f"NVIDIA NIM model '{model_id}' requested, but NVIDIA_API_KEY is not configured in .env. "
                "Please configure NVIDIA_API_KEY or select a local/mock model."
            )

        # 2. OpenCode / Coder specialist routing (local Ollama required)
        if "opencode" in model_lower or "coder" in model_lower:
            installed, reason = await self._ollama_model_installed(model_id)
            if installed:
                return self._providers["ollama"]
            # Never degrade to MockProvider when the local engine is unavailable:
            # a public request must fail with an actionable message instead.
            raise ValueError(
                reason
                or f"Model '{model_id}' requires the local Ollama engine, which is not available here."
            )

        # 3. Local routing
        if model_lower.startswith("hf/") or "gpt2" in model_lower:
            return self._providers["huggingface"]

        if "libra" in model_lower and "mock" not in model_lower:
            return self._providers["libra_lab"]

        if "mock" in model_lower:
            return self._providers["mock-provider"]

        # 3. Check local Ollama daemon (only if the model is actually installed)
        ollama_health = await self._providers["ollama"].health()
        if ollama_health.get("connected"):
            installed, reason = await self._ollama_model_installed(model_id)
            if installed:
                return self._providers["ollama"]
            raise ValueError(reason)

        # Never silently route an unknown/unavailable model to MockProvider.
        # MockProvider remains available only when explicitly requested by model
        # ID/provider. A production chat request must either use the requested
        # real provider or fail with an actionable configuration error.
        raise ValueError(
            f"No active provider is available for model '{model_id}'. "
            "Select an installed Ollama/local Libra model, configure a supported cloud provider, "
            "or explicitly select a mock model for offline testing."
        )

    async def _ollama_model_installed(
        self,
        model_id: str,
        installed_ollama_ids: Optional[set[str]] = None,
    ) -> tuple[bool, str]:
        """Check an Ollama model is actually installed (avoids silent runtime 404s)."""
        # If the caller already fetched the installed-model snapshot, trust it:
        # avoids an HTTP round trip per model on the /models listing endpoint.
        if installed_ollama_ids is not None:
            if model_id in installed_ollama_ids:
                return True, ""
            return (
                False,
                f"Model '{model_id}' is not downloaded locally. Install it with: ollama pull {model_id}",
            )
        # Otherwise probe the running Ollama daemon for health and installed models.
        try:
            health = await self._providers["ollama"].health()
        except Exception:
            return False, "Ollama is not running on this machine. Start Ollama and retry."
        if not health.get("connected"):
            return False, "Ollama is not running on this machine. Start Ollama and retry."
        try:
            installed = await self._providers["ollama"].list_models()
            installed_ollama_ids = {m.id for m in installed}
        except Exception:
            return False, f"Ollama is running but could not verify model '{model_id}'."
        if model_id in installed_ollama_ids:
            return True, ""
        return (
            False,
            f"Model '{model_id}' is not downloaded locally. Install it with: ollama pull {model_id}",
        )

    async def check_model_available(
        self,
        model_id: str,
        installed_ollama_ids: Optional[set[str]] = None,
    ) -> tuple[bool, str]:
        """Return (available, unavailable_reason) mirroring resolve_provider_for_model.

        Used by the Model Registry UI so users see exactly which models can run
        right now and why others cannot (missing key vs. not downloaded).
        """
        model_lower = model_id.lower()

        def has(provider_id: str) -> bool:
            prov = self._providers.get(provider_id)
            return bool(prov and getattr(prov, "api_key", None))

        if (
            model_lower.startswith("gpt-")
            or model_lower.startswith("o1")
            or model_lower.startswith("chatgpt")
        ):
            if has("openai"):
                return True, ""
            return False, "Add OPENAI_API_KEY to .env to enable OpenAI models."

        if "gemini" in model_lower:
            if has("gemini"):
                return True, ""
            return False, "Add GEMINI_API_KEY to .env to enable Gemini models."

        if "claude" in model_lower:
            if has("anthropic"):
                return True, ""
            return False, "Add ANTHROPIC_API_KEY to .env to enable Claude models."

        if "kimi" in model_lower or "moonshot" in model_lower:
            if has("kimi") or has("nvidia"):
                return True, ""
            return (
                False,
                "Add KIMI_API_KEY (or NVIDIA_API_KEY) to .env to enable Kimi/Moonshot models.",
            )

        if (
            "deepseek" in model_lower
            and not model_lower.startswith("deepseek-r1:")
            and "coder" not in model_lower
        ):
            if has("deepseek") or has("nvidia"):
                return True, ""
            return (
                False,
                "Add DEEPSEEK_API_KEY (or NVIDIA_API_KEY) to .env to enable DeepSeek models.",
            )

        if "groq" in model_lower:
            if has("groq"):
                return True, ""
            return False, "Add GROQ_API_KEY to .env to enable Groq models."

        if "openrouter" in model_lower:
            if has("openrouter"):
                return True, ""
            return False, "Add OPENROUTER_API_KEY to .env to enable OpenRouter models."

        if (
            model_lower.startswith("nvidia/")
            or model_lower.startswith("mistralai/")
            or "mistral-large" in model_lower
            or model_lower.startswith("nv-")
            or "nvidia" in model_lower
        ):
            if has("nvidia"):
                return True, ""
            return False, "Add NVIDIA_API_KEY to .env to enable NVIDIA NIM models."

        if "opencode" in model_lower or "coder" in model_lower:
            return await self._ollama_model_installed(model_id, installed_ollama_ids)

        if model_lower.startswith("hf/") or "gpt2" in model_lower:
            return True, ""

        if "libra" in model_lower and "mock" not in model_lower:
            return True, ""

        if "mock" in model_lower:
            return True, ""

        # Anything else routes to the local Ollama daemon at request time.
        return await self._ollama_model_installed(model_id, installed_ollama_ids)

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
