"""
Libra Providers - Moonshot AI / Kimi REST & Streaming Adapter

Connects to the official Moonshot AI (Kimi) API via OpenAI-compatible gateway.
Base URL: https://api.moonshot.cn/v1
Models:
  - kimi-k1.5
  - moonshot-v1-8k
  - moonshot-v1-32k
  - moonshot-v1-128k
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any, Optional

import httpx

from packages.providers.base import ModelMetadata
from packages.providers.openai import OpenAIProvider


class KimiProvider(OpenAIProvider):
    """Adapter for Moonshot AI (Kimi) OpenAI-compatible chat completions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.moonshot.cn/v1",
        timeout: float = 60.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        resolved_key = api_key or os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY") or ""
        super().__init__(
            api_key=resolved_key,
            base_url=base_url,
            provider_name="kimi",
            timeout=timeout,
            http_client=http_client,
        )

    def capabilities(self) -> dict[str, bool]:
        return {
            "supports_text": True,
            "supports_vision": True,
            "supports_tools": True,
            "supports_reasoning": True,
            "supports_embeddings": False,
            "supports_streaming": True,
        }

    def _normalize_model_id(self, model: str) -> str:
        """Map aliases to official Moonshot/Kimi model identifiers."""
        m = model.lower().strip()
        if m in ("kimi", "kimi-chat", "kimi-latest"):
            return "kimi-k1.5"
        if m in ("moonshot", "moonshot-v1"):
            return "moonshot-v1-8k"
        return model

    async def list_models(self) -> list[ModelMetadata]:
        return [
            ModelMetadata(
                id="kimi-k1.5",
                name="Kimi k1.5",
                provider=self.name,
                architecture="Moonshot Multimodal Transformer",
                context_length=128000,
                license="Commercial API",
                is_local=False,
                requires_gpu=False,
                hardware_tier="cloud",
                capabilities=self.capabilities(),
            ),
            ModelMetadata(
                id="moonshot-v1-8k",
                name="Moonshot v1 8K",
                provider=self.name,
                architecture="Moonshot Transformer",
                context_length=8192,
                license="Commercial API",
                is_local=False,
                requires_gpu=False,
                hardware_tier="cloud",
                capabilities=self.capabilities(),
            ),
            ModelMetadata(
                id="moonshot-v1-32k",
                name="Moonshot v1 32K",
                provider=self.name,
                architecture="Moonshot Transformer",
                context_length=32768,
                license="Commercial API",
                is_local=False,
                requires_gpu=False,
                hardware_tier="cloud",
                capabilities=self.capabilities(),
            ),
            ModelMetadata(
                id="moonshot-v1-128k",
                name="Moonshot v1 128K",
                provider=self.name,
                architecture="Moonshot Transformer",
                context_length=128000,
                license="Commercial API",
                is_local=False,
                requires_gpu=False,
                hardware_tier="cloud",
                capabilities=self.capabilities(),
            ),
        ]

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "kimi-k1.5",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_model = self._normalize_model_id(model)
        return await super().chat(
            messages=messages,
            model=normalized_model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            **kwargs,
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str = "kimi-k1.5",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        normalized_model = self._normalize_model_id(model)
        async for chunk in super().stream(
            messages=messages,
            model=normalized_model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            **kwargs,
        ):
            yield chunk
