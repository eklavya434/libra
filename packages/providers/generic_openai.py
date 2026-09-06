"""
Libra Providers - Generic OpenAI-Compatible & OpenRouter Adapter
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx

from packages.providers.base import ModelMetadata
from packages.providers.openai import OpenAIProvider


class GenericOpenAIProvider(OpenAIProvider):
    """Generic adapter for any OpenAI-compatible API gateway (e.g. Together, Anyscale, LM Studio)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:1234/v1",
        provider_name: str = "generic_openai",
        timeout: float = 60.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        super().__init__(
            api_key=api_key or os.getenv("GENERIC_OPENAI_API_KEY", ""),
            base_url=base_url,
            provider_name=provider_name,
            timeout=timeout,
            http_client=http_client,
        )


class OpenRouterProvider(OpenAIProvider):
    """Adapter for the OpenRouter multi-model aggregation gateway."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 60.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        super().__init__(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url=base_url,
            provider_name="openrouter",
            timeout=timeout,
            http_client=http_client,
        )

    def _get_headers(self) -> dict[str, str]:
        headers = super()._get_headers()
        headers["HTTP-Referer"] = "https://github.com/eklavya434/libra"
        headers["X-Title"] = "Project Libra"
        return headers

    async def list_models(self) -> list[ModelMetadata]:
        return [
            ModelMetadata(
                id="openrouter/auto",
                name="OpenRouter Auto Router",
                provider="openrouter",
                architecture="Dynamic Multi-Provider Gateway",
                context_length=128000,
                license="Commercial Gateway",
                is_local=False,
                requires_gpu=False,
                hardware_tier="cloud",
                capabilities=self.capabilities(),
            )
        ]
