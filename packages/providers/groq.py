"""
Libra Providers - Groq High-Speed LPU API Adapter
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx

from packages.providers.base import ModelMetadata
from packages.providers.openai import OpenAIProvider


class GroqProvider(OpenAIProvider):
    """Adapter for the Groq Cloud OpenAI-compatible API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout: float = 60.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        super().__init__(
            api_key=api_key or os.getenv("GROQ_API_KEY"),
            base_url=base_url,
            provider_name="groq",
            timeout=timeout,
            http_client=http_client,
        )

    async def list_models(self) -> list[ModelMetadata]:
        return [
            ModelMetadata(
                id="llama-3.3-70b-versatile",
                name="Llama 3.3 70B (Groq LPU)",
                provider="groq",
                architecture="Llama-3.3 on Groq LPUs",
                context_length=128000,
                license="Commercial Cloud / Llama Community",
                is_local=False,
                requires_gpu=False,
                hardware_tier="cloud",
                capabilities=self.capabilities(),
            ),
            ModelMetadata(
                id="llama-3.1-8b-instant",
                name="Llama 3.1 8B Instant (Groq LPU)",
                provider="groq",
                architecture="Llama-3.1 on Groq LPUs",
                context_length=128000,
                license="Commercial Cloud / Llama Community",
                is_local=False,
                requires_gpu=False,
                hardware_tier="cloud",
                capabilities=self.capabilities(),
            ),
        ]
