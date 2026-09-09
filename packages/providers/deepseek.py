"""
Libra Providers - DeepSeek Commercial API Adapter
"""

from __future__ import annotations

import os
from typing import Optional

import httpx

from packages.providers.base import ModelMetadata
from packages.providers.openai import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    """Adapter for the DeepSeek OpenAI-compatible API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com/v1",
        timeout: float = 60.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        super().__init__(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=base_url,
            provider_name="deepseek",
            timeout=timeout,
            http_client=http_client,
        )

    async def list_models(self) -> list[ModelMetadata]:
        return [
            ModelMetadata(
                id="deepseek-chat",
                name="DeepSeek-V3",
                provider="deepseek",
                architecture="DeepSeek MoE Architecture",
                context_length=64000,
                license="Commercial API",
                is_local=False,
                requires_gpu=False,
                hardware_tier="cloud",
                capabilities=self.capabilities(),
            ),
            ModelMetadata(
                id="deepseek-reasoner",
                name="DeepSeek-R1 (Reasoning)",
                provider="deepseek",
                architecture="DeepSeek MoE with Reinforcement Learning",
                context_length=64000,
                license="Commercial API",
                is_local=False,
                requires_gpu=False,
                hardware_tier="cloud",
                capabilities=self.capabilities(),
            ),
        ]
