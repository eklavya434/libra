"""
Libra Providers - NVIDIA NIM (Inference Microservice) Cloud API Adapter
"""

from __future__ import annotations

import os
from typing import Optional

import httpx

from packages.providers.base import ModelMetadata
from packages.providers.openai import OpenAIProvider


class NvidiaProvider(OpenAIProvider):
    """Adapter for the NVIDIA NIM OpenAI-compatible API (https://integrate.api.nvidia.com/v1)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        timeout: float = 60.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        super().__init__(
            api_key=api_key or os.getenv("NVIDIA_API_KEY"),
            base_url=base_url,
            provider_name="nvidia",
            timeout=timeout,
            http_client=http_client,
        )

    async def list_models(self) -> list[ModelMetadata]:
        return [
            ModelMetadata(
                id="nvidia/llama-3.1-nemotron-70b-instruct",
                name="NVIDIA Llama 3.1 Nemotron 70B Instruct",
                provider="nvidia",
                architecture="Nemotron-aligned Llama 3.1 70B",
                context_length=131072,
                license="NVIDIA Open Model License / Llama 3.1 Community",
                is_local=False,
                requires_gpu=False,
                hardware_tier="cloud",
                capabilities=self.capabilities(),
            ),
            ModelMetadata(
                id="nvidia/nemotron-4-340b-instruct",
                name="NVIDIA Nemotron-4 340B Instruct",
                provider="nvidia",
                architecture="Nemotron-4 340B Transformer",
                context_length=4096,
                license="NVIDIA Open Model License",
                is_local=False,
                requires_gpu=False,
                hardware_tier="cloud",
                capabilities=self.capabilities(),
            ),
            ModelMetadata(
                id="mistralai/mistral-large-2-instruct",
                name="Mistral Large 2 Instruct (NVIDIA NIM)",
                provider="nvidia",
                architecture="Mistral Large 2",
                context_length=128000,
                license="Mistral Commercial / Research",
                is_local=False,
                requires_gpu=False,
                hardware_tier="cloud",
                capabilities=self.capabilities(),
            ),
            ModelMetadata(
                id="deepseek-ai/deepseek-coder-6.7b-instruct",
                name="DeepSeek Coder 6.7B Instruct (NVIDIA NIM)",
                provider="nvidia",
                architecture="DeepSeek Coder",
                context_length=16384,
                license="DeepSeek Open License",
                is_local=False,
                requires_gpu=False,
                hardware_tier="cloud",
                capabilities=self.capabilities(),
            ),
        ]
