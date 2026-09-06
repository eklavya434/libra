"""
Libra Providers - Mock Deterministic Provider
Used for zero-cost offline testing, CI/CD, and fast frontend development.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from packages.providers.base import BaseProvider, ModelMetadata


class MockProvider(BaseProvider):
    """A zero-cost, deterministic mock provider for testing and offline development."""

    def __init__(self, name: str = "mock-provider") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def list_models(self) -> list[ModelMetadata]:
        return [
            ModelMetadata(
                id="libra-mock-v1",
                name="Libra Mock Model v1",
                provider=self.name,
                architecture="Decoder-Only Transformer (Simulation)",
                context_length=512,
                parameter_count="1M",
                license="Apache-2.0",
                is_local=True,
                requires_gpu=False,
                hardware_tier="cpu-light",
                capabilities={
                    "supports_text": True,
                    "supports_vision": False,
                    "supports_tools": True,
                    "supports_reasoning": True,
                    "supports_embeddings": True,
                },
            )
        ]

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "libra-mock-v1",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> dict[str, Any]:
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        reply = f"[MockResponse] Received: '{last_user_msg}'. System operating normally in educational mode."
        return {
            "model": model,
            "provider": self.name,
            "choices": [{"message": {"role": "assistant", "content": reply}}],
            "usage": {
                "prompt_tokens": len(last_user_msg.split()),
                "completion_tokens": len(reply.split()),
                "total_tokens": len(last_user_msg.split()) + len(reply.split()),
            },
        }

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str = "libra-mock-v1",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        reply = f"[MockStream] Hello from Libra! You said: '{last_user_msg}'."
        tokens = reply.split(" ")
        for token in tokens:
            await asyncio.sleep(0.02)  # simulate token latency
            yield token + " "

    async def health(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "provider": self.name,
            "is_mock": True,
            "active": True,
        }

    async def embeddings(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        # Deterministic 16-dimensional dummy embeddings based on length
        return [[float(len(t) % 10 + i) * 0.1 for i in range(16)] for t in texts]

    def capabilities(self) -> dict[str, bool]:
        return {
            "supports_text": True,
            "supports_vision": False,
            "supports_tools": True,
            "supports_reasoning": True,
            "supports_embeddings": True,
        }
