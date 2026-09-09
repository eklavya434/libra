"""
Libra Providers - OpenAI REST & Streaming Adapter

Connects to the official OpenAI API or any compatible gateway (Groq, DeepSeek, OpenRouter).
Supports:
  1. API key discovery and zero-cost offline state
  2. Non-streaming and streaming chat completions via httpx
  3. Cost calculation & usage tracking
  4. Normalized error handling (401, 429, 500)
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any, Optional

import httpx

from packages.providers.base import BaseProvider, ModelMetadata
from packages.providers.cost import calculate_cost
from packages.providers.errors import ProviderAuthenticationError, normalize_http_error


class OpenAIProvider(BaseProvider):
    """Adapter for OpenAI and OpenAI-compatible API services."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        provider_name: str = "openai",
        timeout: float = 60.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self._provider_name = provider_name
        self.api_key = (
            api_key or os.getenv(f"{provider_name.upper()}_API_KEY") or os.getenv("OPENAI_API_KEY")
        )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._custom_client = http_client

    @property
    def name(self) -> str:
        return self._provider_name

    def capabilities(self) -> dict[str, bool]:
        return {
            "supports_text": True,
            "supports_vision": True,
            "supports_tools": True,
            "supports_reasoning": True,
            "supports_embeddings": True,
            "supports_streaming": True,
        }

    def _get_headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ProviderAuthenticationError(
                message=f"API key not configured for provider '{self.name}'. "
                f"Set {self.name.upper()}_API_KEY in your .env file, or use local zero-cost models.",
                provider=self.name,
                status_code=401,
            )
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _get_client(self) -> httpx.AsyncClient:
        if self._custom_client is not None:
            return self._custom_client
        return httpx.AsyncClient(timeout=self.timeout)

    async def health(self) -> dict[str, Any]:
        """Verify API key configuration and network connectivity."""
        if not self.api_key:
            return {
                "status": "unconfigured",
                "provider": self.name,
                "configured": False,
                "guidance": f"Add {self.name.upper()}_API_KEY to .env to enable this commercial provider.",
            }

        try:
            client = self._get_client()
            headers = self._get_headers()
            resp = await client.get(f"{self.base_url}/models", headers=headers)
            if resp.status_code == 200:
                return {"status": "online", "provider": self.name, "configured": True}
            err = normalize_http_error(resp.status_code, resp.text, self.name)
            return {
                "status": "error",
                "provider": self.name,
                "configured": False,
                "error": err.message,
            }
        except Exception as e:
            return {
                "status": "offline",
                "provider": self.name,
                "configured": False,
                "error": str(e),
            }

    async def list_models(self) -> list[ModelMetadata]:
        return [
            ModelMetadata(
                id="gpt-4o",
                name="GPT-4o (Omni)",
                provider=self.name,
                architecture="Proprietary Transformer",
                context_length=128000,
                license="Commercial API",
                is_local=False,
                requires_gpu=False,
                hardware_tier="cloud",
                capabilities=self.capabilities(),
            ),
            ModelMetadata(
                id="gpt-4o-mini",
                name="GPT-4o Mini",
                provider=self.name,
                architecture="Proprietary Transformer",
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
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        headers = self._get_headers()
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        client = self._get_client()
        resp = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
        if resp.status_code != 200:
            raise normalize_http_error(
                resp.status_code,
                resp.json()
                if resp.headers.get("content-type", "").startswith("application/json")
                else resp.text,
                self.name,
            )

        data = resp.json()
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cost_info = calculate_cost(model, prompt_tokens, completion_tokens)

        data["cost"] = cost_info
        return data

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        headers = self._get_headers()
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop

        client = self._get_client()
        async with client.stream(
            "POST", f"{self.base_url}/chat/completions", headers=headers, json=payload
        ) as resp:
            if resp.status_code != 200:
                err_text = await resp.aread()
                raise normalize_http_error(
                    resp.status_code, err_text.decode("utf-8", errors="ignore"), self.name
                )

            async for line in resp.aiter_lines():
                if not line or not line.strip():
                    continue
                if line.startswith("data: "):
                    line_data = line[6:].strip()
                    if line_data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line_data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    async def embeddings(
        self, texts: list[str], model: str = "text-embedding-3-small"
    ) -> list[list[float]]:
        headers = self._get_headers()
        client = self._get_client()
        resp = await client.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json={"model": model, "input": texts},
        )
        if resp.status_code != 200:
            raise normalize_http_error(resp.status_code, resp.text, self.name)
        data = resp.json()
        return [item.get("embedding", []) for item in data.get("data", [])]
