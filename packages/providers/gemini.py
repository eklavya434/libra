"""
Libra Providers - Google Gemini REST & Streaming Adapter
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from typing import Any, Optional

import httpx

from packages.providers.base import BaseProvider, ModelMetadata
from packages.providers.cost import calculate_cost
from packages.providers.errors import ProviderAuthenticationError, normalize_http_error


class GeminiProvider(BaseProvider):
    """Adapter for Google Generative AI (Gemini) REST endpoints."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 60.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._custom_client = http_client

    @property
    def name(self) -> str:
        return "gemini"

    def capabilities(self) -> dict[str, bool]:
        return {
            "supports_text": True,
            "supports_vision": True,
            "supports_tools": True,
            "supports_reasoning": True,
            "supports_embeddings": True,
            "supports_streaming": True,
        }

    def _get_client(self) -> httpx.AsyncClient:
        if self._custom_client is not None:
            return self._custom_client
        return httpx.AsyncClient(timeout=self.timeout)

    async def health(self) -> dict[str, Any]:
        if not self.api_key:
            return {
                "status": "unconfigured",
                "provider": self.name,
                "configured": False,
                "guidance": "Add GEMINI_API_KEY to .env to enable Google Gemini models (free tier available).",
            }
        return {"status": "online", "provider": self.name, "configured": True}

    async def list_models(self) -> list[ModelMetadata]:
        return [
            ModelMetadata(
                id="gemini-1.5-flash",
                name="Gemini 1.5 Flash",
                provider=self.name,
                architecture="Google Multimodal Transformer",
                context_length=1000000,
                license="Commercial API / Free Tier",
                is_local=False,
                requires_gpu=False,
                hardware_tier="cloud",
                capabilities=self.capabilities(),
            ),
            ModelMetadata(
                id="gemini-1.5-pro",
                name="Gemini 1.5 Pro",
                provider=self.name,
                architecture="Google Multimodal Transformer",
                context_length=2000000,
                license="Commercial API",
                is_local=False,
                requires_gpu=False,
                hardware_tier="cloud",
                capabilities=self.capabilities(),
            ),
        ]

    def _convert_messages(self, messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        gemini_contents: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user").lower()
            gemini_role = "model" if role == "assistant" else "user"
            gemini_contents.append(
                {
                    "role": gemini_role,
                    "parts": [{"text": m.get("content", "")}],
                }
            )
        return gemini_contents

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise ProviderAuthenticationError(
                message="GEMINI_API_KEY not configured. Add key to .env or use local models.",
                provider=self.name,
                status_code=401,
            )

        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
        contents = self._convert_messages(messages)
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if stop:
            payload["generationConfig"]["stopSequences"] = stop

        client = self._get_client()
        resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            raise normalize_http_error(resp.status_code, resp.text, self.name)

        data = resp.json()
        text_out = ""
        try:
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text_out = "".join(p.get("text", "") for p in parts)
        except Exception:
            text_out = ""

        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)
        cost_info = calculate_cost(model, prompt_tokens, completion_tokens)

        return {
            "id": f"gemini-{int(time.time())}",
            "provider": self.name,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text_out},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "cost": cost_info,
        }

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        if not self.api_key:
            raise ProviderAuthenticationError(
                message="GEMINI_API_KEY not configured.",
                provider=self.name,
                status_code=401,
            )

        url = f"{self.base_url}/models/{model}:streamGenerateContent?alt=sse&key={self.api_key}"
        contents = self._convert_messages(messages)
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if stop:
            payload["generationConfig"]["stopSequences"] = stop

        client = self._get_client()
        async with client.stream("POST", url, json=payload) as resp:
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
                    try:
                        chunk = json.loads(line_data)
                        candidates = chunk.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            for p in parts:
                                txt = p.get("text", "")
                                if txt:
                                    yield txt
                    except json.JSONDecodeError:
                        continue

    async def embeddings(
        self, texts: list[str], model: str = "text-embedding-004"
    ) -> list[list[float]]:
        if not self.api_key:
            return [[0.0] * 768 for _ in texts]
        url = f"{self.base_url}/models/{model}:batchEmbedContents?key={self.api_key}"
        requests = [
            {"model": f"models/{model}", "content": {"parts": [{"text": t}]}} for t in texts
        ]
        client = self._get_client()
        resp = await client.post(url, json={"requests": requests})
        if resp.status_code == 200:
            data = resp.json()
            return [e.get("values", []) for e in data.get("embeddings", [])]
        return [[0.0] * 768 for _ in texts]
