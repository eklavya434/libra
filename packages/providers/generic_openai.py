"""
Libra Providers - Generic OpenAI-Compatible & OpenRouter Adapter
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any, Optional

import httpx

from packages.providers.base import ModelMetadata
from packages.providers.errors import normalize_http_error
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
        self._api_key = api_key
        super().__init__(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url=base_url,
            provider_name="openrouter",
            timeout=timeout,
            http_client=http_client,
        )

    @property
    def api_key(self) -> Optional[str]:
        if getattr(self, "_api_key", None):
            return self._api_key
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            try:
                from apps.backend.core.config import settings

                key = settings.openrouter_api_key or None
            except Exception:
                pass
        return key

    @api_key.setter
    def api_key(self, val: Optional[str]) -> None:
        self._api_key = val

    def _normalize_model_id(self, model: str) -> str:
        clean = model.strip()
        if clean.startswith("openrouter/"):
            rest = clean[11:]
            if rest in ("auto", ""):
                return "openrouter/auto"
            return rest
        return clean

    def _get_headers(self) -> dict[str, str]:
        headers = super()._get_headers()
        headers["HTTP-Referer"] = "https://www.libraai.me"
        headers["X-Title"] = "Project Libra"
        return headers

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "openrouter/auto",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        data = await super().chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            **kwargs,
        )
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            if not msg.get("content"):
                msg["content"] = msg.get("reasoning_content") or msg.get("reasoning") or ""
        return data

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str = "openrouter/auto",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        headers = self._get_headers()
        norm_model = self._normalize_model_id(model)
        payload = {
            "model": norm_model,
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
                        content = (
                            delta.get("content")
                            or delta.get("reasoning_content")
                            or delta.get("reasoning")
                            or ""
                        )
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

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
