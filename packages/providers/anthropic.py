"""
Libra Providers - Anthropic Claude REST & Streaming Adapter
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


class AnthropicProvider(BaseProvider):
    """Adapter for the Anthropic Claude Messages API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.anthropic.com/v1",
        timeout: float = 60.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._custom_client = http_client

    @property
    def name(self) -> str:
        return "anthropic"

    def capabilities(self) -> dict[str, bool]:
        return {
            "supports_text": True,
            "supports_vision": True,
            "supports_tools": True,
            "supports_reasoning": True,
            "supports_embeddings": False,
            "supports_streaming": True,
        }

    def _get_headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ProviderAuthenticationError(
                message="ANTHROPIC_API_KEY not set. Use local models or configure key in .env.",
                provider=self.name,
                status_code=401,
            )
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
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
                "guidance": "Add ANTHROPIC_API_KEY to .env to enable Claude models.",
            }
        return {"status": "online", "provider": self.name, "configured": True}

    async def list_models(self) -> list[ModelMetadata]:
        return [
            ModelMetadata(
                id="claude-3-5-sonnet-20241022",
                name="Claude 3.5 Sonnet",
                provider=self.name,
                architecture="Proprietary Transformer",
                context_length=200000,
                license="Commercial API",
                is_local=False,
                requires_gpu=False,
                hardware_tier="cloud",
                capabilities=self.capabilities(),
            ),
            ModelMetadata(
                id="claude-3-5-haiku-20241022",
                name="Claude 3.5 Haiku",
                provider=self.name,
                architecture="Proprietary Transformer",
                context_length=200000,
                license="Commercial API",
                is_local=False,
                requires_gpu=False,
                hardware_tier="cloud",
                capabilities=self.capabilities(),
            ),
        ]

    def _extract_system_and_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[Optional[str], list[dict[str, str]]]:
        system_content: Optional[str] = None
        filtered_msgs: list[dict[str, str]] = []
        for m in messages:
            role = m.get("role", "user").lower()
            if role == "system" and system_content is None:
                system_content = m.get("content", "")
            else:
                filtered_msgs.append({"role": role, "content": m.get("content", "")})
        return system_content, filtered_msgs

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "claude-3-5-haiku-20241022",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        headers = self._get_headers()
        system_prompt, clean_msgs = self._extract_system_and_messages(messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": clean_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if stop:
            payload["stop_sequences"] = stop

        client = self._get_client()
        resp = await client.post(f"{self.base_url}/messages", headers=headers, json=payload)
        if resp.status_code != 200:
            raise normalize_http_error(resp.status_code, resp.text, self.name)

        data = resp.json()
        text_content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_content += block.get("text", "")

        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        cost_info = calculate_cost(model, prompt_tokens, completion_tokens)

        return {
            "id": data.get("id", f"anthropic-{int(time.time())}"),
            "provider": self.name,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text_content},
                    "finish_reason": data.get("stop_reason", "stop"),
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
        model: str = "claude-3-5-haiku-20241022",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        headers = self._get_headers()
        system_prompt, clean_msgs = self._extract_system_and_messages(messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": clean_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if stop:
            payload["stop_sequences"] = stop

        client = self._get_client()
        async with client.stream(
            "POST", f"{self.base_url}/messages", headers=headers, json=payload
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
                    try:
                        chunk = json.loads(line_data)
                        c_type = chunk.get("type")
                        if c_type == "content_block_delta":
                            text = chunk.get("delta", {}).get("text", "")
                            if text:
                                yield text
                        elif c_type == "message_stop":
                            break
                    except json.JSONDecodeError:
                        continue

    async def embeddings(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        raise NotImplementedError("Anthropic API does not offer embedding endpoints directly.")
