"""
Libra Providers - Google Gemini REST & Streaming Adapter
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from typing import Any, Optional

import httpx

from packages.core.network import enable_ipv4_preference
from packages.providers.base import BaseProvider, ModelMetadata
from packages.providers.cost import calculate_cost
from packages.providers.errors import ProviderAuthenticationError, normalize_http_error

enable_ipv4_preference()


class GeminiProvider(BaseProvider):
    """Adapter for Google Generative AI (Gemini) REST endpoints."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 60.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            try:
                from apps.backend.core.config import settings

                api_key = settings.gemini_api_key or None
            except Exception:
                pass
        self.api_key = api_key
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

    def _normalize_model(self, model: str) -> str:
        clean = model.strip()
        if clean.startswith("models/"):
            clean = clean[7:]
        if clean in ("gemini-1.5-flash", "gemini-flash", "gemini-flash-1.5", "gemini"):
            return "gemini-2.5-flash"
        if clean in ("gemini-1.5-pro", "gemini-pro", "gemini-pro-1.5"):
            return "gemini-2.5-pro"
        return clean

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
                id="gemini-2.5-flash",
                name="Gemini 2.5 Flash",
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
                id="gemini-2.5-pro",
                name="Gemini 2.5 Pro",
                provider=self.name,
                architecture="Google Multimodal Transformer",
                context_length=2000000,
                license="Commercial API",
                is_local=False,
                requires_gpu=False,
                hardware_tier="cloud",
                capabilities=self.capabilities(),
            ),
            ModelMetadata(
                id="gemini-1.5-flash",
                name="Gemini 1.5 Flash (Compatibility Alias)",
                provider=self.name,
                architecture="Google Multimodal Transformer",
                context_length=1000000,
                license="Commercial API / Free Tier",
                is_local=False,
                requires_gpu=False,
                hardware_tier="cloud",
                capabilities=self.capabilities(),
            ),
        ]

    def _convert_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
        gemini_contents: list[dict[str, Any]] = []
        system_parts: list[dict[str, str]] = []

        for m in messages:
            role = m.get("role", "user").lower()
            content = m.get("content", "")
            if not content or not str(content).strip():
                continue

            # Strip non-printable control characters
            clean_text = "".join(ch for ch in str(content) if ch >= " " or ch in "\n\r\t").strip()
            if not clean_text:
                continue

            # Strip out legacy mock prefix echoes so Gemini never mimics mock echoes
            if clean_text.startswith("[MockStream]"):
                prefix = "[MockStream] Hello from Libra! You said:"
                if prefix in clean_text:
                    clean_text = clean_text.replace(prefix, "").strip().strip("'\"")
                else:
                    clean_text = clean_text.replace("[MockStream]", "").strip()

            if role == "system":
                system_parts.append({"text": clean_text})
            else:
                gemini_role = "model" if role == "assistant" else "user"
                # If consecutive message has the same role, combine them
                if gemini_contents and gemini_contents[-1]["role"] == gemini_role:
                    gemini_contents[-1]["parts"][0]["text"] += f"\n\n{clean_text}"
                else:
                    gemini_contents.append(
                        {
                            "role": gemini_role,
                            "parts": [{"text": clean_text}],
                        }
                    )

        # Gemini requires that the first content turn has role 'user'
        while gemini_contents and gemini_contents[0]["role"] == "model":
            gemini_contents.pop(0)

        # Gemini requires the final message to be 'user' to generate the next 'model' turn
        while gemini_contents and gemini_contents[-1]["role"] == "model":
            gemini_contents.pop()

        if not gemini_contents:
            gemini_contents = [{"role": "user", "parts": [{"text": "Hello"}]}]

        # Default system instruction if none provided
        if not system_parts:
            system_parts = [
                {
                    "text": (
                        "You are Libra, an intelligent and helpful AI conversational assistant. "
                        "Answer user questions accurately, helpfully, and conversationally in Markdown format."
                    )
                }
            ]

        system_instruction = {"parts": system_parts}
        return gemini_contents, system_instruction

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "gemini-2.5-flash",
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

        primary_model = self._normalize_model(model)
        candidate_models = [primary_model]
        for fb in ["gemini-3.5-flash", "gemini-flash-latest"]:
            if fb not in candidate_models:
                candidate_models.append(fb)

        contents, system_instruction = self._convert_messages(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        if stop:
            payload["generationConfig"]["stopSequences"] = stop

        client = self._get_client()
        resp = None
        target_model = primary_model

        for m in candidate_models:
            target_model = m
            url = f"{self.base_url}/models/{target_model}:generateContent?key={self.api_key}"
            for attempt in range(2):
                resp = await client.post(url, json=payload)
                if resp.status_code == 503 and attempt < 1:
                    await asyncio.sleep(1.0)
                    continue
                break
            if resp.status_code == 200:
                break
            if resp.status_code != 429:
                # If it's an auth error or client error (not rate-limit), break immediately
                break

        if resp is None or resp.status_code != 200:
            err_text = resp.text if resp is not None else "No response"
            code = resp.status_code if resp is not None else 500
            raise normalize_http_error(code, err_text, self.name)

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
        cost_info = calculate_cost(target_model, prompt_tokens, completion_tokens)

        return {
            "id": f"gemini-{int(time.time())}",
            "provider": self.name,
            "model": target_model,
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
        model: str = "gemini-2.5-flash",
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

        primary_model = self._normalize_model(model)
        candidate_models = [primary_model]
        if primary_model == "gemini-2.5-flash" and "gemini-2.5-pro" not in candidate_models:
            candidate_models.append("gemini-2.5-pro")

        contents, system_instruction = self._convert_messages(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        if stop:
            payload["generationConfig"]["stopSequences"] = stop

        client = self._get_client()
        last_err: Optional[Exception] = None

        for target_model in candidate_models:
            url = f"{self.base_url}/models/{target_model}:streamGenerateContent?alt=sse&key={self.api_key}"
            for attempt in range(2):
                try:
                    async with client.stream("POST", url, json=payload) as resp:
                        if resp.status_code == 503 and attempt < 1:
                            await asyncio.sleep(1.0)
                            continue
                        if resp.status_code == 429 and target_model != candidate_models[-1]:
                            # Quota exceeded for this specific model; try next candidate model
                            break
                        if resp.status_code != 200:
                            err_text = await resp.aread()
                            last_err = normalize_http_error(
                                resp.status_code,
                                err_text.decode("utf-8", errors="ignore"),
                                self.name,
                            )
                            if resp.status_code == 429:
                                break
                            raise last_err

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
                        return
                except Exception as e:
                    if "429" in str(e) and target_model != candidate_models[-1]:
                        break
                    last_err = e
                    if target_model == candidate_models[-1]:
                        raise e

        if last_err:
            raise last_err

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
