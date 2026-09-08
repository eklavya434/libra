"""
Libra Providers - Ollama Local Inference Adapter

Direct integration with the local Ollama runtime (http://127.0.0.1:11434).
Supports:
  1. Connection & daemon health checks
  2. Model listing from /api/tags
  3. Non-streaming chat completions
  4. Real-time streaming tokens via AsyncIterator
  5. Text embeddings via /api/embeddings
  6. Sampling controls: temperature, top_k, top_p, stop sequences, max_tokens
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from packages.providers.base import BaseProvider, ModelMetadata


class OllamaProvider(BaseProvider):
    """Local inference adapter for the Ollama runtime."""

    def __init__(self, base_url: str | None = None, timeout: float = 60.0):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "ollama"

    def capabilities(self) -> dict[str, bool]:
        return {
            "supports_text": True,
            "supports_vision": True,
            "supports_tools": True,
            "supports_reasoning": True,
            "supports_embeddings": True,
            "supports_streaming": True,
        }

    async def health(self) -> dict[str, Any]:
        """Checks if the local Ollama daemon is active and reachable."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self.base_url}/api/version")
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "status": "online",
                        "provider": "ollama",
                        "version": data.get("version", "unknown"),
                        "base_url": self.base_url,
                        "connected": True,
                    }
        except Exception as e:
            return {
                "status": "offline",
                "provider": "ollama",
                "base_url": self.base_url,
                "connected": False,
                "error": f"Ollama daemon not reachable at {self.base_url}: {type(e).__name__}",
                "guidance": "Run 'ollama serve' or launch the Ollama desktop application to enable local inference.",
            }

        return {
            "status": "error",
            "provider": "ollama",
            "base_url": self.base_url,
            "connected": False,
            "error": "Unexpected response status from Ollama daemon.",
        }

    async def list_models(self) -> list[ModelMetadata]:
        """Queries /api/tags for installed local models."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    tags = resp.json().get("models", [])
                    models: list[ModelMetadata] = []
                    for t in tags:
                        name = t.get("name", "unknown")
                        size_bytes = t.get("size", 0)
                        models.append(
                            ModelMetadata(
                                id=name,
                                name=name,
                                provider="ollama",
                                architecture=t.get("details", {}).get("family", "transformer"),
                                context_length=4096,
                                parameter_count=t.get("details", {}).get("parameter_size", "unknown"),
                                quantization=t.get("details", {}).get("quantization_level", "unknown"),
                                is_local=True,
                                requires_gpu=False,
                                hardware_tier="cpu-friendly",
                                capabilities=self.capabilities(),
                            )
                        )
                    return models
        except Exception:
            pass
        return []

    def _build_options(
        self,
        temperature: float,
        max_tokens: int,
        top_p: float = 0.9,
        top_k: int = 40,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Map standard inference hyperparameters to Ollama option fields."""
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
        }
        if stop:
            options["stop"] = stop
        for k, v in kwargs.items():
            if k in {"seed", "num_ctx", "repeat_penalty"}:
                options[k] = v
        return options

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "llama3.2:1b",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        top_k: int = 40,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a non-streaming chat request with Ollama."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": self._build_options(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
                **kwargs,
            ),
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
                if resp.status_code != 200:
                    raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")

                data = resp.json()
                msg = data.get("message", {})
                content = msg.get("content", "").strip()
                thinking = msg.get("thinking", "").strip()
                if thinking and content:
                    full_content = f"<think>\n{thinking}\n</think>\n\n{content}"
                elif content:
                    full_content = content
                elif thinking:
                    full_content = f"<think>\n{thinking}\n</think>"
                else:
                    full_content = ""

                return {
                    "id": f"ollama-{int(time.time())}",
                    "provider": "ollama",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": msg.get("role", "assistant"),
                                "content": full_content,
                            },
                            "finish_reason": "stop" if data.get("done") else "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0),
                        "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                    },
                }
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Ensure Ollama is running ('ollama serve')."
            )

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str = "llama3.2:1b",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        top_k: int = 40,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream token deltas in real-time from Ollama, supporting both standard and reasoning models."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": self._build_options(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
                **kwargs,
            ),
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                    if resp.status_code != 200:
                        err_text = await resp.aread()
                        raise RuntimeError(f"Ollama stream error {resp.status_code}: {err_text.decode('utf-8', errors='ignore')}")

                    in_thinking = False
                    async for line in resp.aiter_lines():
                        if line and line.strip():
                            chunk = json.loads(line)
                            msg_obj = chunk.get("message", {})
                            thinking_chunk = msg_obj.get("thinking", "")
                            content_chunk = msg_obj.get("content", "")

                            if thinking_chunk:
                                if not in_thinking:
                                    in_thinking = True
                                    yield "<think>\n"
                                yield thinking_chunk

                            if content_chunk:
                                if in_thinking:
                                    in_thinking = False
                                    yield "\n</think>\n\n"
                                yield content_chunk

                            if chunk.get("done", False):
                                if in_thinking:
                                    yield "\n</think>\n"
                                break
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Ensure Ollama is running ('ollama serve')."
            )

    async def embeddings(self, texts: list[str], model: str = "llama3.2:1b") -> list[list[float]]:
        """Generate embeddings for texts via /api/embeddings."""
        results: list[list[float]] = []
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                for text in texts:
                    resp = await client.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": model, "prompt": text},
                    )
                    if resp.status_code == 200:
                        results.append(resp.json().get("embedding", []))
                    else:
                        results.append([])
        except Exception:
            return [[] for _ in texts]
        return results
