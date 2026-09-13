"""
Libra Providers - NVIDIA NIM (Inference Microservice) Cloud API Adapter
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx

from packages.providers.base import ModelMetadata
from packages.providers.cost import calculate_cost
from packages.providers.errors import (
    ProviderAuthenticationError,
    ProviderOfflineError,
    normalize_http_error,
)
from packages.providers.openai import OpenAIProvider


class NvidiaProvider(OpenAIProvider):
    """Adapter for the NVIDIA NIM OpenAI-compatible API (https://integrate.api.nvidia.com/v1).

    Supports model-family scoped keys (e.g. NVIDIA_DEEPSEEK_API_KEY, NVIDIA_KIMI_API_KEY)
    as well as general NVIDIA_API_KEY.
    """

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

    def get_api_key_for_model(self, model: Optional[str] = None) -> Optional[str]:
        """Resolves the appropriate API key, respecting model-specific NVIDIA NIM keys."""
        if model:
            m = model.lower()
            if "deepseek" in m:
                deepseek_key = os.getenv("NVIDIA_DEEPSEEK_API_KEY")
                if deepseek_key:
                    return deepseek_key
            if "kimi" in m or "moonshot" in m:
                kimi_key = os.getenv("NVIDIA_KIMI_API_KEY")
                if kimi_key:
                    return kimi_key
        return self.api_key or os.getenv("NVIDIA_API_KEY")

    def _get_headers(self, model: Optional[str] = None) -> dict[str, str]:
        key = self.get_api_key_for_model(model)
        if not key:
            raise ProviderAuthenticationError(
                message=f"API key not configured for provider '{self.name}'. "
                "Set NVIDIA_API_KEY (or NVIDIA_DEEPSEEK_API_KEY / NVIDIA_KIMI_API_KEY) in your .env file.",
                provider=self.name,
                status_code=401,
            )
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    def _normalize_model_id(self, model: str) -> str:
        m = model.lower().strip()
        if "kimi" in m or "moonshot" in m:
            if "k3" in m:
                return "moonshotai/kimi-k3"
            return "moonshotai/kimi-k2.6"
        if "deepseek" in m:
            if "pro" in m:
                return "deepseek-ai/deepseek-v4-pro-0813"
            return "deepseek-ai/deepseek-v4-flash-0731"
        if "nemotron" in m and "70b" in m:
            return "nvidia/llama-3.1-nemotron-70b-instruct"
        if "nemotron" in m and "340b" in m:
            return "nvidia/nemotron-4-340b-instruct"
        if "mistral" in m and "large" in m:
            return "mistralai/mistral-large-2-instruct"
        return model

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "deepseek-ai/deepseek-v4-flash-0731",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        headers = self._get_headers(model=model)
        norm_model = self._normalize_model_id(model)
        payload: dict[str, Any] = {
            "model": norm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        client = self._get_client()
        try:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            # If model-specific key got 403, retry once with primary general key if different
            if resp.status_code == 403:
                primary_key = self.api_key or os.getenv("NVIDIA_API_KEY")
                current_key = self.get_api_key_for_model(model)
                if primary_key and primary_key != current_key:
                    fallback_headers = {
                        "Authorization": f"Bearer {primary_key}",
                        "Content-Type": "application/json",
                    }
                    resp = await client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=fallback_headers,
                    )
            if resp.status_code != 200:
                if resp.status_code == 403:
                    raise ProviderAuthenticationError(
                        message=(
                            f"[NVIDIA NIM 403 Forbidden] Authorization failed for model '{norm_model}'. "
                            "Your NVIDIA API key authenticated, but NVIDIA NIM denied inference access. "
                            "Ensure your account at https://build.nvidia.com has active credits and "
                            "that you have visited the model card and clicked 'Get API Key' to accept its terms."
                        ),
                        provider=self.name,
                        status_code=403,
                        details=resp.text,
                    )
                raise normalize_http_error(resp.status_code, resp.text, self.name)
            data = resp.json()
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            cost = calculate_cost(norm_model, prompt_tokens, completion_tokens)
            data["cost"] = cost
            return data
        except httpx.RequestError as exc:
            raise ProviderOfflineError(
                message=f"Failed to connect to {self.name.upper()} API at {self.base_url}: {exc}",
                provider=self.name,
                details=str(exc),
            ) from exc

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str = "deepseek-ai/deepseek-v4-flash-0731",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ):
        import json

        headers = self._get_headers(model=model)
        norm_model = self._normalize_model_id(model)
        payload: dict[str, Any] = {
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
        try:
            response = await client.send(
                client.build_request(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                ),
                stream=True,
            )
            # If 403 with model-specific key, try fallback with primary key
            if response.status_code == 403:
                primary_key = self.api_key or os.getenv("NVIDIA_API_KEY")
                current_key = self.get_api_key_for_model(model)
                if primary_key and primary_key != current_key:
                    await response.aclose()
                    fallback_headers = {
                        "Authorization": f"Bearer {primary_key}",
                        "Content-Type": "application/json",
                    }
                    response = await client.send(
                        client.build_request(
                            "POST",
                            f"{self.base_url}/chat/completions",
                            json=payload,
                            headers=fallback_headers,
                        ),
                        stream=True,
                    )

            if response.status_code != 200:
                body = await response.aread()
                if response.status_code == 403:
                    raise ProviderAuthenticationError(
                        message=(
                            f"[NVIDIA NIM 403 Forbidden] Authorization failed for model '{norm_model}'. "
                            "Your NVIDIA API key authenticated, but NVIDIA NIM denied inference access. "
                            "Ensure your account at https://build.nvidia.com has active credits and "
                            "that you have visited the model card and clicked 'Get API Key' to accept its terms."
                        ),
                        provider=self.name,
                        status_code=403,
                        details=body.decode("utf-8"),
                    )
                raise normalize_http_error(response.status_code, body.decode("utf-8"), self.name)

            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
            await response.aclose()
        except httpx.RequestError as exc:
            raise ProviderOfflineError(
                message=f"Streaming error connecting to {self.name.upper()} API: {exc}",
                provider=self.name,
                details=str(exc),
            ) from exc

    async def list_models(self) -> list[ModelMetadata]:
        return [
            ModelMetadata(
                id="deepseek-ai/deepseek-v4-flash-0731",
                name="DeepSeek V4 Flash (NVIDIA NIM)",
                provider="nvidia",
                architecture="DeepSeek V4 Architecture",
                context_length=128000,
                license="Commercial Cloud / DeepSeek",
                is_local=False,
                requires_gpu=False,
                hardware_tier="cloud",
                capabilities=self.capabilities(),
            ),
            ModelMetadata(
                id="moonshotai/kimi-k2.6",
                name="Kimi k2.6 (NVIDIA NIM)",
                provider="nvidia",
                architecture="Moonshot AI Architecture",
                context_length=128000,
                license="Commercial Cloud / Moonshot AI",
                is_local=False,
                requires_gpu=False,
                hardware_tier="cloud",
                capabilities=self.capabilities(),
            ),
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
        ]
