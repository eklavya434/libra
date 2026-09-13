"""
Unit tests for Libra NVIDIA NIM Provider Adapter.
Uses MockTransport for 100% zero-cost testing.
"""

from __future__ import annotations

import httpx
import pytest

from packages.providers.errors import ModelNotFoundError, ProviderAuthenticationError
from packages.providers.nvidia import NvidiaProvider
from packages.providers.router import ProviderRouter


@pytest.mark.asyncio
async def test_nvidia_provider_metadata():
    provider = NvidiaProvider(api_key="test-key")
    assert provider.name == "nvidia"
    assert provider.base_url == "https://integrate.api.nvidia.com/v1"
    models = await provider.list_models()
    assert len(models) >= 3
    model_ids = [m.id for m in models]
    assert "nvidia/llama-3.1-nemotron-70b-instruct" in model_ids
    assert "nvidia/nemotron-4-340b-instruct" in model_ids


@pytest.mark.asyncio
async def test_nvidia_health_check_online():
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        assert request.headers["authorization"] == "Bearer nvapi-test"
        return httpx.Response(
            200, json={"data": [{"id": "nvidia/llama-3.1-nemotron-70b-instruct"}]}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    provider = NvidiaProvider(api_key="nvapi-test", http_client=client)
    health = await provider.health()
    assert health["status"] == "online"
    assert health["configured"] is True


@pytest.mark.asyncio
async def test_nvidia_health_check_unconfigured():
    provider = NvidiaProvider(api_key="")
    provider.api_key = ""
    health = await provider.health()
    assert health["status"] == "unconfigured"
    assert health["configured"] is False


@pytest.mark.asyncio
async def test_nvidia_chat_mock_transport():
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer nvapi-test"
        payload = {
            "id": "chatcmpl-nim-123",
            "choices": [{"message": {"role": "assistant", "content": "Hello from NVIDIA NIM!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    provider = NvidiaProvider(api_key="nvapi-test", http_client=client)

    result = await provider.chat(
        [{"role": "user", "content": "Say hello"}],
        model="nvidia/llama-3.1-nemotron-70b-instruct",
    )
    assert result["choices"][0]["message"]["content"] == "Hello from NVIDIA NIM!"
    assert result["cost"]["prompt_tokens"] == 10
    assert result["cost"]["completion_tokens"] == 8
    assert result["cost"]["total_cost_usd"] > 0.0


@pytest.mark.asyncio
async def test_nvidia_streaming_mock_transport():
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        chunks = [
            'data: {"choices": [{"delta": {"content": "NVIDIA "}}]}\n\n',
            'data: {"choices": [{"delta": {"content": "NIM "}}]}\n\n',
            'data: {"choices": [{"delta": {"content": "Streaming"}}]}\n\n',
            "data: [DONE]\n\n",
        ]
        return httpx.Response(200, content="".join(chunks).encode("utf-8"))

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    provider = NvidiaProvider(api_key="nvapi-test", http_client=client)

    collected = []
    async for chunk in provider.stream(
        [{"role": "user", "content": "Stream test"}],
        model="nvidia/llama-3.1-nemotron-70b-instruct",
    ):
        collected.append(chunk)

    full_text = "".join(collected)
    assert full_text == "NVIDIA NIM Streaming"


@pytest.mark.asyncio
async def test_nvidia_403_authorization_error():
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"status": 403, "title": "Forbidden", "detail": "Authorization failed"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    provider = NvidiaProvider(api_key="nvapi-test", http_client=client)

    with pytest.raises(ProviderAuthenticationError) as exc_info:
        await provider.chat(
            [{"role": "user", "content": "Test"}],
            model="nvidia/llama-3.1-nemotron-70b-instruct",
        )
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_nvidia_410_gone_error():
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            410,
            json={
                "type": "about:blank",
                "title": "Gone",
                "status": 410,
                "detail": "The model has reached its end of life.",
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    provider = NvidiaProvider(api_key="nvapi-test", http_client=client)

    with pytest.raises(ModelNotFoundError) as exc_info:
        await provider.chat(
            [{"role": "user", "content": "Test"}],
            model="meta/llama-3.3-70b-instruct",
        )
    assert exc_info.value.status_code == 410


@pytest.mark.asyncio
async def test_router_resolves_nvidia_provider():
    router = ProviderRouter()
    # Inject api_key for test
    router.providers["nvidia"].api_key = "nvapi-test"

    resolved = await router.resolve_provider_for_model("nvidia/llama-3.1-nemotron-70b-instruct")
    assert isinstance(resolved, NvidiaProvider)
    assert resolved.name == "nvidia"

    resolved_by_alias = router.get_provider("nvidia")
    assert isinstance(resolved_by_alias, NvidiaProvider)

    resolved_by_nim = router.get_provider("nim")
    assert isinstance(resolved_by_nim, NvidiaProvider)


@pytest.mark.asyncio
async def test_nvidia_model_specific_keys(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "general-key")
    monkeypatch.setenv("NVIDIA_DEEPSEEK_API_KEY", "deepseek-special-key")
    monkeypatch.setenv("NVIDIA_KIMI_API_KEY", "kimi-special-key")

    provider = NvidiaProvider()

    assert (
        provider.get_api_key_for_model("deepseek-ai/deepseek-v4-flash-0731")
        == "deepseek-special-key"
    )
    assert provider.get_api_key_for_model("moonshotai/kimi-k2.6") == "kimi-special-key"
    assert provider.get_api_key_for_model("nvidia/llama-3.1-nemotron-70b-instruct") == "general-key"

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer deepseek-special-key"
        payload = {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": "DeepSeek NIM output"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    provider._custom_client = client
    res = await provider.chat(
        [{"role": "user", "content": "Hi"}],
        model="deepseek-ai/deepseek-v4-flash-0731",
    )
    assert res["choices"][0]["message"]["content"] == "DeepSeek NIM output"
