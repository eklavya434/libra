"""
Unit tests for OpenCode, ChatGPT, Kimi, and Claude model integrations.
Verifies:
  1. KimiProvider chat & streaming via mock httpx transport.
  2. OpenAIProvider ChatGPT additions and alias normalization.
  3. AnthropicProvider Claude Opus additions and alias normalization.
  4. ProviderRouter heuristics for OpenCode, ChatGPT, Kimi, and Claude.
  5. ModelRegistry presence for all new models.
  6. Cost engine rates for new open-weights and commercial models.
"""

from __future__ import annotations

import json

import httpx
import pytest

from packages.models.catalog import get_default_registry
from packages.providers.anthropic import AnthropicProvider
from packages.providers.cost import calculate_cost
from packages.providers.kimi import KimiProvider
from packages.providers.openai import OpenAIProvider
from packages.providers.router import ProviderRouter


@pytest.mark.asyncio
async def test_kimi_provider_capabilities_and_metadata():
    provider = KimiProvider(api_key="test-kimi-key")
    assert provider.name == "kimi"
    caps = provider.capabilities()
    assert caps["supports_text"] is True
    assert caps["supports_streaming"] is True

    models = await provider.list_models()
    model_ids = [m.id for m in models]
    assert "kimi-k1.5" in model_ids
    assert "moonshot-v1-8k" in model_ids
    assert "moonshot-v1-32k" in model_ids
    assert "moonshot-v1-128k" in model_ids


@pytest.mark.asyncio
async def test_kimi_provider_chat_mock_transport():
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-kimi-key"
        assert "api.moonshot.cn" in str(request.url)
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "kimi-k1.5"
        payload = {
            "id": "chatcmpl-kimi-123",
            "choices": [{"message": {"role": "assistant", "content": "Hello from Kimi!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    provider = KimiProvider(api_key="test-kimi-key", http_client=client)

    # Test alias normalization 'kimi' -> 'kimi-k1.5'
    result = await provider.chat([{"role": "user", "content": "Hi Kimi"}], model="kimi")
    assert result["choices"][0]["message"]["content"] == "Hello from Kimi!"
    assert result["cost"]["total_tokens"] == 15
    assert result["cost"]["is_free"] is False


@pytest.mark.asyncio
async def test_kimi_provider_stream_mock_transport():
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        assert "api.moonshot.cn" in str(request.url)
        sse_data = (
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_data.encode("utf-8"),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    provider = KimiProvider(api_key="test-kimi-key", http_client=client)

    chunks = []
    async for chunk in provider.stream([{"role": "user", "content": "Hi"}], model="moonshot-v1-8k"):
        chunks.append(chunk)

    assert "".join(chunks) == "Hello world"


@pytest.mark.asyncio
async def test_openai_chatgpt_models_and_normalization():
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "chatgpt-4o-latest"
        payload = {
            "id": "chatcmpl-chatgpt-123",
            "choices": [{"message": {"role": "assistant", "content": "Hello from ChatGPT-4o!"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    provider = OpenAIProvider(api_key="test-key", http_client=client)

    # list_models includes new frontier & reasoning models
    models = await provider.list_models()
    ids = [m.id for m in models]
    assert "chatgpt-4o-latest" in ids
    assert "o1" in ids
    assert "o1-mini" in ids

    # chat alias 'chatgpt' resolves to 'chatgpt-4o-latest'
    res = await provider.chat([{"role": "user", "content": "test"}], model="chatgpt")
    assert res["choices"][0]["message"]["content"] == "Hello from ChatGPT-4o!"
    assert res["cost"]["model"] == "chatgpt-4o-latest"


@pytest.mark.asyncio
async def test_anthropic_claude_opus_and_normalization():
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "claude-3-opus-20240229"
        payload = {
            "id": "msg-opus-123",
            "content": [{"type": "text", "text": "Claude 3 Opus response"}],
            "usage": {"input_tokens": 25, "output_tokens": 12},
        }
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    provider = AnthropicProvider(api_key="test-claude-key", http_client=client)

    models = await provider.list_models()
    ids = [m.id for m in models]
    assert "claude-3-opus-20240229" in ids

    # alias 'claude-3-opus' resolves to 'claude-3-opus-20240229'
    res = await provider.chat([{"role": "user", "content": "hi"}], model="claude-3-opus")
    assert res["choices"][0]["message"]["content"] == "Claude 3 Opus response"
    assert res["model"] == "claude-3-opus-20240229"


@pytest.mark.asyncio
async def test_provider_router_heuristics_for_new_models():
    router = ProviderRouter()

    # Manually configure test keys on providers to test routing resolution
    router.providers["kimi"].api_key = "test-kimi-key"
    router.providers["openai"].api_key = "test-openai-key"
    router.providers["anthropic"].api_key = "test-anthropic-key"

    # 1. Kimi / Moonshot routing
    prov_kimi = await router.resolve_provider_for_model("kimi-k1.5")
    assert prov_kimi.name == "kimi"

    prov_moonshot = await router.resolve_provider_for_model("moonshot-v1-32k")
    assert prov_moonshot.name == "kimi"

    # 2. ChatGPT routing
    prov_chatgpt = await router.resolve_provider_for_model("chatgpt-4o-latest")
    assert prov_chatgpt.name == "openai"

    prov_o1 = await router.resolve_provider_for_model("o1-mini")
    assert prov_o1.name == "openai"

    # 3. Claude routing
    prov_claude = await router.resolve_provider_for_model("claude-3-opus-20240229")
    assert prov_claude.name == "anthropic"

    # 4. OpenCode / Coder with Ollama offline must fail loudly — public requests
    # must never degrade silently to MockProvider. The error carries an
    # actionable "start Ollama / pull" instruction.
    async def mock_ollama_offline():
        return {"connected": False, "status": "offline"}

    router.providers["ollama"].health = mock_ollama_offline
    with pytest.raises(ValueError, match="Ollama"):
        await router.resolve_provider_for_model("opencodeinterpreter:6.7b")

    # 5. OpenCode / Coder routing with mock Ollama connected
    async def mock_ollama_online():
        return {"connected": True, "status": "online"}

    class _FakeInstalledModel:
        id = "qwen2.5-coder:7b"

    async def mock_list_installed():
        return [_FakeInstalledModel()]

    router.providers["ollama"].health = mock_ollama_online
    router.providers["ollama"].list_models = mock_list_installed
    prov_qwen_coder = await router.resolve_provider_for_model("qwen2.5-coder:7b")
    assert prov_qwen_coder.name == "ollama"

    # An Ollama model that is reachable but not downloaded fails fast with an
    # actionable pull instruction instead of a confusing runtime 404.
    with pytest.raises(ValueError, match="ollama pull"):
        await router.resolve_provider_for_model("phi3.5:3.8b")


def test_registry_contains_new_models():
    registry = get_default_registry()

    # OpenCode models
    assert registry.get("opencodeinterpreter:6.7b") is not None
    assert registry.get("opencodeinterpreter:33b") is not None
    assert registry.get("qwen2.5-coder:7b") is not None
    assert registry.get("deepseek-coder:6.7b") is not None

    # ChatGPT models
    assert registry.get("gpt-4o") is not None
    assert registry.get("gpt-4o-mini") is not None
    assert registry.get("chatgpt-4o-latest") is not None
    assert registry.get("o1") is not None
    assert registry.get("o1-mini") is not None

    # Claude models
    assert registry.get("claude-3-5-sonnet-20241022") is not None
    assert registry.get("claude-3-5-haiku-20241022") is not None
    assert registry.get("claude-3-opus-20240229") is not None

    # Kimi models
    assert registry.get("kimi-k1.5") is not None
    assert registry.get("moonshot-v1-8k") is not None
    assert registry.get("moonshot-v1-32k") is not None
    assert registry.get("moonshot-v1-128k") is not None


def test_cost_calculation_for_new_models():
    # OpenCode is zero-cost
    opencode_cost = calculate_cost("opencodeinterpreter:6.7b", 1000, 500)
    assert opencode_cost["is_free"] is True
    assert opencode_cost["total_cost_usd"] == 0.0

    qwen_cost = calculate_cost("qwen2.5-coder:7b", 2000, 800)
    assert qwen_cost["is_free"] is True
    assert qwen_cost["total_cost_usd"] == 0.0

    # ChatGPT commercial cost
    chatgpt_cost = calculate_cost("chatgpt-4o-latest", 1000, 1000)
    assert chatgpt_cost["is_free"] is False
    assert chatgpt_cost["total_cost_usd"] > 0.0

    # Claude Opus commercial cost
    opus_cost = calculate_cost("claude-3-opus-20240229", 1000, 1000)
    assert opus_cost["is_free"] is False
    assert opus_cost["total_cost_usd"] > 0.0

    # Kimi commercial cost
    kimi_cost = calculate_cost("kimi-k1.5", 1000, 1000)
    assert kimi_cost["is_free"] is False
    assert kimi_cost["total_cost_usd"] > 0.0
