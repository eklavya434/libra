"""
Unit tests for Provider abstraction and MockProvider.
"""

import pytest

from packages.providers.base import ModelMetadata
from packages.providers.mock import MockProvider


@pytest.mark.asyncio
async def test_mock_provider_list_models():
    provider = MockProvider()
    assert provider.name == "mock-provider"

    models = await provider.list_models()
    assert len(models) >= 1
    m = models[0]
    assert isinstance(m, ModelMetadata)
    assert m.id == "libra-mock-v1"
    assert m.is_local is True
    assert m.requires_gpu is False


@pytest.mark.asyncio
async def test_mock_provider_chat():
    provider = MockProvider()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what a token is in simple terms."},
    ]
    response = await provider.chat(messages=messages, model="libra-mock-v1")
    assert "choices" in response
    assert len(response["choices"]) == 1
    assert "content" in response["choices"][0]["message"]
    assert "MockResponse" in response["choices"][0]["message"]["content"]
    assert "usage" in response


@pytest.mark.asyncio
async def test_mock_provider_stream():
    provider = MockProvider()
    messages = [{"role": "user", "content": "Hello"}]
    chunks = []
    async for token in provider.stream(messages=messages, model="libra-mock-v1"):
        chunks.append(token)
    assert len(chunks) > 0
    full_text = "".join(chunks)
    assert "Hello from Libra!" in full_text


@pytest.mark.asyncio
async def test_mock_provider_health_and_capabilities():
    provider = MockProvider()
    health = await provider.health()
    assert health["status"] == "healthy"
    assert health["is_mock"] is True

    caps = provider.capabilities()
    assert caps["supports_text"] is True

    embeddings = await provider.embeddings(["test text 1", "test text 2"])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 16
