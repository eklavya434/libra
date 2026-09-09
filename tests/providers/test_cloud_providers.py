"""
Unit tests for Libra Cloud Provider Adapters (Phase 9).
Verifies OpenAI, Gemini, Claude, Groq, and OpenRouter adapters using mock HTTP transports.
No real external calls or paid API charges.
"""

from __future__ import annotations

import httpx
import pytest

from packages.providers.anthropic import AnthropicProvider
from packages.providers.errors import ProviderAuthenticationError, ProviderRateLimitError
from packages.providers.gemini import GeminiProvider
from packages.providers.groq import GroqProvider
from packages.providers.openai import OpenAIProvider


@pytest.mark.asyncio
async def test_openai_chat_mock_transport():
    """Verify OpenAIProvider handles standard chat completion JSON response."""

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" in request.headers
        assert request.headers["authorization"] == "Bearer test-openai-key"
        payload = {
            "id": "chatcmpl-test-123",
            "choices": [{"message": {"role": "assistant", "content": "Hello from OpenAI!"}}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 6, "total_tokens": 21},
        }
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    provider = OpenAIProvider(api_key="test-openai-key", http_client=client)

    result = await provider.chat([{"role": "user", "content": "Say hello"}], model="gpt-4o-mini")
    assert result["choices"][0]["message"]["content"] == "Hello from OpenAI!"
    assert result["cost"]["prompt_tokens"] == 15
    assert result["cost"]["total_cost_usd"] > 0.0


@pytest.mark.asyncio
async def test_openai_auth_error_mock():
    """Verify OpenAIProvider converts 401 response into ProviderAuthenticationError."""

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Invalid API key"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    provider = OpenAIProvider(api_key="invalid-key", http_client=client)

    with pytest.raises(ProviderAuthenticationError) as exc_info:
        await provider.chat([{"role": "user", "content": "Test"}], model="gpt-4o")
    assert exc_info.value.status_code == 401
    assert "Invalid API key" in exc_info.value.message


@pytest.mark.asyncio
async def test_gemini_chat_mock_transport():
    """Verify GeminiProvider parses Gemini API candidate format."""

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        assert "key=test-gemini-key" in str(request.url)
        payload = {
            "candidates": [{"content": {"parts": [{"text": "Greetings from Gemini!"}]}}],
            "usageMetadata": {
                "promptTokenCount": 12,
                "candidatesTokenCount": 5,
                "totalTokenCount": 17,
            },
        }
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    provider = GeminiProvider(api_key="test-gemini-key", http_client=client)

    result = await provider.chat(
        [{"role": "user", "content": "Say hello"}], model="gemini-1.5-flash"
    )
    assert result["choices"][0]["message"]["content"] == "Greetings from Gemini!"
    assert result["cost"]["prompt_tokens"] == 12


@pytest.mark.asyncio
async def test_anthropic_chat_mock_transport():
    """Verify AnthropicProvider parses Claude Messages API format."""

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "test-claude-key"
        payload = {
            "id": "msg-test-123",
            "content": [{"type": "text", "text": "Claude responds here."}],
            "usage": {"input_tokens": 14, "output_tokens": 8},
        }
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    provider = AnthropicProvider(api_key="test-claude-key", http_client=client)

    result = await provider.chat(
        [{"role": "user", "content": "Hi Claude"}], model="claude-3-5-haiku-20241022"
    )
    assert result["choices"][0]["message"]["content"] == "Claude responds here."
    assert result["cost"]["prompt_tokens"] == 14
    assert result["cost"]["completion_tokens"] == 8


@pytest.mark.asyncio
async def test_groq_rate_limit_error_mock():
    """Verify GroqProvider converts 429 response into ProviderRateLimitError."""

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "TPM rate limit exceeded"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    provider = GroqProvider(api_key="test-groq-key", http_client=client)

    with pytest.raises(ProviderRateLimitError) as exc_info:
        await provider.chat(
            [{"role": "user", "content": "Test prompt"}], model="llama-3.1-8b-instant"
        )
    assert exc_info.value.status_code == 429
