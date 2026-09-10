"""
Tests for Phase 50 Chat Stabilization, Persistence, and Router Hardening
"""

import json

import pytest
from fastapi.testclient import TestClient

from apps.backend.main import app
from packages.core.memory import get_conversation_store
from packages.providers.gemini import GeminiProvider
from packages.providers.router import get_router

client = TestClient(app)


def test_chat_auto_generates_and_persists_conversation():
    """Verify that omitting conversation_id auto-assigns one and saves both turns."""
    payload = {
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "model": "libra-mock-v1",
        "stream": False,
    }
    resp = client.post("/api/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "conversation_id" in data
    conv_id = data["conversation_id"]
    assert conv_id.startswith("conv-")

    # Verify conversation exists in DB with both user and assistant turns
    store = get_conversation_store()
    conv = store.get_conversation(conv_id)
    assert conv is not None
    assert "capital of France" in conv.title or conv.title == "What is the capital of France?"

    messages = store.get_messages(conv_id)
    assert len(messages) >= 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"


def test_chat_streaming_includes_conversation_metadata():
    """Verify SSE streaming yields conversation metadata and X-Conversation-Id header."""
    payload = {
        "messages": [{"role": "user", "content": "Say hi!"}],
        "model": "libra-mock-v1",
        "stream": True,
    }
    resp = client.post("/api/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "x-conversation-id" in resp.headers
    conv_id = resp.headers["x-conversation-id"]

    # Verify first data event contains conversation_id
    lines = [l for l in resp.text.strip().split("\n\n") if l.startswith("data: ")]
    assert len(lines) >= 2
    first_chunk = json.loads(lines[0][6:])
    assert first_chunk.get("conversation_id") == conv_id

    # Verify assistant message was saved to SQLite WAL
    store = get_conversation_store()
    messages = store.get_messages(conv_id)
    assert len(messages) >= 2
    assert messages[-1].role == "assistant"
    assert len(messages[-1].content) > 0


def test_gemini_model_normalization():
    """Verify Gemini provider normalizes aliases to valid current models."""
    prov = GeminiProvider(api_key="mock_key")
    assert prov._normalize_model("gemini") == "gemini-2.5-flash"
    assert prov._normalize_model("gemini-1.5-flash") == "gemini-2.5-flash"
    assert prov._normalize_model("gemini-flash") == "gemini-2.5-flash"
    assert prov._normalize_model("gemini-1.5-pro") == "gemini-2.5-pro"
    assert prov._normalize_model("models/gemini-2.5-flash") == "gemini-2.5-flash"


def test_router_raises_explicit_error_for_unconfigured_cloud_provider():
    """Verify router refuses to silently fake cloud models when API key is missing."""
    router = get_router()
    # Temporarily remove OpenAI key if present
    openai_prov = router.get_provider("openai")
    orig_key = getattr(openai_prov, "api_key", None)
    try:
        openai_prov.api_key = ""
        with pytest.raises(ValueError) as excinfo:
            import asyncio

            asyncio.run(router.resolve_provider_for_model("gpt-4o"))
        assert "OPENAI_API_KEY is not configured" in str(excinfo.value)
    finally:
        openai_prov.api_key = orig_key
