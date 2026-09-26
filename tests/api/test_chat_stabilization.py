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


def test_regenerate_does_not_duplicate_user_message():
    """Regenerate re-sends the same tail user message; it must not be stored twice."""
    payload = {
        "messages": [{"role": "user", "content": "Explain attention succinctly."}],
        "model": "libra-mock-v1",
        "stream": True,
    }
    resp = client.post("/api/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    conv_id = resp.headers["x-conversation-id"]

    store = get_conversation_store()
    assert store.get_messages(conv_id)[0].role == "user"

    # Simulate the Regenerate button: same user message re-sent as full history tail
    regen_payload = {
        "messages": [{"role": "user", "content": "Explain attention succinctly."}],
        "model": "libra-mock-v1",
        "stream": True,
        "conversation_id": conv_id,
    }
    regen_resp = client.post("/api/v1/chat/completions", json=regen_payload)
    assert regen_resp.status_code == 200

    messages = store.get_messages(conv_id)
    user_msgs = [m for m in messages if m.role == "user"]
    assert len(user_msgs) == 1, f"regenerate duplicated the user message: {len(user_msgs)}"
    # Old answer replaced with new regenerated answer (no duplicate assistant responses)
    assert [m.role for m in messages] == ["user", "assistant"]


def test_regenerate_dedupes_only_identical_tail_user_turn():
    """A genuinely new user turn is still appended after a regenerate."""
    payload = {
        "messages": [{"role": "user", "content": "First prompt in thread."}],
        "model": "libra-mock-v1",
        "stream": True,
    }
    resp = client.post("/api/v1/chat/completions", json=payload)
    conv_id = resp.headers["x-conversation-id"]

    client.post(
        "/api/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "First prompt in thread."}],
            "model": "libra-mock-v1",
            "stream": True,
            "conversation_id": conv_id,
        },
    )

    client.post(
        "/api/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Second prompt in thread."}],
            "model": "libra-mock-v1",
            "stream": True,
            "conversation_id": conv_id,
        },
    )

    store = get_conversation_store()
    messages = store.get_messages(conv_id)
    roles = [m.role for m in messages]
    assert roles.count("user") == 2, f"expected 2 user messages, got {roles}"
    assert roles.count("assistant") == 2, f"expected 2 assistant messages, got {roles}"
    assert roles == ["user", "assistant", "user", "assistant"]


class FakeGeminiResponse:
    def __init__(self, status_code: int, body: bytes):
        self.status_code = status_code
        self.text = body.decode("utf-8", errors="ignore")
        self._json = json.loads(body)

    def json(self):
        return self._json

    async def aread(self):
        return self.text.encode()


class FakeGeminiHTTPClient:
    """Records requested model URLs; always returns a minimal success payload."""

    def __init__(self):
        self.requests: list[str] = []
        self.timeout = 5.0

    async def post(self, url: str, json=None) -> FakeGeminiResponse:
        self.requests.append(url)
        body = (
            b'{"candidates":[{"content":{"parts":[{"text":"hi"}]}}],'
            b'"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1}}'
        )
        return FakeGeminiResponse(200, body)


def test_gemini_chat_never_requests_invalid_fallback_models():
    """The chat path must only try valid Gemini model IDs, never stale aliases."""
    fake = FakeGeminiHTTPClient()
    prov = GeminiProvider(api_key="mock_key", http_client=fake)

    import asyncio

    asyncio.run(prov.chat(messages=[{"role": "user", "content": "hi"}], model="gemini-2.5-flash"))

    assert fake.requests, "expected at least one Gemini API request"
    requested = " ".join(fake.requests)
    assert "gemini-3.5-flash" not in requested
    assert "gemini-flash-latest" not in requested
    assert "/models/gemini-2.5-flash:" in requested


def test_repeated_user_prompt_never_duplicates_assistant_responses():
    """When a user sends identical consecutive turns ('hii' then 'hii'), both must be stored with alternating assistant turns."""
    from packages.core.memory import get_conversation_store

    store = get_conversation_store()

    # Create clean conversation
    conv = store.create_conversation(title="Repeat Turn Test", model="libra-mock-v1")

    # Turn 1: user says "hii"
    payload1 = {
        "conversation_id": conv.id,
        "model": "libra-mock-v1",
        "messages": [{"role": "user", "content": "hii"}],
        "stream": False,
    }
    r1 = client.post("/api/v1/chat/completions", json=payload1)
    assert r1.status_code == 200

    msgs1 = store.get_messages(conv.id)
    assert len(msgs1) == 2
    assert msgs1[0].role == "user" and msgs1[0].content == "hii"
    assert msgs1[1].role == "assistant"

    # Turn 2: user sends "hii" again
    payload2 = {
        "conversation_id": conv.id,
        "model": "libra-mock-v1",
        "messages": [
            {"role": "user", "content": "hii"},
            {"role": "assistant", "content": msgs1[1].content},
            {"role": "user", "content": "hii"},
        ],
        "stream": False,
    }
    r2 = client.post("/api/v1/chat/completions", json=payload2)
    assert r2.status_code == 200

    msgs2 = store.get_messages(conv.id)
    # Must have 4 messages in exact order: user -> assistant -> user -> assistant
    assert len(msgs2) == 4
    roles = [m.role for m in msgs2]
    assert roles == ["user", "assistant", "user", "assistant"], f"Roles were corrupted: {roles}"
    assert msgs2[2].role == "user" and msgs2[2].content == "hii"
    assert msgs2[3].role == "assistant"


def test_regenerate_replaces_last_assistant_response():
    """Regenerating a turn must replace the prior assistant response, avoiding duplicate assistant bubbles."""
    from packages.core.memory import get_conversation_store

    store = get_conversation_store()

    conv = store.create_conversation(title="Regenerate Test", model="libra-mock-v1")

    # Initial turn
    payload1 = {
        "conversation_id": conv.id,
        "model": "libra-mock-v1",
        "messages": [{"role": "user", "content": "Tell me a joke"}],
        "stream": False,
    }
    r1 = client.post("/api/v1/chat/completions", json=payload1)
    assert r1.status_code == 200

    msgs1 = store.get_messages(conv.id)
    assert len(msgs1) == 2
    old_assistant_id = msgs1[1].id

    # Regenerate turn: user message re-sent without the assistant reply
    payload_regen = {
        "conversation_id": conv.id,
        "model": "libra-mock-v1",
        "messages": [{"role": "user", "content": "Tell me a joke"}],
        "stream": False,
    }
    r_regen = client.post("/api/v1/chat/completions", json=payload_regen)
    assert r_regen.status_code == 200

    msgs_after = store.get_messages(conv.id)
    # Must STILL have exactly 2 messages (not 3), replacing the old assistant response
    assert len(msgs_after) == 2
    assert [m.role for m in msgs_after] == ["user", "assistant"]
    assert msgs_after[1].id != old_assistant_id


def test_models_default_endpoint():
    """The /api/v1/models/default endpoint returns a valid default model or unconfigured."""
    r = client.get("/api/v1/models/default")
    assert r.status_code == 200
    data = r.json()
    assert "default_model" in data
    assert data["default_model"] in (
        "gemini-2.5-flash",
        "libra-mock-v1",
        "llama3.2:1b",
        "gpt-4o-mini",
        None,
    )
    if data["default_model"] is None:
        assert data["configured"] is False
    else:
        assert data["configured"] is True
