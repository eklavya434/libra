"""
Tests for guest session identity isolation and public-safety gates.

Coverage:
- Session token minting and resolution (/api/v1/auth/session).
- Per-session conversation isolation (unknown session cannot read/write
  another session's conversations; no silent 500 on foreign ids).
- Legacy owner-less conversations remain readable for backward compatibility.
- Notebook code execution is disabled by default on shared deployments.
"""

import pytest
from fastapi.testclient import TestClient

import packages.core.memory.sqlite_store as sqlite_module
from apps.backend.main import app
from packages.core.memory import SQLiteConversationStore

BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


@pytest.fixture(autouse=True)
def isolated_memory_store(monkeypatch):
    """Fresh in-memory SQLite store per test."""
    test_store = SQLiteConversationStore(db_path=":memory:")
    monkeypatch.setattr(sqlite_module, "_global_store", test_store)
    return test_store


@pytest.fixture
def client():
    # Server exceptions are handled by Libra's JSON 500 handler; make the test
    # client observe the response instead of re-raising into the test.
    return TestClient(app, raise_server_exceptions=False)


def _mint(client) -> str:
    """Use the session endpoint to mint a fresh guest token."""
    res = client.get("/api/v1/auth/session")
    assert res.status_code == 200
    token = res.json().get("token")
    assert token, "expected a fresh session token to be minted"
    return token


def test_session_mint_and_resolve(client):
    # No header + no browser UA still mints via /auth/session
    res = client.get("/api/v1/auth/session")
    body = res.json()
    assert body["guest"] is True
    assert "expires_at" in body

    # The raw token is echoed on the response header too
    header_token = res.headers.get("X-Libra-Session")
    assert body.get("token") == header_token

    # Resolving the same token returns info and does not mint a new one
    resolved = client.get("/api/v1/auth/session", headers={"X-Libra-Session": header_token}).json()
    assert resolved["session_id"] == body["session_id"]
    assert "token" not in resolved


def test_missing_session_header_uses_public_scope(client):
    # Without a session header (and without minting via /auth/session), the
    # request deterministically runs in the shared public scope.
    res = client.get("/api/v1/models", headers={"user-agent": BROWSER_UA})
    assert res.status_code == 200
    assert res.headers.get("X-Libra-Session") is None


def test_conversations_are_isolated_per_session(client):
    token_a = _mint(client)
    token_b = _mint(client)

    # Session A creates a conversation
    created = client.post(
        "/api/v1/conversations",
        headers={"X-Libra-Session": token_a},
        json={"title": "Secret of A"},
    )
    assert created.status_code == 200
    conv_a = created.json()["id"]

    # Session B cannot read it
    assert (
        client.get(
            f"/api/v1/conversations/{conv_a}",
            headers={"X-Libra-Session": token_b},
        ).status_code
        == 404
    )
    # Session B's list does not include it
    list_b = client.get("/api/v1/conversations", headers={"X-Libra-Session": token_b}).json()
    assert all(c["id"] != conv_a for c in list_b)
    # Session B cannot modify it
    assert (
        client.patch(
            f"/api/v1/conversations/{conv_a}",
            headers={"X-Libra-Session": token_b},
            json={"title": "hijack"},
        ).status_code
        == 404
    )
    assert (
        client.delete(
            f"/api/v1/conversations/{conv_a}",
            headers={"X-Libra-Session": token_b},
        ).status_code
        == 404
    )
    # Session A can still read it
    assert (
        client.get(
            f"/api/v1/conversations/{conv_a}",
            headers={"X-Libra-Session": token_a},
        ).status_code
        == 200
    )


def test_chat_with_foreign_conversation_id_returns_403(client):
    token_a = _mint(client)
    token_b = _mint(client)

    conv = client.post(
        "/api/v1/conversations",
        headers={"X-Libra-Session": token_a},
        json={"title": "A-only"},
    ).json()

    res = client.post(
        "/api/v1/chat/completions",
        headers={"X-Libra-Session": token_b},
        json={
            "conversation_id": conv["id"],
            "model": "libra-mock-v1",
            "messages": [{"role": "user", "content": "Should not be allowed"}],
            "stream": False,
        },
    )
    assert res.status_code == 403
    assert "not accessible" in res.json().get("detail", "")


def test_legacy_ownerless_conversations_stay_shared(client):
    token = _mint(client)
    # Simulate a pre-multi-user row: no owner.
    store = sqlite_module.get_conversation_store()
    legacy = store.create_conversation(title="Legacy Row", owner_id=None)

    # Any session can read AND delete a legacy row (documented compatibility).
    listed = client.get("/api/v1/conversations", headers={"X-Libra-Session": token}).json()
    assert any(c["id"] == legacy.id for c in listed)
    assert (
        client.get(
            f"/api/v1/conversations/{legacy.id}",
            headers={"X-Libra-Session": token},
        ).status_code
        == 200
    )


def test_invalid_session_token_triggers_fresh_mint_on_bootstrap(client):
    res = client.get("/api/v1/auth/session", headers={"X-Libra-Session": "deadbeef" * 8})
    # An invalid header is ignored and a fresh token is minted.
    assert res.status_code == 200
    assert "token" in res.json()
    assert res.json()["guest"] is True


def test_notebook_code_execution_gated_off_by_default(client):
    res = client.post(
        "/api/v1/notebook/sessions/demo/execute",
        json={"code": "print('hello')"},
    )
    assert res.status_code == 503
    # Exact mission message: secure sandbox infrastructure is not available.
    assert "secure sandbox infrastructure is not available" in res.json()["detail"]


def test_exception_handler_never_leaks_stack_traces(client, monkeypatch):
    # Trigger an unhandled exception inside a route via a patched provider.
    def boom(*args, **kwargs):
        raise RuntimeError("super-secret-internal-detail")

    import apps.backend.api.v1.endpoints.chat as chat_endpoints

    store = sqlite_module.get_conversation_store()
    token = _mint(client)
    conv = store.create_conversation(title="x", owner_id="public")

    monkeypatch.setattr(chat_endpoints, "get_router", boom)
    res = client.post(
        "/api/v1/chat/completions",
        headers={"X-Libra-Session": token},
        json={
            "conversation_id": conv.id,
            "model": "libra-mock-v1",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    assert res.status_code == 500
    body = res.json()
    assert "super-secret-internal-detail" not in res.text
    assert body["detail"] == "Internal server error"
    assert "request_id" in body


# ---------------------------------------------------------------------------
# Optional Supabase JWT authentication (opt-in; guests remain the default)
# ---------------------------------------------------------------------------


def _sign_supabase_jwt(payload: dict, secret: str = "test-jwt-secret") -> str:
    import base64
    import hashlib
    import hmac
    import json

    def b64(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    header = b64(b'{"alg":"HS256","typ":"JWT"}')
    body = b64(json.dumps(payload, separators=(",", ":")).encode("ascii"))
    sig = b64(
        hmac.new(
            secret.encode("utf-8"), f"{header}.{body}".encode("ascii"), hashlib.sha256
        ).digest()
    )
    return f"{header}.{body}.{sig}"


@pytest.fixture
def supabase_auth_enabled(monkeypatch):
    """Turn on Supabase JWT auth for the identity middleware."""
    from apps.backend.middleware import identity as identity_module

    monkeypatch.setattr(identity_module.settings, "supabase_auth_enabled", True)
    monkeypatch.setattr(identity_module.settings, "supabase_jwt_secret", "test-jwt-secret")
    monkeypatch.setattr(identity_module.settings, "supabase_jwt_audience", "authenticated")
    return identity_module


def _valid_jwt(sub: str) -> str:
    import time

    return _sign_supabase_jwt(
        {
            "sub": sub,
            "aud": "authenticated",
            "iat": int(time.time()) - 60,
            "exp": int(time.time()) + 3600,
        }
    )


def test_supabase_jwt_identity_is_scoped_per_user(client, supabase_auth_enabled):
    alice = _valid_jwt("alice-1")
    bob = _valid_jwt("bob-2")
    headers_a = {"Authorization": f"Bearer {alice}"}
    headers_b = {"Authorization": f"Bearer {bob}"}

    created = client.post(
        "/api/v1/conversations",
        headers=headers_a,
        json={"title": "Alice Secret", "model": "libra-mock-v1"},
    )
    assert created.status_code == 200
    conv_id = created.json()["id"]

    # Alice sees her conversation...
    alice_list = client.get("/api/v1/conversations", headers=headers_a).json()
    assert any(c["id"] == conv_id for c in alice_list)

    # ...Bob does not (per-user JWT scoping).
    bob_list = client.get("/api/v1/conversations", headers=headers_b).json()
    assert not any(c["id"] == conv_id for c in bob_list)


def test_supabase_invalid_jwt_falls_back_to_public_scope(client, supabase_auth_enabled):
    bob = _valid_jwt("bob-2")
    bob_conv = client.post(
        "/api/v1/conversations",
        headers={"Authorization": f"Bearer {bob}"},
        json={"title": "Bob", "model": "libra-mock-v1"},
    )
    assert bob_conv.status_code == 200
    bob_id = bob_conv.json()["id"]

    bad = client.get(
        "/api/v1/conversations", headers={"Authorization": "Bearer not-a-real-jwt"}
    ).json()
    assert not any(c["id"] == bob_id for c in bad)


def test_supabase_auth_disabled_ignores_bearer(client, supabase_auth_enabled, monkeypatch):
    from apps.backend.middleware import identity as identity_module

    # Turn the feature back off mid-test: bearer tokens must be ignored entirely.
    monkeypatch.setattr(identity_module.settings, "supabase_auth_enabled", False)
    token = _valid_jwt("anyone")
    created = client.post(
        "/api/v1/conversations",
        headers={"Authorization": f"Bearer {token}"},
        json={"title": "Guest Scoped", "model": "libra-mock-v1"},
    )
    assert created.status_code == 200
    conv_id = created.json()["id"]

    # Without the header the same conversation is visible => it was treated as guest scope.
    guest_list = client.get("/api/v1/conversations").json()
    assert any(c["id"] == conv_id for c in guest_list)
