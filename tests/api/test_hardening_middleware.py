"""
Hardening regressions for Batch 4/5/9: error disclosure, security headers,
chunked payload enforcement, request-id sanitization, and sliding session TTL.
"""

import json

from fastapi.testclient import TestClient

from apps.backend.main import app
from apps.backend.middleware.security import RequestSizeLimiterMiddleware

client = TestClient(app)


def _chunked_http_scope(path: str = "/api/v1/noop") -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }


def _chunked_receive(chunks: list[bytes]):
    iterator = iter(chunks)

    async def receive():
        try:
            chunk = next(iterator)
            return {"type": "http.request", "body": chunk, "more_body": True}
        except StopIteration:
            return {"type": "http.request", "body": b"", "more_body": False}

    return receive


async def _run_asgi(middleware, scope, receive):
    messages = []

    async def send(message):
        messages.append(message)

    await middleware(scope, receive, send)
    return messages


def _status_and_body(messages):
    start = next(m for m in messages if m["type"] == "http.response.start")
    bodies = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return start["status"], bodies


def test_chunked_body_over_limit_rejected():
    async def downstream(scope, receive, send):
        raise AssertionError("downstream must not run for an oversized chunked body")

    guard = RequestSizeLimiterMiddleware(downstream, max_bytes=100)
    chunks = [b"x" * 40] * 10  # 400 bytes (>100) with no content-length header

    scope = _chunked_http_scope()
    messages = _run_asgi_middleware(guard, scope, _chunked_receive(chunks))
    status, body = _status_and_body(messages)
    assert status == 413
    assert b"Payload too large" in body


def test_chunked_body_under_limit_replayed_to_app():
    async def downstream(scope, receive, send):
        # The app must see the full body even though it was streamed.
        payload = bytearray()
        while True:
            message = await receive()
            if message.get("type") == "http.request":
                payload.extend(message.get("body", b""))
                if not message.get("more_body", False):
                    break
            elif message.get("type") == "http.disconnect":
                break
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send(
            {"type": "http.response.body", "body": json.dumps({"len": len(payload)}).encode()}
        )

    guard = RequestSizeLimiterMiddleware(downstream, max_bytes=100)
    chunks = [b"ab" * 10] * 3  # 60 bytes total — under the 100 byte ceiling

    messages = _run_asgi_middleware(guard, _chunked_http_scope(), _chunked_receive(chunks))
    status, body = _status_and_body(messages)
    assert status == 200
    assert json.loads(body.decode()) == {"len": 60}


def test_content_length_over_limit_rejected():
    async def downstream(scope, receive, send):
        raise AssertionError("downstream must not run with an oversized content-length")

    guard = RequestSizeLimiterMiddleware(downstream, max_bytes=100)
    scope = {
        **_chunked_http_scope(),
        "headers": [(b"host", b"testserver"), (b"content-length", b"500")],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    messages = _run_asgi_middleware(guard, scope, receive)
    status, _ = _status_and_body(messages)
    assert status == 413


def test_request_id_sanitized_when_client_supplies_junk():
    junk = "x" * 200
    response = client.get("/api/v1/health", headers={"X-Request-Id": junk})
    assert response.status_code == 200
    issued = response.headers.get("X-Request-Id", "")
    assert issued != junk
    assert len(issued) == 16  # generated replacement

    response = client.get("/api/v1/health", headers={"X-Request-Id": "my.const-123/abc_de"})
    assert response.headers.get("X-Request-Id") == "my.const-123/abc_de"


def test_security_headers_include_csp_and_hsts_over_https():
    response = client.get("/api/v1/health", headers={"X-Forwarded-Proto": "https"})
    assert (
        response.headers.get("Content-Security-Policy")
        == "default-src 'none'; frame-ancestors 'none'"
    )
    assert (
        response.headers.get("Strict-Transport-Security") == "max-age=63072000; includeSubDomains"
    )

    response = client.get("/api/v1/health", headers={"X-Forwarded-Proto": "http"})
    assert "Strict-Transport-Security" not in response.headers


def test_cors_exposes_libra_headers():
    origin = "https://libra-frontend.vercel.app"
    response = client.get(
        "/api/v1/health",
        headers={"Origin": origin},
    )
    assert response.headers.get("access-control-allow-origin") == origin
    exposed = response.headers.get("access-control-expose-headers", "")
    assert "X-Request-Id" in exposed
    assert "X-Libra-Session" in exposed
    assert "Retry-After" in exposed


def test_readyz_degraded_reflects_store_backend(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://configured-but-unreachable/db")
    from packages.core.memory.sqlite_store import SQLiteConversationStore

    def fake_store():
        return SQLiteConversationStore(db_path=":memory:")

    import apps.backend.main as main_module

    monkeypatch.setattr(main_module, "get_conversation_store", fake_store)
    response = client.get("/readyz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["store_backend"] == "SQLiteConversationStore"
    assert data["database_url_configured"] is True


def test_readyz_never_leaks_exception_details(monkeypatch):
    def broken_store():
        raise RuntimeError("custom-secret-detail-xyz")

    import apps.backend.main as main_module

    monkeypatch.setattr(main_module, "get_conversation_store", broken_store)
    response = client.get("/readyz")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert "custom-secret-detail-xyz" not in json.dumps(data)


def test_log_redaction_formatter_scrubs_secrets():
    import logging

    from apps.backend.core.logging import RedactionFormatter

    formatter = RedactionFormatter("%(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="api_key=AIzaSySecretValue1234567890 connection ok",
        args=(),
        exc_info=None,
    )
    rendered = formatter.format(record)
    assert "AIzaSySecretValue1234567890" not in rendered
    assert "********" in rendered

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='payload={"Authorization": "Bearer sk-proj-SuperSecretTokenAbCdEfGh"}',
        args=(),
        exc_info=None,
    )
    rendered = formatter.format(record)
    assert "sk-proj-SuperSecretTokenAbCdEfGh" not in rendered


def _run_asgi_middleware(middleware, scope, receive):
    import asyncio

    return asyncio.run(_run_asgi(middleware, scope, receive))
