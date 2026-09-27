"""
Libra Backend - Security & Defense Middleware

Provides:
1. SecurityHeadersMiddleware: Sets OWASP-recommended HTTP security headers.
2. RequestSizeLimiterMiddleware: Rejects oversized payloads to prevent memory exhaustion attacks.
3. RateLimitingMiddleware: Token-bucket throttling per client IP.
"""

from __future__ import annotations

from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from apps.backend.core.config import settings
from packages.core.security.rate_limiter import TokenBucketRateLimiter

# Shared global rate limiter instance (single source of truth for enforcement
# AND telemetry; consumed by RateLimitingMiddleware below and by the security
# stats endpoint). Uvicorn runs one process, so a process-wide bucket is correct.
_global_rate_limiter = TokenBucketRateLimiter(requests_per_minute=120, burst_capacity=30)


def get_global_rate_limiter() -> TokenBucketRateLimiter:
    return _global_rate_limiter


def reset_global_rate_limiter() -> None:
    """Testing hook: restores a full token bucket for every client.

    Each test deserves a full bucket regardless of scheduling; prod behavior is
    unaffected because this is never called outside the test suite.
    """
    _global_rate_limiter.reset()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Enforces essential OWASP security response headers."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        # HSTS is only meaningful when the client actually used HTTPS (the
        # Render/Vercel edges terminate TLS, so X-Forwarded-Proto carries it).
        if request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https":
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        return response


class RequestSizeLimiterMiddleware:
    """Enforces maximum incoming payload size (default 10 MB).

    Handles both ``Content-Length`` headers and chunked/streamed bodies, so an
    attacker cannot bypass the limit by omitting the length. Implemented as a
    plain ASGI middleware (not BaseHTTPMiddleware) so the bounded body can be
    replayed to the downstream app — with BaseHTTPMiddleware the app would
    receive an already-consumed stream and see an empty body.
    """

    def __init__(self, app, max_bytes: int = 10 * 1024 * 1024) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def _reject(self, send) -> None:
        body = (
            f'{{"detail": "Payload too large. Maximum allowed size is '
            f'{self.max_bytes // (1024 * 1024)} MB."}}'
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                pass
            await self.app(scope, receive, send)
            return

        # No content-length (chunked or proxy-stripped): buffer the stream but
        # hard-stop once the ceiling is crossed so memory stays bounded.
        payload = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            payload.extend(message.get("body", b""))
            if len(payload) > self.max_bytes:
                await self._reject(send)
                return
            if not message.get("more_body", False):
                break

        # Replay the fully-buffered body so application endpoints see it intact.
        body_bytes = bytes(payload)
        sent = False

        async def replayed_receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body_bytes, "more_body": False}
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replayed_receive, send)


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Token-bucket client IP rate limiter on compute-heavy API endpoints.

    Coverage is broader than the original chat-only guard: file upload, OCR,
    notebook/code execution, RAG indexing, batch jobs, tools, agents and
    dynamic routing are all state-changing or expensive.
    """

    # Path prefixes that are rate-limited per client IP.
    _LIMITED_PREFIXES = (
        "/api/v1/chat",
        "/api/v1/agents",
        "/api/v1/security",
        "/api/v1/upload",
        "/api/v1/files",
        "/api/v1/parse",
        "/api/v1/chunk",
        "/api/v1/qa",
        "/api/v1/document",
        "/api/v1/document_ocr",
        "/api/v1/ocr",
        "/api/v1/notebook",
        "/api/v1/coder",
        "/api/v1/rag",
        "/api/v1/batch",
        "/api/v1/tools",
        "/api/v1/routing",
        "/api/v1/multimodal",
        "/api/v1/capstone",
    )

    def __init__(
        self,
        app,
        requests_per_minute: int = 120,
        burst_capacity: int = 30,
    ) -> None:
        super().__init__(app)
        # Enforce with the single global limiter so the stats endpoint reports
        # the exact same state (one source of truth per process).
        self.rate_limiter = get_global_rate_limiter()
        if (
            self.rate_limiter.limit != requests_per_minute
            or self.rate_limiter.capacity != burst_capacity
        ):
            self.rate_limiter.limit = requests_per_minute
            self.rate_limiter.capacity = burst_capacity
            self.rate_limiter.replenish_rate = requests_per_minute / 60.0

    def _client_ip(self, request: Request) -> str:
        # Trust X-Forwarded-For only when a trusted proxy count is configured
        # (LIBRA_TRUSTED_PROXIES, default 1: the Render/Railway edge TLS terminator).
        # The edge overwrites XFF, so the first entry is the real client IP.
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded and settings.libra_trusted_proxies > 0:
            return forwarded.split(",")[0].strip()
        host = request.client.host if request.client else "127.0.0.1"
        # Starlette's test client appears as "testclient"; keep it bucketed alone.
        return "127.0.0.1" if host == "testclient" else host

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        if path.startswith(self._LIMITED_PREFIXES):
            info = self.rate_limiter.check(self._client_ip(request))
            if not info.allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Too many requests. Rate limit exceeded.",
                        "retry_after_seconds": info.retry_after_seconds,
                    },
                    headers={
                        "Retry-After": str(int(info.retry_after_seconds)),
                        "X-RateLimit-Limit": str(info.limit),
                        "X-RateLimit-Remaining": "0",
                    },
                )

        response = await call_next(request)
        return response
