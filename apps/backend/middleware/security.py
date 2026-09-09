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

from packages.core.security.rate_limiter import TokenBucketRateLimiter

# Shared global rate limiter instance
_global_rate_limiter = TokenBucketRateLimiter(requests_per_minute=120, burst_capacity=30)


def get_global_rate_limiter() -> TokenBucketRateLimiter:
    return _global_rate_limiter


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Enforces essential OWASP security response headers."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response


class RequestSizeLimiterMiddleware(BaseHTTPMiddleware):
    """Enforces maximum incoming payload size (default 10 MB)."""

    def __init__(self, app, max_bytes: int = 10 * 1024 * 1024) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self.max_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": f"Payload too large. Maximum allowed size is {self.max_bytes // (1024 * 1024)} MB."
                        },
                    )
            except ValueError:
                pass
        return await call_next(request)


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Token-bucket client IP rate limiter on sensitive API endpoints."""

    def __init__(self, app, rate_limiter: TokenBucketRateLimiter | None = None) -> None:
        super().__init__(app)
        self.rate_limiter = rate_limiter or _global_rate_limiter

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Rate limit only state-altering or compute-heavy endpoints
        path = request.url.path
        if (
            path.startswith("/api/v1/chat")
            or path.startswith("/api/v1/agents")
            or path.startswith("/api/v1/security")
        ):
            client_ip = request.headers.get("x-forwarded-for") or (
                request.client.host if request.client else "127.0.0.1"
            )
            client_ip = client_ip.split(",")[0].strip()

            info = self.rate_limiter.check(client_ip)
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
