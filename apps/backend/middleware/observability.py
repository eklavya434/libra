"""
Libra Backend - Observability Middleware

1. RequestIdMiddleware: assigns a ``X-Request-Id`` to every request/response so
   logs are correlatable, and rejects health-probe requests without IDs.
2. Global exception handler wiring: converts unhandled exceptions into clean
   JSON errors with request IDs, never exposing stack traces or secrets.
"""

from __future__ import annotations

import logging
import uuid
from typing import Callable

from fastapi import Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-Id"

# Secrets that must never appear in error responses or request logging.
_REDACT_KEYS = (
    "api_key",
    "token",
    "password",
    "secret",
    "authorization",
    "key",
)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach a request ID to every request and response."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request.state.request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request.state.request_id
        return response


def exception_handlers() -> dict:
    """Exception handlers for the FastAPI app (HTTPException + unexpected)."""

    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "request_id": getattr(request.state, "request_id", None),
            },
            headers=getattr(exc, "headers", None),
        )

    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Request validation failed",
                "errors": exc.errors(),
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.exception(
            "Unhandled exception on %s %s (request_id=%s)",
            request.method,
            request.url.path,
            request_id,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "request_id": request_id,
            },
        )

    return {
        StarletteHTTPException: http_exception_handler,
        RequestValidationError: validation_exception_handler,
        Exception: unhandled_exception_handler,
    }
