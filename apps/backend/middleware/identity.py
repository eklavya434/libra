"""
Libra Backend - Session Identity Middleware

Resolves a session identity for every request.

Identity sources, in priority order:
1. ``Authorization: Bearer <jwt>``  — authenticated Supabase user, ONLY when
   ``LIBRA_SUPABASE_AUTH_ENABLED=true`` + ``SUPABASE_JWT_SECRET`` is set.
   Maps to ``session_id = supabase:<sub>``.
2. ``X-Libra-Session`` opaque guest token (minted by ``GET /api/v1/auth/session``,
   held in localStorage). Only its SHA-256 hash is persisted.
3. Fallback: shared ``"public"`` scope — deterministic, never a 401, keeps the
   anonymous product usable.

Sessions are ONLY minted by the explicit auth endpoint, never implicitly.
"""

from __future__ import annotations

import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from apps.backend.core.config import settings
from packages.core.memory import get_conversation_store
from packages.core.security.supabase_jwt import JWTValidationError, validate_access_token

logger = logging.getLogger(__name__)

SESSION_HEADER = "X-Libra-Session"
AUTHORIZATION_HEADER = "Authorization"
BEARER_PREFIX = "Bearer "
PUBLIC_SCOPE = "public"


def _supabase_auth_active() -> bool:
    """JWT auth is active only when explicitly enabled AND a secret is present."""
    return settings.supabase_auth_enabled and bool(settings.supabase_jwt_secret)


class SessionIdentityMiddleware(BaseHTTPMiddleware):
    """Attach session identity to ``request.state`` for all requests."""

    def __init__(self, app, ttl_days: int = 30) -> None:
        super().__init__(app)
        self.ttl_days = ttl_days

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        store = get_conversation_store()
        raw_token = request.headers.get(SESSION_HEADER)
        session_id = PUBLIC_SCOPE
        authenticated = False
        user_id: str | None = None

        if _supabase_auth_active():
            bearer = request.headers.get(AUTHORIZATION_HEADER, "")
            if bearer.startswith(BEARER_PREFIX):
                jwt_token = bearer[len(BEARER_PREFIX) :].strip()
                try:
                    payload = validate_access_token(
                        jwt_token,
                        jwt_secret=settings.supabase_jwt_secret,
                        audience=settings.supabase_jwt_audience,
                    )
                    subject = str(payload["sub"])
                    session_id = f"supabase:{subject}"
                    authenticated = True
                    user_id = subject
                except JWTValidationError:
                    # Invalid/expired token: behave like a guest, never a 401 wall.
                    logger.debug("Ignoring invalid Supabase JWT.")

        if session_id == PUBLIC_SCOPE and raw_token:
            info = store.get_session(raw_token)
            if info:
                session_id = info["session_id"]
            else:
                # Header present but invalid/expired: degrade to public scope.
                logger.debug("Ignoring invalid or expired session token")

        request.state.session_id = session_id
        request.state.session_token = raw_token
        request.state.session_ttl_days = self.ttl_days
        request.state.authenticated = authenticated
        request.state.user_id = user_id

        response = await call_next(request)
        return response
