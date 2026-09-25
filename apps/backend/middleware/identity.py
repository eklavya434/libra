"""
Libra Backend - Session Identity Middleware

Resolves a guest session identity for every request.

Identity source of truth: the ``X-Libra-Session`` header carrying an opaque
high-entropy token (minted by ``GET /api/v1/auth/session`` and held by the
frontend in localStorage). Only the SHA-256 hash of the token is persisted,
so a leaked database never exposes usable tokens.

Behaviour:
- Valid token  -> ``request.state.session_id`` is the session identity and
  ``request.state.session_token`` carries the raw token. last_seen refreshes.
- No/invalid/expired token -> request falls back to the shared ``"public"``
  scope (deterministic, never a 401, keeps the anonymous app usable).

Sessions are ONLY minted by the explicit auth endpoint, never implicitly.
"""

from __future__ import annotations

import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from packages.core.memory import get_conversation_store

logger = logging.getLogger(__name__)

SESSION_HEADER = "X-Libra-Session"
PUBLIC_SCOPE = "public"


class SessionIdentityMiddleware(BaseHTTPMiddleware):
    """Attach guest session identity to ``request.state`` for all requests."""

    def __init__(self, app, ttl_days: int = 30) -> None:
        super().__init__(app)
        self.ttl_days = ttl_days

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        store = get_conversation_store()
        raw_token = request.headers.get(SESSION_HEADER)
        session_id = PUBLIC_SCOPE

        if raw_token:
            info = store.get_session(raw_token)
            if info:
                session_id = info["session_id"]
            else:
                # Header present but invalid/expired: degrade to public scope.
                # (A stale localStorage token simply behaves like a new guest.)
                logger.debug("Ignoring invalid or expired session token")

        request.state.session_id = session_id
        request.state.session_token = raw_token
        request.state.session_ttl_days = self.ttl_days

        response = await call_next(request)
        return response
