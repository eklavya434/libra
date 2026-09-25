"""
Libra API v1 - Guest Session Endpoints

Bootstraps the anonymous session identity used for per-visitor conversation
isolation. Sessions are opaque high-entropy tokens; only their SHA-256 hash is
stored server-side.
"""

from __future__ import annotations

import json
import secrets

from fastapi import APIRouter, Request, Response

from packages.core.memory import get_conversation_store

router = APIRouter()


@router.get("/session", summary="Resolve or mint a guest session")
async def get_session(request: Request) -> Response:
    """Return session info. Mints a fresh session when the caller has none.

    The raw token is delivered in the JSON body (and echoed on the
    ``X-Libra-Session`` response header) so the frontend can persist it and
    send it on subsequent requests.
    """
    store = get_conversation_store()
    raw_token = request.headers.get("X-Libra-Session")
    if raw_token:
        info = store.get_session(raw_token)
        if info:
            return Response(
                status_code=200,
                media_type="application/json",
                content=json.dumps(
                    {
                        "session_id": info["session_id"],
                        "guest": True,
                        "expires_at": info["expires_at"],
                    }
                ),
            )
    # No valid session: mint one.
    token = secrets.token_hex(32)
    store.create_session(token, ttl_days=request.state.session_ttl_days)
    info = store.get_session(token)
    if info is None:
        raise RuntimeError("Failed to persist session")
    body = json.dumps(
        {
            "session_id": info["session_id"],
            "token": token,
            "guest": True,
            "expires_at": info["expires_at"],
        }
    )
    return Response(
        status_code=200,
        media_type="application/json",
        headers={"X-Libra-Session": token},
        content=body,
    )


@router.post("/session", summary="Resolve or mint a guest session (idempotent)")
async def ensure_session(request: Request) -> Response:
    """Same as GET /session but POST, for clients that require a body."""
    return await get_session(request)


@router.post("/logout", summary="Invalidate the current guest session")
async def logout(request: Request) -> dict:
    """Delete the session token so a future request starts as a fresh guest."""
    store = get_conversation_store()
    raw_token = request.headers.get("X-Libra-Session")
    if raw_token:
        store.delete_session(raw_token)
    return {"status": "ok"}
