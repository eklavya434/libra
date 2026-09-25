"""
Libra Core Security - Supabase JWT Access-Token Validation

Validates `Authorization: Bearer <jwt>` access tokens issued by Supabase Auth.

Supabase signs project access tokens with HS256 using the raw project JWT
secret from Dashboard → Settings → API → JWT Secret. Some integrations
base64-decode that secret before signing, so both the raw secret and its
URL-safe base64 decoding are accepted as candidate keys.

RS256/JWKS verification (Supabase's option for third-party signing keys) is out
of scope here and documented as not-yet-implemented rather than silently skipped.

Everything is pure stdlib (hmac, hashlib, base64, json) — no crypto dependency.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import time
from typing import Any, Optional

_VALID_ALGS = {"HS256"}


class JWTValidationError(ValueError):
    """Raised when a token fails structural or cryptographic validation."""


def _b64url_decode(data: str) -> bytes:
    """Decode base64url with padding restored (whitespace-tolerant)."""
    value = data.encode("ascii") + b"=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(value)


def _decode_segment(segment: str) -> dict[str, Any]:
    try:
        payload = json.loads(_b64url_decode(segment).decode("utf-8"))
    except (ValueError, binascii.Error, UnicodeDecodeError) as exc:
        raise JWTValidationError("Token segment is not valid base64url JSON.") from exc
    if not isinstance(payload, dict):
        raise JWTValidationError("Token segment is not a JSON object.")
    return payload


def _candidate_keys(secret: str) -> list[bytes]:
    """Return plausible HMAC key encodings for a Supabase JWT secret."""
    keys: list[bytes] = [secret.encode("utf-8")]
    try:
        keys.append(_b64url_decode(secret))
    except (ValueError, binascii.Error):
        pass
    return keys


def validate_access_token(
    token: str,
    jwt_secret: str = "",
    audience: str = "authenticated",
    now: Optional[float] = None,
) -> dict[str, Any]:
    """Validate a Supabase HS256 access token and return its payload.

    Raises :class:`JWTValidationError` (a ValueError) for any invalid token.
    """
    if not token or token.count(".") != 2:
        raise JWTValidationError("Malformed JWT: expected three dot-separated segments.")

    header_segment, payload_segment, signature_segment = token.split(".")
    header = _decode_segment(header_segment)
    payload = _decode_segment(payload_segment)

    alg = header.get("alg")
    if alg not in _VALID_ALGS:
        raise JWTValidationError(
            f"Unsupported or missing JWT algorithm: {alg!r}. Only HS256 is implemented."
        )

    if not jwt_secret:
        raise JWTValidationError(
            "SUPABASE_JWT_SECRET is not configured; cannot verify Supabase tokens."
        )

    signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
    expected = _b64url_decode(signature_segment)
    if not any(
        hmac.compare_digest(expected, hmac.new(key, signing_input, hashlib.sha256).digest())
        for key in _candidate_keys(jwt_secret)
    ):
        raise JWTValidationError("JWT signature verification failed.")

    timestamp = now if now is not None else time.time()
    exp = payload.get("exp")
    if isinstance(exp, (int, float)) and exp < timestamp:
        raise JWTValidationError("JWT is expired.")

    if audience:
        token_aud = payload.get("aud")
        audiences = token_aud if isinstance(token_aud, list) else [token_aud]
        if audience not in audiences:
            raise JWTValidationError(f"JWT audience mismatch: expected {audience!r}.")

    sub = payload.get("sub")
    if not sub:
        raise JWTValidationError("JWT is missing the 'sub' claim.")

    return payload
