"""
Tests for the Supabase HS256 JWT validator (pure stdlib).

Covers signature verification (both raw-secret and base64-decoded-secret key
encodings), expiry, audience, malformed input, and graceful ValueError raises.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import pytest

from packages.core.security.supabase_jwt import (
    JWTValidationError,
    validate_access_token,
)

SECRET = "super-secret-supabase-project-key"


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _sign(payload: dict, secret: str | bytes = SECRET, alg: str = "HS256") -> str:
    header = {"alg": alg, "typ": "JWT"}
    h = _b64(json.dumps(header, separators=(",", ":")).encode("ascii"))
    p = _b64(json.dumps(payload, separators=(",", ":")).encode("ascii"))
    key = secret.encode("utf-8") if isinstance(secret, str) else secret
    sig = _b64(hmac.new(key, f"{h}.{p}".encode("ascii"), hashlib.sha256).digest())
    return f"{h}.{p}.{sig}"


def _payload(**overrides) -> dict:
    base = {
        "sub": "user-123",
        "aud": "authenticated",
        "iat": int(time.time()) - 60,
        "exp": int(time.time()) + 3600,
        "role": "authenticated",
    }
    base.update(overrides)
    return base


def test_valid_token_returns_payload():
    token = _sign(_payload())
    payload = validate_access_token(token, jwt_secret=SECRET)
    assert payload["sub"] == "user-123"


def test_base64_decoded_secret_encoding_is_accepted():
    # Some Supabase clients base64-decode the JWT secret before signing: the
    # configured secret is the base64 *string* while the HMAC key is its
    # decoded bytes. The validator must accept both encodings.
    b64_secret = "c2VjcmV0LWtleS1mb3ItdGVzdHM="  # base64 of "secret-key-for-tests"
    decoded_key = base64.urlsafe_b64decode(b64_secret.encode("utf-8"))
    assert decoded_key == b"secret-key-for-tests"
    token = _sign(_payload(), secret=decoded_key)
    payload = validate_access_token(token, jwt_secret=b64_secret)
    assert payload["sub"] == "user-123"


def test_wrong_secret_rejected():
    token = _sign(_payload(), secret="attacker-key")
    with pytest.raises(JWTValidationError, match="signature"):
        validate_access_token(token, jwt_secret=SECRET)


def test_expired_token_rejected():
    token = _sign(_payload(exp=int(time.time()) - 10))
    with pytest.raises(JWTValidationError, match="expired"):
        validate_access_token(token, jwt_secret=SECRET)


def test_audience_mismatch_rejected():
    token = _sign(_payload(aud="service_role"))
    with pytest.raises(JWTValidationError, match="audience"):
        validate_access_token(token, jwt_secret=SECRET)


def test_list_audience_accepts_expected_value():
    token = _sign(_payload(aud=["authenticated"]))
    payload = validate_access_token(token, jwt_secret=SECRET)
    assert payload["sub"] == "user-123"


def test_missing_secret_raises():
    token = _sign(_payload())
    with pytest.raises(JWTValidationError, match="SUPABASE_JWT_SECRET"):
        validate_access_token(token, jwt_secret="")


def test_unsupported_algorithm_rejected():
    token = _sign(_payload(), alg="RS256")
    with pytest.raises(JWTValidationError, match="Only HS256"):
        validate_access_token(token, jwt_secret=SECRET)


def test_malformed_token_raises():
    with pytest.raises(JWTValidationError, match="segment"):
        validate_access_token("a.b.c", jwt_secret=SECRET)
    with pytest.raises(JWTValidationError, match="Malformed"):
        validate_access_token("a.b", jwt_secret=SECRET)


def test_missing_sub_rejected():
    token = _sign({k: v for k, v in _payload().items() if k != "sub"})
    with pytest.raises(JWTValidationError, match="sub"):
        validate_access_token(token, jwt_secret=SECRET)
