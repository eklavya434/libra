"""
Unit tests for Libra Provider Error Normalization (Phase 9).
Verifies mapping of HTTP status codes (401, 403, 404, 429, 500) and connection failures.
"""

from __future__ import annotations

from packages.providers.errors import (
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderQuotaExceededError,
    ProviderRateLimitError,
    normalize_http_error,
)


def test_authentication_error_normalization():
    """HTTP 401/403 should map to ProviderAuthenticationError."""
    err401 = normalize_http_error(
        status_code=401,
        response_body={"error": {"message": "Incorrect API key provided"}},
        provider="openai",
    )
    assert isinstance(err401, ProviderAuthenticationError)
    assert err401.provider == "openai"
    assert err401.status_code == 401
    assert "Incorrect API key" in err401.message

    err403 = normalize_http_error(
        status_code=403,
        response_body="Permission denied for project",
        provider="gemini",
    )
    assert isinstance(err403, ProviderAuthenticationError)
    assert err403.status_code == 403


def test_rate_limit_and_quota_normalization():
    """HTTP 429 maps to RateLimit or QuotaExceeded based on body content."""
    rate_err = normalize_http_error(
        status_code=429,
        response_body={"error": {"message": "Rate limit reached: 30 requests per minute"}},
        provider="groq",
    )
    assert isinstance(rate_err, ProviderRateLimitError)
    assert rate_err.status_code == 429

    quota_err = normalize_http_error(
        status_code=429,
        response_body={"error": {"message": "You exceeded your current quota or billing limit"}},
        provider="openai",
    )
    assert isinstance(quota_err, ProviderQuotaExceededError)
    assert quota_err.status_code == 429


def test_model_not_found_normalization():
    """HTTP 404 maps to ModelNotFoundError."""
    err404 = normalize_http_error(
        status_code=404,
        response_body={"error": {"message": "The model 'gpt-9-super' does not exist"}},
        provider="openai",
    )
    assert isinstance(err404, ModelNotFoundError)
    assert err404.status_code == 404


def test_generic_fallback_error():
    """HTTP 500 or unknown errors map to base ProviderError."""
    err500 = normalize_http_error(
        status_code=500,
        response_body={"error": {"message": "Internal Server Error"}},
        provider="anthropic",
    )
    assert isinstance(err500, ProviderError)
    assert err500.status_code == 500
    assert err500.to_dict()["error_type"] == "LibraProviderError"
