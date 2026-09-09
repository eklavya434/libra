"""
Libra Providers - Normalized Error Hierarchy

Translates heterogeneous HTTP status codes and error payloads from diverse
providers (OpenAI, Gemini, Anthropic, Ollama, Groq) into consistent Libra exceptions.
"""

from __future__ import annotations

from typing import Any, Optional


class LibraProviderError(Exception):
    """Base exception for all model provider failures."""

    def __init__(
        self, message: str, provider: str, status_code: Optional[int] = None, details: Any = None
    ):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "provider": self.provider,
            "status_code": self.status_code,
            "details": self.details,
        }


ProviderError = LibraProviderError


class ProviderAuthenticationError(LibraProviderError):
    """Raised when API key is missing, invalid, or expired (HTTP 401/403)."""


class ProviderRateLimitError(LibraProviderError):
    """Raised when request frequency or token velocity limits are exceeded (HTTP 429)."""


class ProviderQuotaExceededError(LibraProviderError):
    """Raised when credit balance or billing quota is exhausted."""


class ModelNotFoundError(LibraProviderError):
    """Raised when the specified model is unknown or unsupported (HTTP 404)."""


class ProviderOfflineError(LibraProviderError):
    """Raised when a local daemon or remote gateway is unreachable (ConnectionError / Timeout)."""


def normalize_http_error(
    status_code: int,
    response_body: str | dict[str, Any],
    provider: str,
) -> LibraProviderError:
    """Parses raw HTTP error response and returns the corresponding LibraProviderError."""
    msg = ""
    if isinstance(response_body, dict):
        # Extract message across OpenAI, Anthropic, Gemini schemas
        msg = (
            response_body.get("error", {}).get("message")
            if isinstance(response_body.get("error"), dict)
            else response_body.get("message") or response_body.get("error") or ""
        )
    if not msg:
        msg = str(response_body)

    if status_code in (401, 403):
        return ProviderAuthenticationError(
            message=f"[{provider.upper()} Auth Error] {msg or 'Invalid or missing API key.'}",
            provider=provider,
            status_code=status_code,
            details=response_body,
        )

    if status_code == 429:
        if "quota" in msg.lower() or "credit" in msg.lower() or "billing" in msg.lower():
            return ProviderQuotaExceededError(
                message=f"[{provider.upper()} Quota Exceeded] {msg or 'Billing quota or credit limit exhausted.'}",
                provider=provider,
                status_code=status_code,
                details=response_body,
            )
        return ProviderRateLimitError(
            message=f"[{provider.upper()} Rate Limit] {msg or 'Too many requests; please slow down.'}",
            provider=provider,
            status_code=status_code,
            details=response_body,
        )

    if status_code == 404:
        return ModelNotFoundError(
            message=f"[{provider.upper()} Model Not Found] {msg or 'Requested model does not exist.'}",
            provider=provider,
            status_code=status_code,
            details=response_body,
        )

    return LibraProviderError(
        message=f"[{provider.upper()} Error {status_code}] {msg}",
        provider=provider,
        status_code=status_code,
        details=response_body,
    )
