"""Shared FastAPI dependencies for the v1 API."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any

from fastapi import HTTPException


def resolve_effective_model_id(model_id: str | None) -> str:
    """Pick a real model id for a request: the client's explicit id, otherwise
    the configured system default.

    Public endpoints must never silently run on MockProvider, so when neither
    an explicit model nor a configured provider exists we return 400 with an
    actionable message instead.
    """
    if model_id and model_id.strip():
        return model_id.strip()
    # Lazy import keeps this helper monkeypatchable and avoids import cycles.
    from apps.backend.api.v1.endpoints.models import get_system_default_model

    default_id = get_system_default_model()
    if not default_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "No model specified and no provider is configured. Pass an explicit "
                "model id or add a provider API key (e.g. GEMINI_API_KEY, "
                "ANTHROPIC_API_KEY) to .env."
            ),
        )
    return default_id


def extract_provider_text(res: dict[str, Any]) -> str:
    """Extract the assistant text from any provider chat response dict."""
    choices = res.get("choices")
    if choices:
        message = choices[0].get("message", {}) if isinstance(choices, list) else {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
    content = res.get("content")
    if isinstance(content, str) and content.strip():
        return content
    text = res.get("text")
    if isinstance(text, str) and text.strip():
        return text
    raise RuntimeError("Provider response contained no usable text content.")


async def _provider_completion_async(model_id: str, prompt: str, **kwargs: Any) -> str:
    """Run a single-turn completion through the provider router and return text."""
    from packages.providers.router import get_router

    res = await get_router().chat(
        messages=[{"role": "user", "content": prompt}],
        model_id=model_id,
        **kwargs,
    )
    return extract_provider_text(res)


def run_provider_completion(model_id: str, prompt: str, **kwargs: Any) -> str:
    """Synchronous completion bridge for sync benchmark loops.

    Raised errors surface as an honest 503 with a generic client message; the
    underlying exception is logged server-side. Results are never fabricated.
    """
    import logging

    logger = logging.getLogger("libra.deps")

    async def _run() -> str:
        return await _provider_completion_async(model_id, prompt, **kwargs)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _run())
                return future.result()
        return loop.run_until_complete(_run())
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Provider completion failed for model '%s': %s", model_id, exc)
        raise HTTPException(
            status_code=503,
            detail=f"The provider for model '{model_id}' failed to respond.",
        ) from exc
