"""Gemini adapter must never fake a successful completion.

Two honesty defects are locked down here:

1. ``finishReason`` was discarded and ``finish_reason`` hardcoded to ``stop``,
   so a ``MAX_TOKENS`` truncation looked identical to a clean completion.
2. A 200 response carrying no visible text was returned as an empty successful
   completion. A thinking model (Gemini 2.5) can spend the whole ``max_tokens``
   budget on internal reasoning and emit no answer at all, which surfaced to
   users as a blank assistant bubble with ``finish_reason: stop``.
"""

from __future__ import annotations

import json

import httpx
import pytest

from packages.providers.errors import LibraProviderError
from packages.providers.gemini import GeminiProvider, map_finish_reason

_MESSAGES = [{"role": "user", "content": "Reply with exactly: OK"}]


def _provider(handler) -> GeminiProvider:
    return GeminiProvider(
        api_key="test-key-not-real",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _generate_content_payload(finish_reason: str, text: str | None) -> dict:
    content: dict = {"parts": []}
    if text is not None:
        content["parts"] = [{"text": text}]
    return {
        "candidates": [{"content": content, "finishReason": finish_reason}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 0},
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("STOP", "stop"),
        ("MAX_TOKENS", "length"),
        ("SAFETY", "content_filter"),
        ("RECITATION", "content_filter"),
        ("PROHIBITED_CONTENT", "content_filter"),
        ("SPII", "content_filter"),
    ],
)
def test_finish_reason_is_mapped_not_collapsed(raw, expected):
    assert map_finish_reason(raw) == expected


def test_unknown_finish_reason_degrades_to_stop():
    assert map_finish_reason("SOMETHING_NEW") == "stop"


@pytest.mark.asyncio
async def test_max_tokens_truncation_is_not_reported_as_stop():
    """A truncated completion must surface as ``length``, not ``stop``."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "partial answer"}]},
                        "finishReason": "MAX_TOKENS",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 8},
            },
        )

    result = await _provider(handler).chat(messages=_MESSAGES, max_tokens=8)
    assert result["choices"][0]["finish_reason"] == "length"
    assert result["choices"][0]["message"]["content"] == "partial answer"


@pytest.mark.asyncio
async def test_empty_completion_raises_instead_of_returning_a_fake_success():
    """The exact live bug: 200 + no text + finish_reason stop must not happen."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_generate_content_payload("MAX_TOKENS", None))

    with pytest.raises(LibraProviderError) as excinfo:
        await _provider(handler).chat(messages=_MESSAGES, max_tokens=8)

    error = excinfo.value
    assert error.status_code == 502
    assert "no visible text" in error.message
    assert "max_tokens" in error.message
    assert "512" in error.message, "error must tell the caller how to fix it"


@pytest.mark.asyncio
async def test_empty_completion_never_carries_a_success_payload():
    """A caller must never receive a 200-shaped dict with empty content."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json=_generate_content_payload("STOP", None))

    with pytest.raises(LibraProviderError):
        await _provider(handler).chat(messages=_MESSAGES, max_tokens=64)
    assert captured["payload"]["generationConfig"]["maxOutputTokens"] == 64


@pytest.mark.asyncio
async def test_blocked_completion_reports_a_content_filter_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_generate_content_payload("SAFETY", None))

    with pytest.raises(LibraProviderError) as excinfo:
        await _provider(handler).chat(messages=_MESSAGES, max_tokens=64)
    assert "blocked" in excinfo.value.message
    assert "SAFETY" in excinfo.value.message


@pytest.mark.asyncio
async def test_whitespace_only_completion_is_treated_as_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_generate_content_payload("STOP", "   \n  "))

    with pytest.raises(LibraProviderError) as excinfo:
        await _provider(handler).chat(messages=_MESSAGES, max_tokens=64)
    assert "empty completion" in excinfo.value.message


@pytest.mark.asyncio
async def test_successful_completion_is_unaffected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_generate_content_payload("STOP", "OK"))

    result = await _provider(handler).chat(messages=_MESSAGES, max_tokens=512)
    assert result["choices"][0]["message"]["content"] == "OK"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["provider"] == "gemini"


@pytest.mark.asyncio
async def test_stream_with_no_visible_text_raises_instead_of_silent_truncation():
    """A stream that yields nothing must fail loudly, not end as a 0-token success."""

    def handler(request: httpx.Request) -> httpx.Response:
        chunk = {
            "candidates": [{"content": {"parts": [{"text": ""}]}, "finishReason": "MAX_TOKENS"}]
        }
        body = "data: " + json.dumps(chunk) + "\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    provider = _provider(handler)
    with pytest.raises(LibraProviderError) as excinfo:
        async for _ in provider.stream(messages=_MESSAGES, max_tokens=8):
            pass
    assert "no visible text" in excinfo.value.message


@pytest.mark.asyncio
async def test_stream_yields_text_when_present():
    def handler(request: httpx.Request) -> httpx.Response:
        body = ""
        for piece in ("Hel", "lo"):
            body += (
                "data: "
                + json.dumps({"candidates": [{"content": {"parts": [{"text": piece}]}}]})
                + "\n\n"
            )
        body += (
            "data: "
            + json.dumps({"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]})
            + "\n\n"
        )
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    provider = _provider(handler)
    chunks = [chunk async for chunk in provider.stream(messages=_MESSAGES, max_tokens=512)]
    assert "".join(chunks) == "Hello"
