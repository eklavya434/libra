"""
Tests for gedo/Schelling model default resolution.

Covers packages.providers (opencode degrade) and the /models/default endpoint.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from apps.backend.api.v1.endpoints import models as models_endpoint
from apps.backend.main import app

# Provider id -> representative catalog model id expected as default when configured.
_PROVIDER_DEFAULTS = {
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-3-5-haiku-20241022",
    "nvidia": "nvidia/llama-3.1-nemotron-70b-instruct",
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-ai/deepseek-v4-flash-0731",
    "kimi": "kimi-k1.5",
}
_KEYS = ["gemini", "anthropic", "nvidia", "openai", "deepseek", "kimi"]


def _set_provider_keys(monkeypatch, **present) -> None:
    router = models_endpoint.provider_router
    for pid in _KEYS:
        prov = router.get_provider(pid)
        monkeypatch.setattr(prov, "api_key", present.get(pid), raising=False)


def test_default_model_is_none_when_no_provider_configured(monkeypatch):
    _set_provider_keys(monkeypatch)
    assert models_endpoint.get_system_default_model() is None


@pytest.mark.parametrize("pid,expected", list(_PROVIDER_DEFAULTS.items()))
def test_default_model_prefers_configured_provider(monkeypatch, pid, expected):
    _set_provider_keys(monkeypatch, **{pid: "test-key"})
    assert models_endpoint.get_system_default_model() == expected


def test_default_model_prefers_gemini_over_openai(monkeypatch):
    _set_provider_keys(monkeypatch, gemini="k", openai="k")
    assert models_endpoint.get_system_default_model() == "gemini-2.5-flash"


def test_default_model_when_all_configured_prefers_gemini(monkeypatch):
    _set_provider_keys(monkeypatch, gemini="k", anthropic="k", nvidia="k")
    assert models_endpoint.get_system_default_model() == "gemini-2.5-flash"


def test_models_default_endpoint_reports_unconfigured(monkeypatch):
    _set_provider_keys(monkeypatch)
    client = TestClient(app, raise_server_exceptions=False)
    res = client.get("/api/v1/models/default")
    assert res.status_code == 200
    data = res.json()
    assert data["configured"] is False
    assert data["default_model"] is None
    assert "No provider" in data["message"]


def test_models_default_endpoint_reports_configured(monkeypatch):
    _set_provider_keys(monkeypatch, gemini="test-key")
    client = TestClient(app, raise_server_exceptions=False)
    res = client.get("/api/v1/models/default")
    assert res.status_code == 200
    data = res.json()
    assert data["configured"] is True
    assert data["default_model"] == "gemini-2.5-flash"
    assert data["model"]["model_id"] == "gemini-2.5-flash"


def test_routing_open_code_never_degrades_to_mock(monkeypatch):
    """An opencode/coder request without a running Ollama engine fails loudly."""
    from packages.providers.router import get_router

    router = get_router()
    ollama = router.get_provider("ollama")
    monkeypatch.setattr(ollama, "health", _unreachable_health, raising=False)
    monkeypatch.setattr(ollama, "list_models", _no_models, raising=False)

    async def attempt() -> None:
        try:
            await router.resolve_provider_for_model(model_id="opencode-coder")
        except ValueError:
            return
        raise AssertionError("Expected ValueError; provider resolved instead.")

    asyncio.run(attempt())


async def _unreachable_health():
    return {"connected": False}


async def _no_models():
    return []
