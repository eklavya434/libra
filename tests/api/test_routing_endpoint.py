"""
Tests for Routing & Speculative API Endpoints (Phase 21)
"""

from fastapi.testclient import TestClient

from apps.backend.main import app
from packages.providers.mock import MockProvider
from packages.providers.router import ProviderRouter
from packages.routing.dynamic_router import DynamicRouter

client = TestClient(app)


class _AllMockRouter(ProviderRouter):
    """Maps every (provider, model) pair to the deterministic MockProvider so
    /routing/generate is testable without a live Ollama engine."""

    def get_provider(self, name: str):
        return MockProvider(name=name)


def test_endpoint_classify_prompt():
    response = client.post(
        "/api/v1/routing/classify",
        json={"prompt": "Write a quicksort function in Python"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "intent" in data
    assert data["intent"] == "code_generation"
    assert "recommended_tier" in data
    assert "complexity" in data
    assert "overall_score" in data["complexity"]


def test_endpoint_routing_decision():
    response = client.post(
        "/api/v1/routing/decision",
        json={"prompt": "Hi", "policy": "auto"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["selected_tier"] == "fast_local"
    assert "selected_model" in data
    assert "fallback_chain" in data


def test_endpoint_routing_generate(monkeypatch):
    import apps.backend.api.v1.endpoints.routing as routing_endpoint

    monkeypatch.setattr(
        routing_endpoint,
        "get_dynamic_router",
        lambda: DynamicRouter(provider_router=_AllMockRouter()),
    )
    response = client.post(
        "/api/v1/routing/generate",
        json={
            "messages": [{"role": "user", "content": "Explain gravity in one sentence"}],
            "policy": "auto",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "content" in data
    assert data["content"], "expected a real (stubbed) provider response"
    assert "tier_used" in data
    assert "routing_decision" in data


def test_endpoint_routing_generate_fails_loudly_when_no_provider(monkeypatch):
    """No live provider must yield an honest 503, never a fabricated reply."""
    import apps.backend.api.v1.endpoints.routing as routing_endpoint
    from apps.backend.api.v1.endpoints.routing import logger as routing_logger  # noqa: F401

    class _Unavailable:
        def get_provider(self, name):
            raise ValueError("no provider")

        async def health(self):
            return {"connected": False}

    from packages.providers.base import BaseProvider

    class _FailingProvider(BaseProvider):
        @property
        def name(self) -> str:
            return "stub-fail"

        async def chat(self, messages, model="x", **kwargs):
            raise RuntimeError("simulated provider outage")

        async def health(self):
            return {"healthy": True, "connected": True}

        async def list_models(self):
            return []

        async def stream(self, messages, model="x", **kwargs):
            if False:
                yield ""

        async def embeddings(self, texts, model=None):
            return [[0.0] * 8 for _ in texts]

        def capabilities(self):
            return {"supports_text": True}

    failing_router = ProviderRouter()
    failing_router._providers["ollama"] = _FailingProvider()
    monkeypatch.setattr(
        routing_endpoint,
        "get_dynamic_router",
        lambda: DynamicRouter(provider_router=failing_router),
    )
    response = client.post(
        "/api/v1/routing/generate",
        json={"messages": [{"role": "user", "content": "Hello"}], "policy": "auto"},
    )
    assert response.status_code == 503
    body = response.json()["detail"]
    assert "no model provider could serve this request" in body
    assert "simulated provider outage" not in body


def test_endpoint_speculative_generate():
    response = client.post(
        "/api/v1/routing/speculative",
        json={
            "prompt": "Hello",
            "max_new_tokens": 10,
            "lookahead_k": 3,
            "temperature": 0.0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "tokens_generated" in data
    assert data["tokens_generated"] == 10
    assert "acceptance_rate" in data
    assert "target_forward_passes" in data
    assert "theoretical_speedup" in data
    assert data["verified_equivalent"] is True
