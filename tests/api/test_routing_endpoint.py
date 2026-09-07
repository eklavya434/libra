"""
Tests for Routing & Speculative API Endpoints (Phase 21)
"""

from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


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


def test_endpoint_routing_generate():
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
    assert "tier_used" in data
    assert "routing_decision" in data


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
