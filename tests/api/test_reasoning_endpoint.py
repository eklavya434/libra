"""
API tests for Reasoning Engine Endpoints (Phase 29).
"""

from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_reasoning_generate_endpoint():
    payload = {
        "prompt": "What is the square root of 144?",
        "model": "libra-mock-v1",
        "max_tokens": 64,
    }
    response = client.post("/api/v1/reasoning/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "model" in data
    assert "trace" in data
    assert "final_answer" in data["trace"]


def test_reasoning_self_consistency_endpoint():
    payload = {
        "prompt": "If a train travels at 60 mph for 2 hours, how far does it travel?",
        "model": "libra-mock-v1",
        "num_paths": 2,
        "max_tokens": 48,
    }
    response = client.post("/api/v1/reasoning/self-consistency", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "consensus_answer" in data
    assert "confidence" in data
    assert data["total_paths"] >= 1


def test_reasoning_best_of_n_endpoint():
    payload = {
        "prompt": "Explain photosynthesis concisely.",
        "model": "libra-mock-v1",
        "n_candidates": 2,
        "max_tokens": 48,
    }
    response = client.post("/api/v1/reasoning/best-of-n", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "best_candidate" in data
    assert "best_score" in data
    assert len(data["candidates"]) >= 1


def test_reasoning_parse_endpoint():
    payload = {
        "text": "<think>\n1. Break down problem\n2. Solve step\n</think>\nThe answer is 42.",
        "duration_ms": 850.0,
    }
    response = client.post("/api/v1/reasoning/parse", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["has_thought"] is True
    assert data["final_answer"] == "The answer is 42."
    assert len(data["steps"]) >= 2
