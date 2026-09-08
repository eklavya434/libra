"""
Tests for API v1 Telemetry Endpoints
Verifies /generate, /stream, and /analyze endpoints.
"""

from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_telemetry_generate_endpoint():
    """Verifies that /api/v1/telemetry/generate produces valid sequence telemetry."""
    payload = {
        "prompt": "Hello AI",
        "max_tokens": 4,
        "temperature": 0.8,
        "candidate_top_k": 3,
    }
    response = client.post("/api/v1/telemetry/generate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "tokens" in data
    assert len(data["tokens"]) == 4
    assert "mean_surprisal_bits" in data
    assert "perplexity" in data
    assert data["perplexity"] > 0.0
    assert "tokens_per_second" in data
    assert data["total_tokens"] == 4

    first_tok = data["tokens"][0]
    assert "token_text" in first_tok
    assert "surprisal_bits" in first_tok
    assert "entropy_bits" in first_tok
    assert "top_k" in first_tok
    assert len(first_tok["top_k"]) <= 3


def test_telemetry_stream_endpoint():
    """Verifies that /api/v1/telemetry/stream delivers SSE events for tokens and completion."""
    payload = {
        "prompt": "Testing SSE",
        "max_tokens": 3,
        "temperature": 0.5,
        "candidate_top_k": 2,
    }
    response = client.post("/api/v1/telemetry/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    body = response.text
    assert "event: token" in body
    assert "event: done" in body
    assert "data: [DONE]" in body


def test_telemetry_analyze_endpoint():
    """Verifies that /api/v1/telemetry/analyze evaluates teacher-forcing sequence surprisal."""
    payload = {
        "text": "Deep learning from scratch",
        "candidate_top_k": 3,
    }
    response = client.post("/api/v1/telemetry/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "tokens" in data
    assert len(data["tokens"]) > 0
    assert "perplexity" in data
    assert data["perplexity"] > 0.0
    assert data["max_surprisal_token"] is not None


def test_telemetry_analyze_too_short():
    """Verifies that /api/v1/telemetry/analyze rejects single-character inputs."""
    payload = {
        "text": "A",
    }
    response = client.post("/api/v1/telemetry/analyze", json=payload)
    # min_length=2 on pydantic schema -> 422 Unprocessable Entity
    assert response.status_code == 422
