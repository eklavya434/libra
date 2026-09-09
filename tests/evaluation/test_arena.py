"""
Unit & Integration Tests for Libra Model Comparison Arena (Phase 10)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apps.backend.main import app
from packages.evaluation.arena import ModelComparisonArena


@pytest.mark.asyncio
async def test_arena_concurrent_benchmarking():
    """Verify ModelComparisonArena executes multiple models concurrently."""
    arena = ModelComparisonArena()
    models = [
        {"model": "libra-mock-v1", "provider": "mock-provider"},
        {"model": "mock-model-b", "provider": "mock-provider"},
    ]

    result = await arena.compare(
        prompt="Explain backpropagation",
        models=models,
        max_tokens=64,
        temperature=0.5,
    )

    assert result["prompt"] == "Explain backpropagation"
    assert result["total_models"] == 2
    assert result["successful_models"] == 2
    assert len(result["results"]) == 2

    for m in result["results"]:
        assert m["success"] is True
        assert m["output_text"] != ""
        assert m["total_latency_ms"] >= 0.0
        assert m["tokens_per_second"] >= 0.0
        assert m["is_free"] is True
        assert m["cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_arena_fault_isolation():
    """Verify that an invalid or failing model does not crash the other models."""
    arena = ModelComparisonArena()
    models = [
        {"model": "libra-mock-v1", "provider": "mock-provider"},
        {"model": "nonexistent-broken-model", "provider": "invalid-provider-name-xyz"},
    ]

    result = await arena.compare(
        prompt="Hello world",
        models=models,
        max_tokens=32,
    )

    assert result["total_models"] == 2
    assert result["successful_models"] == 1

    # First model succeeded
    m1 = result["results"][0]
    assert m1["success"] is True

    # Second model cleanly recorded error
    m2 = result["results"][1]
    assert m2["success"] is False
    assert m2["error"] is not None


def test_arena_api_endpoint():
    """Verify POST /api/v1/arena/compare FastAPI endpoint."""
    client = TestClient(app)
    payload = {
        "prompt": "What is attention?",
        "models": [{"model": "libra-mock-v1", "provider": "mock-provider"}],
        "temperature": 0.7,
        "max_tokens": 64,
    }

    resp = client.post("/api/v1/arena/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["prompt"] == "What is attention?"
    assert data["successful_models"] == 1
    assert len(data["results"]) == 1
    assert data["results"][0]["model"] == "libra-mock-v1"
    assert data["rankings"]["highest_throughput"] == "libra-mock-v1"
