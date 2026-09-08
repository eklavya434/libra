"""
API Tests for Phase 31: PagedAttention & Continuous Batching Endpoints
"""

import pytest
from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_paged_simulate_endpoint():
    payload = {
        "batch_size": 8,
        "avg_prompt_tokens": 100,
        "max_output_tokens": 200,
        "block_size": 16,
    }
    response = client.post("/api/v1/paged/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "contiguous_waste_percent" in data
    assert "paged_internal_frag_percent" in data
    assert data["contiguous_waste_percent"] > data["paged_internal_frag_percent"]
    assert data["memory_savings_percent"] > 0.0
    assert data["max_concurrency_multiplier"] > 1.0


def test_paged_batch_run_endpoint():
    payload = {
        "prompts": [
            "Test query one",
            "Test query two longer",
        ],
        "max_tokens": 6,
    }
    response = client.post("/api/v1/paged/batch_run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_requests"] == 2
    assert data["total_tokens_generated"] == 12
    assert len(data["completed_requests"]) == 2
    assert len(data["timeline"]) >= 6


def test_paged_memory_stats_endpoint():
    response = client.get("/api/v1/paged/memory_stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_blocks" in data
    assert "allocated_blocks" in data
    assert "free_blocks" in data
