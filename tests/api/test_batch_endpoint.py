"""
Tests for Batch Inference API endpoints (apps/backend/api/v1/endpoints/batch.py)
"""

import pytest
from fastapi.testclient import TestClient

from apps.backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_batch_analyze_endpoint(client):
    payload = {
        "sequence_lengths": [12, 14, 16, 120, 130, 140],
        "max_batch_size": 4,
        "max_tokens_per_bucket": 512,
    }
    response = client.post("/api/v1/batch/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["num_sequences"] == 6
    assert "naive" in data
    assert "binned" in data
    assert data["efficiency_gain_pct"] >= 0.0
    assert data["estimated_speedup"] >= 1.0


def test_batch_job_submission_and_retrieval(client):
    payload = {
        "name": "Integration Batch Test",
        "prompts": ["What is attention?", "Explain transformer feedforward layers."],
        "max_new_tokens": 16,
        "binning_strategy": "dynamic_length",
        "priority": "NORMAL",
    }
    response = client.post("/api/v1/batch/jobs", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "submitted"
    job = data["job"]
    job_id = job["job_id"]
    assert job["num_prompts"] == 2

    # Fetch status
    get_res = client.get(f"/api/v1/batch/jobs/{job_id}")
    assert get_res.status_code == 200
    job_data = get_res.json()
    assert job_data["job_id"] == job_id
    assert job_data["name"] == "Integration Batch Test"


def test_batch_jobs_list(client):
    response = client.get("/api/v1/batch/jobs")
    assert response.status_code == 200
    data = response.json()
    assert "total_jobs" in data
    assert "active_jobs" in data
    assert "worker_concurrency" in data
    assert isinstance(data["jobs"], list)


def test_batch_cancel_nonexistent_job(client):
    response = client.delete("/api/v1/batch/jobs/nonexistent_id_123")
    assert response.status_code == 404
