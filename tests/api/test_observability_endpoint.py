"""
Tests for Observability API endpoints and middleware (apps/backend/api/v1/endpoints/observability.py)
"""

import pytest
from fastapi.testclient import TestClient

from apps.backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_tracing_middleware_injects_headers(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert "X-Trace-Id" in response.headers
    assert len(response.headers["X-Trace-Id"]) == 32
    assert "Server-Timing" in response.headers
    assert "dur=" in response.headers["Server-Timing"]


def test_list_traces_endpoint(client):
    # Perform a request to generate a trace
    client.get("/api/v1/health")

    response = client.get("/api/v1/observability/traces")
    assert response.status_code == 200
    data = response.json()
    assert "traces" in data
    assert data["total"] >= 1
    first_trace = data["traces"][0]
    assert "trace_id" in first_trace
    assert "root_name" in first_trace
    assert "total_duration_ms" in first_trace


def test_get_trace_detail_endpoint(client):
    # Perform a request to generate a trace
    res = client.get("/api/v1/health")
    trace_id = res.headers["X-Trace-Id"]

    response = client.get(f"/api/v1/observability/traces/{trace_id}")
    assert response.status_code == 200
    detail = response.json()
    assert detail["trace_id"] == trace_id
    assert "spans" in detail
    assert len(detail["spans"]) >= 1
    first_span = detail["spans"][0]
    assert "offset_ms" in first_span
    assert "duration_ms" in first_span


def test_observability_metrics_endpoint(client):
    response = client.get("/api/v1/observability/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "avg_latency_ms" in data
    assert "p50_latency_ms" in data
    assert "buffered_traces_count" in data


def test_clear_traces_endpoint(client):
    response = client.post("/api/v1/observability/traces/clear")
    assert response.status_code == 200
    assert response.json()["status"] == "cleared"

    # Verify traces list is empty
    list_res = client.get("/api/v1/observability/traces")
    assert list_res.json()["total"] == 0
