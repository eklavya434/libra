"""
Tests for Capstone System Status Endpoint (Phase 36)
"""

from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_capstone_status_endpoint():
    response = client.get("/api/v1/capstone/status")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "operational"
    assert data["total_phases"] == 36
    assert data["completed_phases"] == 36
    assert data["curriculum_complete"] is True
    assert len(data["pillars"]) == 7
    assert "$0" in data["cost_policy"]
    assert data["disk_quota_used_mb"] < data["disk_quota_max_mb"]
