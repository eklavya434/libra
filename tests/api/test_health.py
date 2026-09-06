"""
API integration tests for Health and Models endpoints.
"""

from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "docs_url" in data
    assert "health_url" in data
    assert data["learning_mode"] is True


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "hardware" in data
    hw = data["hardware"]
    assert "cpu_model" in hw
    assert "ram_total_gb" in hw
    assert "disk_free_gb" in hw
    assert "device_tier" in hw
    assert data["learning_mode"] is True


def test_models_endpoint():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert data["count"] >= 1
    model = data["models"][0]
    assert model["id"] == "libra-mock-v1"
    assert model["provider"] == "mock-provider"
