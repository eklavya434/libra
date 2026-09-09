"""
Tests for Mixture of Experts (MoE) API Endpoints (Phase 45)
"""

from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_moe_presets():
    response = client.get("/api/v1/moe/presets")
    assert response.status_code == 200
    data = response.json()
    assert "presets" in data
    assert len(data["presets"]) >= 3
    assert any(p["id"] == "mixtral_top2" for p in data["presets"])


def test_moe_forward_endpoint():
    payload = {
        "prompt": "Mixture of experts routes tokens to sub-networks.",
    }
    response = client.post("/api/v1/moe/forward", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prompt"] == payload["prompt"]
    assert "tokens" in data
    assert len(data["tokens"]) > 0
    assert "layer_stats" in data
    assert "parameter_efficiency" in data
    assert data["parameter_efficiency"]["sparsity_ratio"] > 1.0


def test_moe_train_endpoint():
    payload = {
        "steps": 5,
        "aux_loss_coef": 0.02,
        "lr": 0.005,
        "custom_text": "Sparse routing scales parameter capacity.",
    }
    response = client.post("/api/v1/moe/train", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "history" in data
    assert len(data["history"]) == 5
    assert "initial_total_loss" in data
    assert "final_total_loss" in data
    assert "parameter_efficiency" in data


def test_moe_utilization_endpoint():
    response = client.get("/api/v1/moe/utilization")
    assert response.status_code == 200
    data = response.json()

    assert "total_tokens_evaluated" in data
    assert "expert_counts" in data
    assert len(data["expert_counts"]) == 4
    assert "coefficient_of_variation" in data
    assert "parameter_efficiency" in data
