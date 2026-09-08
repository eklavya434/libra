"""
Tests for PEFT & LoRA REST Endpoints
Phase 25: Parameter-Efficient Fine-Tuning (PEFT & LoRA)
"""

from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_peft_apply_endpoint():
    payload = {
        "vocab_size": 50,
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 2,
        "rank": 4,
        "alpha": 8.0,
        "target_modules": ["q_proj", "v_proj"],
    }
    response = client.post("/api/v1/peft/apply", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["rank"] == 4
    assert data["alpha"] == 8.0
    assert data["scaling_factor"] == 2.0
    assert data["parameter_summary"]["trainable_parameters"] > 0
    assert data["parameter_summary"]["frozen_parameters"] > 0
    assert "fewer trainable parameters" in data["parameter_savings_ratio"]


def test_peft_train_step_endpoint():
    payload = {
        "vocab_size": 50,
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 2,
        "rank": 4,
        "alpha": 8.0,
        "learning_rate": 0.01,
        "seq_len": 6,
    }
    response = client.post("/api/v1/peft/train_step", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "loss" in data["step_metrics"]
    assert "grad_norm" in data["step_metrics"]
    assert data["step_metrics"]["loss"] >= 0.0


def test_peft_merge_endpoint():
    payload = {
        "vocab_size": 50,
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 2,
        "rank": 4,
        "alpha": 8.0,
    }
    response = client.post("/api/v1/peft/merge", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["is_numerically_identical"] is True
    assert data["max_discrepancy_after_merge"] < 1e-5
