"""
Tests for Knowledge Distillation REST Endpoints (Phase 44)
"""

from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_distillation_presets():
    response = client.get("/api/v1/distillation/presets")
    assert response.status_code == 200
    data = response.json()
    assert "presets" in data
    assert len(data["presets"]) >= 3
    assert any(p["id"] == "standard_balanced" for p in data["presets"])


def test_distillation_train_endpoint():
    payload = {
        "temperature": 2.0,
        "alpha": 0.5,
        "lr": 0.005,
        "steps": 5,
        "hidden_loss_weight": 0.0,
        "custom_text": "Distillation transfers soft knowledge effectively.",
    }
    response = client.post("/api/v1/distillation/train", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "telemetry" in data
    assert len(data["telemetry"]) == 5
    assert "compression_stats" in data
    assert data["compression_stats"]["compression_ratio"] > 1.0
    assert "initial_total_loss" in data
    assert "final_total_loss" in data


def test_distillation_soft_labels_endpoint():
    payload = {
        "prompt": "The quick brown fox jumps",
        "temperatures": [1.0, 3.0],
        "top_k": 4,
    }
    response = client.post("/api/v1/distillation/soft_labels", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prompt"] == "The quick brown fox jumps"
    assert "temperature_analysis" in data
    assert len(data["temperature_analysis"]) == 2
    for entry in data["temperature_analysis"]:
        assert "kl_divergence" in entry
        assert "teacher_top_tokens" in entry
        assert len(entry["teacher_top_tokens"]) <= 4


def test_distillation_evaluate_endpoint():
    payload = {
        "prompts": ["Knowledge distillation compresses networks."],
        "runs": 2,
    }
    response = client.post("/api/v1/distillation/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "benchmark" in data
    assert data["benchmark"]["speedup_ratio"] > 0
    assert "compression_stats" in data
    assert "sample_predictions" in data
    assert len(data["sample_predictions"]) == 1
    assert "tokens_match" in data["sample_predictions"][0]
