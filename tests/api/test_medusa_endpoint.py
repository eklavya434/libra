"""
Tests for Medusa API Endpoints (Phase 46)
"""

from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_medusa_presets():
    response = client.get("/api/v1/medusa/presets")
    assert response.status_code == 200
    data = response.json()
    assert "presets" in data
    assert len(data["presets"]) >= 3
    assert any(p["id"] == "medusa_3heads" for p in data["presets"])


def test_medusa_generate_endpoint():
    payload = {
        "prompt": "Speculative decoding accelerates language models.",
        "max_new_tokens": 10,
        "temperature": 0.0,
    }
    response = client.post("/api/v1/medusa/generate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prompt"] == payload["prompt"]
    assert "generated_text" in data
    assert data["generated_tokens_count"] >= 10
    assert data["speedup_ratio"] > 0
    assert "steps" in data
    assert len(data["steps"]) > 0


def test_medusa_benchmark_endpoint():
    payload = {
        "prompts": ["Medusa verification test."],
        "max_new_tokens": 8,
    }
    response = client.post("/api/v1/medusa/benchmark", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "benchmark" in data
    assert "speedup_ratio" in data["benchmark"]
    assert "head_accuracies" in data


def test_medusa_train_endpoint():
    payload = {
        "steps": 5,
        "lr": 0.005,
        "decay": 0.8,
        "custom_text": "Medusa heads learn future tokens rapidly.",
    }
    response = client.post("/api/v1/medusa/train", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "history" in data
    assert len(data["history"]) == 5
    assert "initial_loss" in data
    assert "final_loss" in data
    assert data["num_heads"] >= 2
