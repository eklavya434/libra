"""Integration tests for KTO and Online DPO API endpoints."""

from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_kto_presets_endpoint():
    response = client.get("/api/v1/kto/presets")
    assert response.status_code == 200
    data = response.json()
    assert "presets" in data
    assert len(data["presets"]) >= 3
    assert "prospect_theory_defaults" in data
    assert data["prospect_theory_defaults"]["lambda_U"] > 1.0


def test_kto_step_endpoint():
    payload = {
        "samples": [
            {
                "prompt": "Explain gradient descent.",
                "completion": "It iteratively moves parameters in direction of steepest descent.",
                "is_desirable": True,
            },
            {
                "prompt": "Explain gradient descent.",
                "completion": "I do not know.",
                "is_desirable": False,
            },
        ],
        "beta": 0.1,
        "desirable_weight": 1.0,
        "undesirable_weight": 1.5,
        "lr": 1e-4,
    }
    response = client.post("/api/v1/kto/step", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "loss" in data
    assert "reward_margin" in data
    assert data["loss_aversion_ratio"] == 1.5


def test_online_dpo_step_endpoint():
    payload = {
        "prompts": ["What is backpropagation?"],
        "beta": 0.1,
        "temperature": 0.8,
        "num_candidates": 2,
    }
    response = client.post("/api/v1/kto/online-dpo/step", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert "prompt" in data[0]
    assert "chosen_text" in data[0]
    assert "rejected_text" in data[0]
    assert data[0]["chosen_oracle_score"] >= data[0]["rejected_oracle_score"]


def test_alignment_eval_endpoint():
    payload = {
        "prompts": ["What is attention in transformers?", "Explain loss aversion."],
    }
    response = client.post("/api/v1/kto/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "paradigms" in data
    assert len(data["paradigms"]) >= 2
    assert "winner" in data
    assert "sample_efficiency_notes" in data
