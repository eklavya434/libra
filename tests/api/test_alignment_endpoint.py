"""
Tests for Alignment API Endpoints (Phase 22)
"""

from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_endpoint_score_reward():
    response = client.post(
        "/api/v1/alignment/reward",
        json={
            "prompt": "What is Python?",
            "completion": "Python is a high-level programming language.",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "reward_score" in data
    assert isinstance(data["reward_score"], float)
    assert data["total_tokens"] > 0


def test_endpoint_rank_reward():
    response = client.post(
        "/api/v1/alignment/reward/rank",
        json={
            "prompt": "Explain gravity: ",
            "completions": [
                "Gravity is the curvature of spacetime caused by mass and energy.",
                "I don't know and I don't care.",
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "ranked_completions" in data
    assert len(data["ranked_completions"]) == 2
    assert "winner" in data


def test_endpoint_dpo_step():
    response = client.post(
        "/api/v1/alignment/dpo/step",
        json={
            "pairs": [
                {
                    "prompt": "Say hello: ",
                    "chosen": "Hello! How can I help you today?",
                    "rejected": "Go away.",
                }
            ],
            "beta": 0.1,
            "lr": 1e-4,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "loss" in data
    assert "accuracy" in data
    assert "reward_margin" in data
    assert "chosen_implicit_reward" in data
    assert "rejected_implicit_reward" in data
