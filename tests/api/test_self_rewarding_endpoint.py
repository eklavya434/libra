"""
Tests for Self-Rewarding API Endpoints (Phase 49)
"""

from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_presets_endpoint():
    response = client.get("/api/v1/self-rewarding/presets")
    assert response.status_code == 200
    data = response.json()
    assert "rubrics" in data
    assert "prompts" in data
    assert len(data["rubrics"]) >= 2


def test_judge_single_and_pairwise():
    # Test single response
    single_req = {
        "instruction": "Explain gravity in simple words.",
        "response": "Gravity is the attractive force between masses.",
    }
    r1 = client.post("/api/v1/self-rewarding/judge", json=single_req)
    assert r1.status_code == 200
    d1 = r1.json()
    assert not d1["is_pairwise"]
    assert 1.0 <= d1["overall_score"] <= 5.0
    assert 0.0 <= d1["normalized_score"] <= 1.0

    # Test pairwise comparative judging
    pair_req = {
        "instruction": "Define a prime number.",
        "response": "A prime number is a natural number greater than 1 that is only divisible by 1 and itself.",
        "response_b": "Numbers like 2, 3, 5.",
        "debias_position": True,
    }
    r2 = client.post("/api/v1/self-rewarding/judge", json=pair_req)
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["is_pairwise"]
    assert d2["pairwise_winner"] in ["Response A", "Response B", "Tie"]
    assert d2["position_bias_analysis"] is not None


def test_iterate_endpoint():
    req = {
        "prompt": "What is reinforcement learning from human feedback?",
        "iteration_id": 1,
        "num_candidates": 2,
        "temperature": 0.7,
        "beta": 0.1,
    }
    res = client.post("/api/v1/self-rewarding/iterate", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["iteration_id"] == 1
    assert len(data["candidates"]) == 2
    assert "dpo_loss" in data
    assert "implicit_reward_margin" in data


def test_benchmark_endpoint():
    req = {
        "prompts": ["What is a token?", "Explain attention."],
        "num_iterations": 2,
    }
    res = client.post("/api/v1/self-rewarding/benchmark", json=req)
    assert res.status_code == 200
    data = res.json()
    assert len(data["iterations"]) == 2
    assert data["prompts_evaluated"] == 2
    assert "summary" in data
