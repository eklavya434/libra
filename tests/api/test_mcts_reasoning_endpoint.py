"""
Integration tests for MCTS reasoning, PRM scoring, and self-play endpoints.
"""

from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_api_mcts_presets():
    resp = client.get("/api/v1/reasoning/mcts/presets")
    assert resp.status_code == 200
    presets = resp.json()
    assert len(presets) >= 3
    assert any("24" in p["id"] for p in presets)


def test_api_mcts_search():
    payload = {
        "prompt": "Using numbers [4, 4, 7, 7], make 24.",
        "n_simulations": 10,
        "branch_factor": 2,
        "max_depth": 3,
    }
    resp = client.post("/api/v1/reasoning/mcts/search", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_simulations"] == 10
    assert len(data["optimal_path"]) > 0
    assert len(data["tree_nodes"]) > 0


def test_api_prm_score():
    payload = {
        "steps": [
            "Step 1: Notice that 7 / 7 = 1.",
            "Step 2: 7 - 1 = 6.",
            "Step 3: 4 * 6 = 24.",
        ]
    }
    resp = client.post("/api/v1/reasoning/mcts/prm/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_valid_trace"] is True
    assert len(data["steps"]) == 3
    assert data["mean_step_score"] >= 0.70


def test_api_self_play_generate():
    resp = client.post("/api/v1/reasoning/mcts/self_play/generate", json={"count": 2})
    assert resp.status_code == 200
    pairs = resp.json()
    assert len(pairs) == 2
    assert pairs[0]["reward_margin"] > 0
