"""Integration tests for Verifiable Search API endpoints."""

from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_verifiable_search_presets_endpoint():
    response = client.get("/api/v1/verifiable-search/presets")
    assert response.status_code == 200
    data = response.json()
    assert "presets" in data
    assert len(data["presets"]) >= 3
    assert "verifiable_search_theory" in data


def test_verifiable_search_solve_endpoint():
    payload = {
        "prompt": "Calculate (15 * 4) + (24 / 3) - 17",
        "beam_width": 3,
        "max_depth": 5,
        "step_prune_threshold": 0.55,
        "branching_factor": 2,
    }
    response = client.post("/api/v1/verifiable-search/solve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_verified"] is True
    assert "51" in data["final_answer"]
    assert data["total_steps_explored"] > 0
    assert data["pruned_branches_count"] >= 1
    assert len(data["tree_nodes"]) >= 4


def test_verifiable_search_benchmark_endpoint():
    payload = {
        "problems": [
            {
                "prompt": "Calculate (15 * 4) + (24 / 3) - 17",
                "expected_answer": "51",
            },
        ],
        "n_best_of_n": 3,
    }
    response = client.post("/api/v1/verifiable-search/benchmark", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "methods" in data
    assert len(data["methods"]) == 3
    assert data["winner"] == "Step-Level Verifiable Search (PRM)"
