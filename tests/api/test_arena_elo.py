"""
API tests for Model Arena, Elo Leaderboard, Battles, and Tournaments (Phase 28).
"""

from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_get_leaderboard():
    response = client.get("/api/v1/arena/leaderboard?category=overall")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_record_vote():
    payload = {
        "model_a": "model_test_x",
        "model_b": "model_test_y",
        "winner": "model_a",
        "category": "coding",
        "prompt": "Write a hello world function",
    }
    response = client.post("/api/v1/arena/vote", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "match" in data
    assert data["match"]["winner"] == "model_a"
    assert data["model_a_rating"] > 1200.0
    assert data["model_b_rating"] < 1200.0


def test_battle_endpoint():
    payload = {
        "prompt": "Explain what a neural network is in 2 sentences.",
        "model_a": {"model": "libra-mock-v1", "provider": "mock-provider"},
        "model_b": {"model": "libra-mock-v2", "provider": "mock-provider"},
        "category": "factual",
        "max_tokens": 64,
    }
    response = client.post("/api/v1/arena/battle", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "match" in data
    assert "verdict" in data
    assert "completion_a" in data
    assert "completion_b" in data
    assert data["verdict"]["position_swapped"] is True


def test_tournament_endpoint():
    payload = {
        "models": [
            {"model": "libra-mock-v1", "provider": "mock-provider"},
            {"model": "libra-mock-v2", "provider": "mock-provider"},
        ],
        "categories": ["instruction"],
        "prompts_per_category": 1,
        "max_tokens": 48,
    }
    response = client.post("/api/v1/arena/tournament", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "tournament_id" in data
    assert data["total_matches"] >= 1
    assert "leaderboard_rankings" in data
