"""
Integration tests for Long-Context NIAH & Attention Compaction endpoints.
"""

from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_api_long_context_presets():
    resp = client.get("/api/v1/long_context/presets")
    assert resp.status_code == 200
    presets = resp.json()
    assert len(presets) >= 2
    assert any("vault" in p["id"] for p in presets)


def test_api_single_needle_grid():
    payload = {
        "model": "mock",
        "needle": "The vault access PIN is 492019.",
        "target_key": "492019",
        "retrieval_prompt": "What is the vault access PIN?",
        "context_lengths": [150, 300],
        "depth_fractions": [0.2, 0.8],
    }
    resp = client.post("/api/v1/long_context/evaluate/needle", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_trials"] == 4
    assert "accuracy_percent" in data
    assert len(data["results"]) == 4


def test_api_multi_needle():
    payload = {
        "model": "mock",
        "context_length": 250,
        "needles": [
            {"key": "Key1", "fact": "Secret Key1 is Active.", "depth_fraction": 0.3},
            {"key": "Key2", "fact": "Secret Key2 is Verified.", "depth_fraction": 0.7},
        ],
        "question": "What are Key1 and Key2?",
    }
    resp = client.post("/api/v1/long_context/evaluate/multi_needle", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_needles"] == 2
    assert "partial_score" in data


def test_api_compaction_simulate():
    payload = {
        "sequence_length": 512,
        "max_budget": 128,
        "n_sink": 4,
        "n_recent": 32,
        "n_layers": 2,
        "n_kv_heads": 2,
        "head_dim": 32,
    }
    resp = client.post("/api/v1/long_context/compaction/simulate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["max_budget"] == 128
    assert data["final_stats"]["cached_tokens"] <= 128
    assert data["final_stats"]["memory_savings_pct"] > 0
    assert len(data["timeline"]) > 0
    assert len(data["token_sample"]) > 0
