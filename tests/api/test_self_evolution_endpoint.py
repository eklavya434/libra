"""
Tests for Self-Evolution & Grand Capstone API Endpoints (Phase 50)
"""

from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_presets_endpoint():
    res = client.get("/api/v1/self-evolution/presets")
    assert res.status_code == 200
    data = res.json()
    assert "curricula" in data
    assert "stages" in data
    assert len(data["stages"]) == 6


def test_single_cycle_endpoint():
    req = {
        "seed_prompt": "Explain gradient descent in machine learning.",
        "cycle_id": 1,
        "candidates_per_prompt": 2,
        "dpo_beta": 0.1,
    }
    res = client.post("/api/v1/self-evolution/cycle", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["cycle_id"] == 1
    assert data["candidates_generated"] == 2
    assert "dpo_loss" in data
    assert "pre_evolution_reasoning_acc" in data
    assert "post_evolution_reasoning_acc" in data


def test_grand_audit_endpoint():
    res = client.get("/api/v1/self-evolution/grand-audit")
    assert res.status_code == 200
    data = res.json()
    assert data["total_pillars"] == 10
    assert data["pillars_passed"] == 10
    assert data["all_passed"] is True
    assert data["completion_score_pct"] == 100.0
    assert "SUMMA CUM LAUDE" in data["graduation_honors"]
