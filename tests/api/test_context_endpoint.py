"""
API Tests for Phase 30: Context & RoPE Scaling Endpoints
"""

import pytest
from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_rope_scaling_endpoint_yarn():
    payload = {
        "dim": 64,
        "max_seq_len": 2048,
        "scale": 4.0,
        "original_max_seq_len": 512,
        "theta_base": 10000.0,
        "scaling_type": "yarn",
    }
    response = client.post("/api/v1/context/scale", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["scaling_type"] == "yarn"
    assert data["scale"] == 4.0
    assert len(data["channels"]) == 32
    assert "attn_temperature_factor" in data
    assert data["attn_temperature_factor"] > 0.0


def test_rope_scaling_endpoint_dynamic_ntk():
    payload = {
        "dim": 64,
        "max_seq_len": 1024,
        "scale": 2.0,
        "original_max_seq_len": 512,
        "theta_base": 10000.0,
        "scaling_type": "dynamic_ntk",
    }
    response = client.post("/api/v1/context/scale", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["scaling_type"] == "dynamic_ntk"
    assert len(data["channels"]) == 32


def test_needle_evaluation_endpoint():
    payload = {
        "model": "mock",
        "context_lengths": [100, 200],
        "depth_fractions": [0.0, 0.5, 1.0],
    }
    response = client.post("/api/v1/context/needle", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "mock"
    assert data["total_trials"] == 6
    assert data["accuracy_percent"] == 100.0
    assert len(data["results"]) == 6


def test_perplexity_scaling_endpoint():
    payload = {
        "scale_factors": [1.0, 2.0, 4.0],
        "scaling_type": "yarn",
    }
    response = client.post("/api/v1/context/perplexity", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["scaling_type"] == "yarn"
    assert len(data["points"]) == 3
