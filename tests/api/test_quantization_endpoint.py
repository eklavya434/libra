"""
Tests for Quantization REST Endpoints
Phase 24: Quantization (INT8 / INT4 & Post-Training Quantization)
"""

from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_quantization_benchmark_endpoint():
    payload = {
        "vocab_size": 50,
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 2,
        "seq_len": 8,
        "per_channel": True,
    }
    response = client.post("/api/v1/quantization/benchmark", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "benchmark" in data

    bench = data["benchmark"]
    assert "fp32_baseline" in bench
    assert "quantized_tiers" in bench

    int8_tier = bench["quantized_tiers"]["INT8"]
    int4_tier = bench["quantized_tiers"]["INT4"]

    # Check that quantized models use less memory than FP32
    assert int8_tier["total_bytes"] < bench["fp32_baseline"]["total_bytes"]
    assert int4_tier["total_bytes"] < int8_tier["total_bytes"]

    # High fidelity preservation
    assert int8_tier["cosine_similarity"] > 0.98
    assert int4_tier["cosine_similarity"] > 0.90


def test_quantization_convert_endpoint():
    payload = {
        "vocab_size": 50,
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 2,
        "mode": "int8",
        "per_channel": True,
    }
    response = client.post("/api/v1/quantization/convert", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["mode"] == "int8"
    assert "memory_audit" in data
    assert "fidelity_audit" in data
    assert data["fidelity_audit"]["cosine_similarity"] > 0.98
