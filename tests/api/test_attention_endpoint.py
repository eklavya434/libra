"""
Tests for Attention and KV Cache API Endpoints
Phase 23: Key-Value (KV) Cache Optimization and Grouped-Query Attention (MQA/GQA)
"""

from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_attention_benchmark_endpoint():
    payload = {
        "vocab_size": 50,
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 2,
        "max_seq_len": 64,
        "prompt_len": 6,
        "max_new_tokens": 8,
    }
    response = client.post("/api/v1/attention/benchmark", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "architectures" in data
    assert "MHA" in data["architectures"]
    assert "GQA" in data["architectures"]
    assert "MQA" in data["architectures"]

    # Verify MHA, GQA, MQA properties
    mha = data["architectures"]["MHA"]
    gqa = data["architectures"]["GQA"]
    mqa = data["architectures"]["MQA"]

    assert mha["n_kv_heads"] == 4
    assert gqa["n_kv_heads"] == 2
    assert mqa["n_kv_heads"] == 1

    # Exact token match between cached and uncached runs
    assert mha["tokens_match"] is True
    assert gqa["tokens_match"] is True
    assert mqa["tokens_match"] is True

    # Memory reduction ratios
    assert mha["mha_memory_reduction_ratio"] == 1.0
    assert gqa["mha_memory_reduction_ratio"] == 2.0
    assert mqa["mha_memory_reduction_ratio"] == 4.0
    assert mqa["total_kv_memory_bytes"] < mha["total_kv_memory_bytes"]


def test_attention_generate_cached():
    payload = {
        "vocab_size": 50,
        "d_model": 32,
        "n_heads": 4,
        "n_kv_heads": 2,
        "n_layers": 2,
        "prompt_tokens": [1, 2, 3],
        "max_new_tokens": 5,
        "use_cache": True,
    }
    response = client.post("/api/v1/attention/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["use_cache"] is True
    assert len(data["generated_tokens"]) == 8
    assert len(data["new_tokens"]) == 5
    assert data["prompt_tokens"] == [1, 2, 3]


def test_attention_generate_invalid_heads():
    payload = {
        "vocab_size": 50,
        "d_model": 32,
        "n_heads": 4,
        "n_kv_heads": 3,  # Invalid: 4 % 3 != 0
        "n_layers": 2,
        "prompt_tokens": [1, 2, 3],
        "max_new_tokens": 5,
        "use_cache": True,
    }
    response = client.post("/api/v1/attention/generate", json=payload)
    assert response.status_code == 400
    assert "must be divisible" in response.json()["detail"]
