"""
Tests for Corpus Formatting and Packing API Endpoints (Phase 33)
"""

import pytest
from fastapi.testclient import TestClient
from apps.backend.main import app

client = TestClient(app)


def test_format_chat_endpoint():
    payload = {
        "messages": [
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        "max_length": 128,
    }
    response = client.post("/api/v1/corpus/format_chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "raw_chatml" in data
    assert "<|im_start|>system" in data["raw_chatml"]
    assert "<|im_start|>assistant" in data["raw_chatml"]
    assert data["total_tokens"] > 0
    assert data["trainable_tokens"] > 0
    assert data["masked_tokens"] > 0
    assert len(data["tokens"]) == data["total_tokens"]


def test_pack_endpoint():
    payload = {
        "max_length": 512,
    }
    response = client.post("/api/v1/corpus/pack", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["original_dialogues"] > 0
    assert data["packed_sequences"] >= 1
    assert data["tokens_packed"] > 0
    assert data["efficiency_gain_percent"] >= 0.0
    assert len(data["packed_batches"]) == data["packed_sequences"]


def test_sample_dataset_endpoint():
    response = client.get("/api/v1/corpus/sample_dataset")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "turns" in data[0]
    assert "chatml" in data[0]
