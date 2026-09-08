"""
API Tests for Phase 32: Multi-Modal Endpoints
"""

import pytest
from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_embed_image_endpoint():
    payload = {
        "pattern": "checkerboard",
        "image_size": 32,
        "patch_size": 8,
        "vision_dim": 32,
    }
    response = client.post("/api/v1/multimodal/embed_image", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["image_size"] == 32
    assert data["patch_size"] == 8
    assert data["num_patches"] == 16
    assert len(data["patch_grid"]) == 16


def test_vlm_generate_endpoint():
    payload = {
        "prompt": "This image shows a",
        "pattern": "checkerboard",
        "max_new_tokens": 4,
    }
    response = client.post("/api/v1/multimodal/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["num_visual_tokens"] == 16
    assert len(data["generated_tokens"]) == 4
    assert len(data["generated_text"]) > 0
    assert data["latency_ms"] >= 0.0
