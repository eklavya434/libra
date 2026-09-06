import json
import pytest
from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_chat_completion_non_streaming():
    payload = {
        "messages": [
            {"role": "user", "content": "Hello, can you help me?"}
        ],
        "model": "libra-mock-v1",
        "provider": "mock-provider",
        "temperature": 0.5,
        "max_tokens": 100,
        "stream": False,
    }

    resp = client.post("/api/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "content" in data["choices"][0]["message"]
    assert "usage" in data


def test_chat_completion_streaming():
    payload = {
        "messages": [
            {"role": "user", "content": "Tell me a short story."}
        ],
        "model": "libra-mock-v1",
        "provider": "mock-provider",
        "stream": True,
    }

    resp = client.post("/api/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    # Verify event stream contents
    lines = resp.text.strip().split("\n\n")
    data_lines = [l for l in lines if l.startswith("data: ")]
    assert len(data_lines) > 0
    assert "[DONE]" in resp.text


def test_chat_completion_invalid_request():
    # Empty messages list should fail validation
    payload = {
        "messages": [],
        "model": "libra-mock-v1",
    }
    resp = client.post("/api/v1/chat/completions", json=payload)
    assert resp.status_code == 422
