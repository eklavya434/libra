"""
Tests for Coder & PAL Endpoints (apps/backend/api/v1/endpoints/coder.py)

Validates:
1. POST /api/v1/coder/pal
2. POST /api/v1/coder/generate
3. POST /api/v1/coder/debug
"""

import pytest
from fastapi.testclient import TestClient

from apps.backend.main import app
from packages.providers.base import BaseProvider, ModelMetadata
from packages.providers.router import get_router


class MockCoderProvider(BaseProvider):
    """Deterministic mock provider for Coder and PAL API endpoints."""

    @property
    def name(self) -> str:
        return "coder-mock"

    async def list_models(self) -> list[ModelMetadata]:
        return []

    async def chat(self, messages, model=None, **kwargs):
        # Inspect prompt to return appropriate mock response
        if "Program-Aided Language" in messages[0]["content"] or "PAL" in messages[0]["content"]:
            # PAL request
            return {
                "choices": [
                    {"message": {"content": ("```python\ndef solution():\n    return 15 * 4\n```")}}
                ]
            }
        elif "Auto-Debugger" in messages[0]["content"]:
            # Debugging fix request
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "Fixed off-by-one error:\n"
                                "```python\n"
                                "def get_first(lst):\n"
                                "    return lst[0]\n"
                                "```"
                            )
                        }
                    }
                ]
            }
        else:
            # General code generation request
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "```python\n"
                                "def multiply(a: int, b: int) -> int:\n"
                                "    return a * b\n"
                                "```"
                            )
                        }
                    }
                ]
            }

    async def stream(self, messages, model=None, **kwargs):
        yield ""

    async def health(self):
        return {"status": "healthy"}

    async def embeddings(self, texts, model=None):
        return [[0.0] * 16]

    def capabilities(self):
        return {"supports_text": True}


@pytest.fixture
def client():
    router = get_router()
    router._providers["coder-mock"] = MockCoderProvider()
    return TestClient(app)


def test_api_pal_endpoint(client):
    payload = {
        "prompt": "What is 15 multiplied by 4?",
        "model_id": "coder-mock",
        "provider_name": "coder-mock",
        "timeout_sec": 10.0,
    }
    response = client.post("/api/v1/coder/pal", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["final_answer"] == "60"
    assert len(data["steps"]) >= 2


def test_api_code_generate_endpoint(client):
    payload = {
        "prompt": "Write a function multiply(a, b)",
        "assertions": "assert multiply(3, 4) == 12",
        "model_id": "coder-mock",
        "provider_name": "coder-mock",
        "max_iterations": 2,
    }
    response = client.post("/api/v1/coder/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "def multiply" in data["final_code"]
    assert data["total_attempts"] == 1


def test_api_code_debug_endpoint(client):
    payload = {
        "task_description": "Return the first element of a list",
        "code": "def get_first(lst):\n    return lst[1]\n",  # broken: returns 2nd element
        "assertions": "assert get_first([100, 200]) == 100",
        "model_id": "coder-mock",
        "provider_name": "coder-mock",
        "max_iterations": 3,
    }
    response = client.post("/api/v1/coder/debug", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "lst[0]" in data["final_code"]
    assert data["total_attempts"] == 2
