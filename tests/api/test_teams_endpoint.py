"""
Tests for Multi-Agent Teams Endpoints (apps/backend/api/v1/endpoints/teams.py)

Validates:
1. POST /api/v1/teams/collaborate
2. POST /api/v1/teams/collaborate/stream
"""

import pytest
from fastapi.testclient import TestClient

from apps.backend.main import app
from packages.providers.base import BaseProvider, ModelMetadata
from packages.providers.router import get_router


class MockTeamProvider(BaseProvider):
    """Deterministic mock provider for multi-agent REST endpoints."""

    @property
    def name(self) -> str:
        return "api-team-mock"

    async def list_models(self) -> list[ModelMetadata]:
        return []

    async def chat(self, messages, model=None, **kwargs):
        sys_prompt = messages[0]["content"] if messages else ""

        if "Adversarial Senior Code Reviewer" in sys_prompt:
            return {
                "choices": [
                    {"message": {"content": "Code conforms to specification.\nVERDICT: APPROVED"}}
                ]
            }
        elif "Lead Software Architect" in sys_prompt:
            return {
                "choices": [{"message": {"content": "Specification: Define double(x: int) -> int"}}]
            }
        elif "Senior Implementation Engineer" in sys_prompt:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "```python\ndef double(x: int) -> int:\n    return x * 2\n```"
                        }
                    }
                ]
            }
        elif (
            "Automated QA and Testing Specialist" in sys_prompt
            or "Testing Specialist" in sys_prompt
        ):
            return {"choices": [{"message": {"content": "```python\nassert double(5) == 10\n```"}}]}
        else:
            return {"choices": [{"message": {"content": "Default response."}}]}

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
    router._providers["api-team-mock"] = MockTeamProvider()
    return TestClient(app)


def test_api_teams_collaborate(client):
    payload = {
        "prompt": "Write a function to double an integer",
        "model_id": "api-team-mock",
        "provider_name": "api-team-mock",
        "max_rounds": 2,
    }
    response = client.post("/api/v1/teams/collaborate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["consensus_reached"] is True
    assert "def double" in data["final_code"]
    assert len(data["rounds"]) >= 1


def test_api_teams_collaborate_stream(client):
    payload = {
        "prompt": "Write a function to double an integer",
        "model_id": "api-team-mock",
        "provider_name": "api-team-mock",
        "max_rounds": 1,
    }
    response = client.post("/api/v1/teams/collaborate/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    text = response.text
    assert "data: " in text
    assert '"type": "start"' in text
    assert '"type": "finish"' in text
