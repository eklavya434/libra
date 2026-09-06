"""
Tests for Autonomous Agent Endpoints (apps/backend/api/v1/endpoints/agents.py)
"""

import json
import pytest
from fastapi.testclient import TestClient
from apps.backend.main import app
from packages.providers.base import BaseProvider, ModelMetadata
from packages.providers.router import get_router


class SimpleAgentMockProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "simple-agent-mock"

    async def list_models(self):
        return []

    async def chat(self, messages, model=None, **kwargs):
        last_content = messages[-1]["content"]
        if "master planning agent" in last_content:
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"problem_summary": "Task", "steps": ["Step 1", "Step 2"]}'
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "content": "Thought: Direct calculation.\nFinal Answer: Agent completed task successfully."
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
    router._providers["simple-agent-mock"] = SimpleAgentMockProvider()
    return TestClient(app)


def test_react_agent_endpoint(client):
    res = client.post(
        "/api/v1/agents/react",
        json={
            "prompt": "Solve 2 + 2",
            "provider_name": "simple-agent-mock",
            "model_id": "simple-agent-mock",
            "max_steps": 3,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert "Agent completed task successfully." in data["final_answer"]
    assert len(data["steps"]) >= 1


def test_plan_and_solve_endpoint(client):
    res = client.post(
        "/api/v1/agents/plan-and-solve",
        json={
            "prompt": "Plan and solve multi-step task",
            "provider_name": "simple-agent-mock",
            "model_id": "simple-agent-mock",
            "max_steps": 5,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert len(data["steps"]) >= 2


def test_react_agent_streaming_endpoint(client):
    res = client.post(
        "/api/v1/agents/react/stream",
        json={
            "prompt": "Stream test",
            "provider_name": "simple-agent-mock",
            "model_id": "simple-agent-mock",
            "max_steps": 3,
        },
    )
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    text = res.text
    assert "data: " in text
    assert '"type": "start"' in text
    assert '"type": "finish"' in text
