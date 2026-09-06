"""
Tests for Structured Output Endpoints (apps/backend/api/v1/endpoints/structured.py)
"""

import pytest
from fastapi.testclient import TestClient
from apps.backend.main import app
from packages.providers.base import BaseProvider, ModelMetadata
from packages.providers.router import get_router


class MockJSONProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "mock-json"

    async def list_models(self):
        return []

    async def chat(self, messages, model=None, **kwargs):
        # Return valid JSON conforming to schema
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"sentiment": "positive", "confidence": 0.98}'
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
    # Register mock JSON provider into global router
    router = get_router()
    router._providers["mock-json"] = MockJSONProvider()
    return TestClient(app)


def test_validate_structured_endpoint(client):
    schema = {
        "title": "SentimentSchema",
        "type": "object",
        "properties": {
            "sentiment": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["sentiment", "confidence"],
    }

    # 1. Valid test
    res_valid = client.post(
        "/api/v1/structured/validate",
        json={
            "text": '{"sentiment": "positive", "confidence": 0.95}',
            "schema_dict": schema,
        },
    )
    assert res_valid.status_code == 200
    data = res_valid.json()
    assert data["is_valid"] is True
    assert data["error"] is None
    assert data["parsed_data"]["sentiment"] == "positive"

    # 2. Missing required field
    res_missing = client.post(
        "/api/v1/structured/validate",
        json={
            "text": '{"sentiment": "positive"}',
            "schema_dict": schema,
        },
    )
    assert res_missing.status_code == 200
    assert res_missing.json()["is_valid"] is False
    assert "Missing required property: 'confidence'" in res_missing.json()["error"]


def test_generate_structured_endpoint(client):
    schema = {
        "title": "SentimentSchema",
        "type": "object",
        "properties": {
            "sentiment": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["sentiment", "confidence"],
    }

    res = client.post(
        "/api/v1/structured/generate",
        json={
            "prompt": "Analyze: I love this educational lab!",
            "schema_dict": schema,
            "provider_name": "mock-json",
            "model_id": "mock-json",
            "max_retries": 1,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["data"]["sentiment"] == "positive"
    assert data["data"]["confidence"] == 0.98
    assert data["attempts"] == 1
