"""Full-stack integration tests for FastAPI, providers, and database."""

import pytest
from fastapi.testclient import TestClient

from apps.backend.main import app
from packages.core.memory.sqlite_store import SQLiteConversationStore
from packages.models.catalog import get_default_registry


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health_endpoint(client):
    """Verify health endpoint returns status ok and expected fields."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_model_registry_integration():
    """Verify model registry contains local and cloud providers with valid configs."""
    registry = get_default_registry()
    models = registry.list_models()
    assert len(models) >= 10
    model_ids = [m.model_id for m in models]
    assert "libra-mock-v1" in model_ids
    assert "gemini-2.5-flash" in model_ids


def test_chat_mock_streaming_integration(client):
    """Verify end-to-end chat completion against mock provider."""
    payload = {
        "model": "libra-mock-v1",
        "messages": [{"role": "user", "content": "Integration test prompt"}],
        "stream": False,
        "temperature": 0.0,
    }
    response = client.post("/api/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert len(data["choices"][0]["message"]["content"]) > 0


def test_database_integration_lifecycle(tmp_path):
    """Verify SQLite WAL memory creates tables and persists messages."""
    db_file = tmp_path / "test_integration.db"
    store = SQLiteConversationStore(db_path=str(db_file))
    conv = store.create_conversation(title="Integration Session")
    assert conv.id is not None

    store.add_message(conv.id, "user", "Hello database")
    store.add_message(conv.id, "assistant", "Hello user")

    detail = store.get_conversation(conv.id)
    assert detail is not None
    assert len(detail.messages) == 2
    assert detail.messages[0].content == "Hello database"
    assert detail.messages[1].content == "Hello user"
