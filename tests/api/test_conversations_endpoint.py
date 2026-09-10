"""
Tests for Conversations REST API & Chat Persistence Integration
"""

import pytest
from fastapi.testclient import TestClient

import packages.core.memory.sqlite_store as sqlite_module
from apps.backend.main import app
from packages.core.memory import SQLiteConversationStore


@pytest.fixture(autouse=True)
def isolated_memory_store(monkeypatch):
    """Overrides the global conversation store with an in-memory SQLite store for tests."""
    test_store = SQLiteConversationStore(db_path=":memory:")
    monkeypatch.setattr(sqlite_module, "_global_store", test_store)
    return test_store


@pytest.fixture
def client():
    return TestClient(app)


def test_create_and_list_conversations(client):
    res = client.post(
        "/api/v1/conversations",
        json={"title": "Test Chat", "model": "libra-mock-v1", "system_prompt": "Act as a tester."},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["title"] == "Test Chat"
    conv_id = data["id"]

    list_res = client.get("/api/v1/conversations")
    assert list_res.status_code == 200
    conv_list = list_res.json()
    assert len(conv_list) >= 1
    assert any(c["id"] == conv_id for c in conv_list)


def test_get_conversation_detail(client):
    res = client.post("/api/v1/conversations", json={"title": "Detail Chat"})
    conv_id = res.json()["id"]

    # Append a message
    msg_res = client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"role": "user", "content": "Hello testing"},
    )
    assert msg_res.status_code == 200

    detail_res = client.get(f"/api/v1/conversations/{conv_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["id"] == conv_id
    assert len(detail["messages"]) == 1
    assert detail["messages"][0]["content"] == "Hello testing"


def test_append_message_requires_existing_conversation(client):
    response = client.post(
        "/api/v1/conversations/does-not-exist/messages",
        json={"role": "user", "content": "This must not create a conversation."},
    )

    assert response.status_code == 404


def test_update_and_delete_conversation(client):
    res = client.post("/api/v1/conversations", json={"title": "Old Name"})
    conv_id = res.json()["id"]

    # Update
    patch_res = client.patch(
        f"/api/v1/conversations/{conv_id}",
        json={"title": "New Renamed Title"},
    )
    assert patch_res.status_code == 200
    assert patch_res.json()["title"] == "New Renamed Title"

    # Delete
    del_res = client.delete(f"/api/v1/conversations/{conv_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # Verify 404
    get_res = client.get(f"/api/v1/conversations/{conv_id}")
    assert get_res.status_code == 404


def test_chat_completion_with_persistent_conversation(client):
    # Create conversation
    conv_res = client.post(
        "/api/v1/conversations", json={"title": "Persistent Chat", "model": "libra-mock-v1"}
    )
    conv_id = conv_res.json()["id"]

    # Send chat completion request targeting this conversation
    chat_res = client.post(
        "/api/v1/chat/completions",
        json={
            "conversation_id": conv_id,
            "model": "libra-mock-v1",
            "messages": [{"role": "user", "content": "Tell me a joke."}],
            "stream": False,
        },
    )
    assert chat_res.status_code == 200

    # Verify conversation now has both user message and assistant message stored!
    detail_res = client.get(f"/api/v1/conversations/{conv_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert len(detail["messages"]) >= 2
    assert detail["messages"][0]["role"] == "user"
    assert detail["messages"][0]["content"] == "Tell me a joke."
    assert detail["messages"][1]["role"] == "assistant"
    assert len(detail["messages"][1]["content"]) > 0
