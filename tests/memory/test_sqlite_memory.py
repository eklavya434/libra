"""
Tests for SQLite Conversation Store (packages/core/memory/sqlite_store.py)
"""

import pytest
from packages.core.memory.sqlite_store import SQLiteConversationStore


@pytest.fixture
def memory_store():
    """Returns an isolated in-memory SQLite store."""
    return SQLiteConversationStore(db_path=":memory:")


def test_create_and_get_conversation(memory_store):
    conv = memory_store.create_conversation(
        title="Test Conversation",
        model="libra-llama-tied",
        system_prompt="You are a helpful assistant.",
    )
    assert conv.id is not None
    assert conv.title == "Test Conversation"
    assert conv.model == "libra-llama-tied"
    assert conv.system_prompt == "You are a helpful assistant."

    detail = memory_store.get_conversation(conv.id)
    assert detail is not None
    assert detail.id == conv.id
    assert detail.title == "Test Conversation"
    assert detail.messages == []
    assert detail.message_count == 0


def test_list_conversations(memory_store):
    c1 = memory_store.create_conversation(title="First")
    c2 = memory_store.create_conversation(title="Second")

    conversations = memory_store.list_conversations()
    assert len(conversations) == 2
    ids = [c.id for c in conversations]
    assert c1.id in ids
    assert c2.id in ids


def test_update_conversation(memory_store):
    conv = memory_store.create_conversation(title="Initial Title")
    updated = memory_store.update_conversation(
        conv_id=conv.id,
        title="Updated Title",
        model="deepseek-r1-mock",
    )
    assert updated is not None
    assert updated.title == "Updated Title"
    assert updated.model == "deepseek-r1-mock"

    detail = memory_store.get_conversation(conv.id)
    assert detail.title == "Updated Title"


def test_add_and_get_messages(memory_store):
    conv = memory_store.create_conversation(title="Chat Session")
    m1 = memory_store.add_message(conv.id, role="user", content="Hello!")
    m2 = memory_store.add_message(conv.id, role="assistant", content="Hi, how can I help?")

    assert m1.role == "user"
    assert m2.role == "assistant"

    messages = memory_store.get_messages(conv.id)
    assert len(messages) == 2
    assert messages[0].content == "Hello!"
    assert messages[1].content == "Hi, how can I help?"

    detail = memory_store.get_conversation(conv.id)
    assert detail.message_count == 2
    assert len(detail.messages) == 2


def test_auto_title_from_first_user_message(memory_store):
    conv = memory_store.create_conversation()  # default "New Conversation"
    assert conv.title == "New Conversation"

    memory_store.add_message(conv.id, role="user", content="What is the circumference of Earth?")
    detail = memory_store.get_conversation(conv.id)
    assert detail.title == "What is the circumference of Earth?"


def test_delete_conversation_cascades_messages(memory_store):
    conv = memory_store.create_conversation(title="To Delete")
    memory_store.add_message(conv.id, role="user", content="Message 1")
    memory_store.add_message(conv.id, role="assistant", content="Message 2")

    assert len(memory_store.get_messages(conv.id)) == 2

    success = memory_store.delete_conversation(conv.id)
    assert success is True
    assert memory_store.get_conversation(conv.id) is None
    assert memory_store.get_messages(conv.id) == []


def test_clear_messages(memory_store):
    conv = memory_store.create_conversation(title="Keep Conversation")
    memory_store.add_message(conv.id, role="user", content="Message 1")
    assert len(memory_store.get_messages(conv.id)) == 1

    memory_store.clear_messages(conv.id)
    detail = memory_store.get_conversation(conv.id)
    assert detail is not None
    assert detail.messages == []
    assert detail.message_count == 0
