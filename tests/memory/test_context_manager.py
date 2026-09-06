"""
Tests for Context Window Manager (packages/core/memory/context_manager.py)
"""

from packages.core.memory.context_manager import ContextWindowManager
from packages.core.memory.models import Message


def test_token_estimation():
    mgr = ContextWindowManager(chars_per_token=4.0)
    assert mgr.estimate_tokens("") == 0
    # "Hello" is 5 chars -> ceil(5/4) = 2 + 4 = 6 tokens
    assert mgr.estimate_tokens("Hello") == 6


def test_system_prompt_preservation():
    mgr = ContextWindowManager(max_context_tokens=100, reserved_completion_tokens=20)
    messages = [
        {"role": "system", "content": "You are a concise tutor."},
        {"role": "user", "content": "Turn 1: What is 1+1?"},
        {"role": "assistant", "content": "It is 2."},
        {"role": "user", "content": "Turn 2: What is 2+2?"},
    ]

    result = mgr.prepare_context(messages)
    assert len(result.messages) > 0
    # Ensure system prompt is the first message
    assert result.messages[0]["role"] == "system"
    assert result.messages[0]["content"] == "You are a concise tutor."
    # Ensure latest user message is present
    assert result.messages[-1]["content"] == "Turn 2: What is 2+2?"


def test_sliding_window_truncation():
    # Strict budget: 60 tokens total, 10 reserved -> 50 tokens for input
    mgr = ContextWindowManager(max_context_tokens=60, reserved_completion_tokens=10, chars_per_token=4.0)

    # Each message here will be ~15-20 tokens
    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Very long user turn number 1 that consumes lots of tokens."},
        {"role": "assistant", "content": "Very long assistant response turn 1 that consumes tokens."},
        {"role": "user", "content": "Latest user question?"},
    ]

    result = mgr.prepare_context(messages)
    # The oldest dialogue turns should be dropped
    assert result.truncated_count > 0
    # System prompt remains
    assert result.messages[0]["role"] == "system"
    # Latest message remains
    assert result.messages[-1]["content"] == "Latest user question?"
    # Total input tokens fits within budget (<= 50)
    assert result.total_tokens <= 50


def test_override_system_prompt():
    mgr = ContextWindowManager(max_context_tokens=200, reserved_completion_tokens=50)
    messages = [
        {"role": "system", "content": "Original prompt"},
        {"role": "user", "content": "Hi"},
    ]

    result = mgr.prepare_context(messages, override_system_prompt="Overridden system prompt!")
    assert result.messages[0]["role"] == "system"
    assert result.messages[0]["content"] == "Overridden system prompt!"


def test_message_objects_supported():
    mgr = ContextWindowManager(max_context_tokens=500, reserved_completion_tokens=100)
    msg1 = Message(id="1", conversation_id="c1", role="user", content="Hello world", token_count=5)
    msg2 = Message(id="2", conversation_id="c1", role="assistant", content="Hello back", token_count=6)

    result = mgr.prepare_context([msg1, msg2])
    assert len(result.messages) == 2
    assert result.messages[0]["content"] == "Hello world"
    assert result.messages[1]["content"] == "Hello back"
