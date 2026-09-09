"""
Libra Core Memory - Data Models & Schemas

Pydantic models representing conversation sessions, messages,
and context parameters for stateful multi-turn chat.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    """Return the current UTC timestamp formatted as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class Message(BaseModel):
    """Represents a single message in a multi-turn conversation."""

    id: str = Field(..., description="Unique message ID")
    conversation_id: str = Field(..., description="Parent conversation session ID")
    role: str = Field(..., description="Role of the sender: system, user, or assistant")
    content: str = Field(..., description="Text content of the message")
    token_count: int = Field(0, description="Estimated or exact token count")
    created_at: str = Field(default_factory=utc_now_iso, description="ISO timestamp when created")


class Conversation(BaseModel):
    """Metadata summary of a conversation session."""

    id: str = Field(..., description="Unique conversation session ID")
    title: str = Field("New Conversation", description="User-friendly title of the conversation")
    model: str = Field("libra-llama-tied", description="Default model assigned to this session")
    system_prompt: Optional[str] = Field(None, description="Optional custom system prompt")
    created_at: str = Field(default_factory=utc_now_iso, description="ISO timestamp when created")
    updated_at: str = Field(default_factory=utc_now_iso, description="ISO timestamp of last update")
    message_count: int = Field(0, description="Total number of messages in the session")


class ConversationDetail(Conversation):
    """Detailed conversation object including ordered messages."""

    messages: list[Message] = Field(
        default_factory=list, description="Ordered conversation history"
    )


class CreateConversationRequest(BaseModel):
    """Payload to initialize a new conversation."""

    title: Optional[str] = Field(None, description="Optional initial conversation title")
    model: str = Field("libra-llama-tied", description="Target model ID")
    system_prompt: Optional[str] = Field(
        None, description="Optional persistent system instructions"
    )


class UpdateConversationRequest(BaseModel):
    """Payload to update conversation properties."""

    title: Optional[str] = Field(None, description="Updated conversation title")
    model: Optional[str] = Field(None, description="Updated model ID")
    system_prompt: Optional[str] = Field(None, description="Updated system prompt")


class AddMessageRequest(BaseModel):
    """Payload to add a new message to a conversation."""

    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message text content")
    token_count: Optional[int] = Field(None, description="Optional precomputed token count")
