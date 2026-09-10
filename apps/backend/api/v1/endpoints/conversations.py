"""
Libra API v1 - Conversations & Session Management Endpoints

Provides RESTful CRUD operations for multi-turn conversation sessions,
history retrieval, and message management.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from packages.core.memory import (
    AddMessageRequest,
    Conversation,
    ConversationDetail,
    CreateConversationRequest,
    Message,
    UpdateConversationRequest,
    get_conversation_store,
)

router = APIRouter()


@router.get("", response_model=list[Conversation], summary="List conversation sessions")
async def list_conversations(
    limit: int = Query(50, ge=1, le=100, description="Max conversations to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
) -> list[Conversation]:
    """Retrieve all conversations sorted by last updated timestamp."""
    store = get_conversation_store()
    return store.list_conversations(limit=limit, offset=offset)


@router.post("", response_model=Conversation, summary="Create a new conversation session")
async def create_conversation(request: CreateConversationRequest) -> Conversation:
    """Initialize a brand new conversation session."""
    store = get_conversation_store()
    return store.create_conversation(
        title=request.title,
        model=request.model,
        system_prompt=request.system_prompt,
    )


@router.get(
    "/{conv_id}", response_model=ConversationDetail, summary="Get conversation details and messages"
)
async def get_conversation(conv_id: str) -> ConversationDetail:
    """Fetch a single conversation session with all chronologically ordered messages."""
    store = get_conversation_store()
    conv = store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail=f"Conversation '{conv_id}' not found")
    return conv


@router.patch(
    "/{conv_id}", response_model=Conversation, summary="Update conversation settings or title"
)
async def update_conversation(conv_id: str, request: UpdateConversationRequest) -> Conversation:
    """Update title, model, or system prompt for an existing conversation."""
    store = get_conversation_store()
    updated = store.update_conversation(
        conv_id=conv_id,
        title=request.title,
        model=request.model,
        system_prompt=request.system_prompt,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Conversation '{conv_id}' not found")
    return updated


@router.delete("/{conv_id}", summary="Delete conversation session")
async def delete_conversation(conv_id: str) -> dict[str, Any]:
    """Delete a conversation and cascade delete all its stored messages."""
    store = get_conversation_store()
    success = store.delete_conversation(conv_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Conversation '{conv_id}' not found")
    return {"status": "deleted", "id": conv_id}


@router.post(
    "/{conv_id}/messages", response_model=Message, summary="Append a message to a conversation"
)
async def append_message(conv_id: str, request: AddMessageRequest) -> Message:
    """Directly append a message to an existing conversation."""
    store = get_conversation_store()
    if not store.get_conversation(conv_id):
        raise HTTPException(status_code=404, detail=f"Conversation '{conv_id}' not found")
    return store.add_message(
        conversation_id=conv_id,
        role=request.role,
        content=request.content,
        token_count=request.token_count or 0,
    )


@router.delete("/{conv_id}/messages", summary="Clear all messages in conversation")
async def clear_messages(conv_id: str) -> dict[str, Any]:
    """Clear message history in a conversation without removing the conversation session."""
    store = get_conversation_store()
    conv = store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail=f"Conversation '{conv_id}' not found")
    store.clear_messages(conv_id)
    return {"status": "cleared", "id": conv_id}
