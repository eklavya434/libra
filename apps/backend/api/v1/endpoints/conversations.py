"""
Libra API v1 - Conversations & Session Management Endpoints

Provides RESTful CRUD operations for multi-turn conversation sessions,
history retrieval, and message management.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

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


def _session_id(request: Request) -> str:
    """The guest session identity resolved by the identity middleware."""
    return request.state.session_id


@router.get("", response_model=list[Conversation], summary="List conversation sessions")
async def list_conversations(
    request: Request,
    limit: int = Query(50, ge=1, le=100, description="Max conversations to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
) -> list[Conversation]:
    """Retrieve conversations visible to the current session, newest first."""
    store = get_conversation_store()
    return store.list_conversations(limit=limit, offset=offset, owner_id=_session_id(request))


@router.post("", response_model=Conversation, summary="Create a new conversation session")
async def create_conversation(request: Request, payload: CreateConversationRequest) -> Conversation:
    """Initialize a brand new conversation session owned by the current session."""
    from apps.backend.api.v1.endpoints.models import get_system_default_model

    store = get_conversation_store()
    model = payload.model or get_system_default_model() or "libra-llama-tied"
    return store.create_conversation(
        title=payload.title,
        model=model,
        system_prompt=payload.system_prompt,
        owner_id=_session_id(request),
    )


@router.get(
    "/{conv_id}", response_model=ConversationDetail, summary="Get conversation details and messages"
)
async def get_conversation(request: Request, conv_id: str) -> ConversationDetail:
    """Fetch a single conversation session with all chronologically ordered messages."""
    store = get_conversation_store()
    conv = store.get_conversation(conv_id, owner_id=_session_id(request))
    if not conv:
        raise HTTPException(status_code=404, detail=f"Conversation '{conv_id}' not found")
    return conv


@router.patch(
    "/{conv_id}", response_model=Conversation, summary="Update conversation settings or title"
)
async def update_conversation(
    request: Request, conv_id: str, payload: UpdateConversationRequest
) -> Conversation:
    """Update title, model, or system prompt for an existing conversation."""
    store = get_conversation_store()
    updated = store.update_conversation(
        conv_id=conv_id,
        title=payload.title,
        model=payload.model,
        system_prompt=payload.system_prompt,
        owner_id=_session_id(request),
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Conversation '{conv_id}' not found")
    return updated


@router.delete("/{conv_id}", summary="Delete conversation session")
async def delete_conversation(request: Request, conv_id: str) -> dict[str, Any]:
    """Delete a conversation owned by the current session (or a legacy row)."""
    store = get_conversation_store()
    success = store.delete_conversation(conv_id, owner_id=_session_id(request))
    if not success:
        raise HTTPException(status_code=404, detail=f"Conversation '{conv_id}' not found")
    return {"status": "deleted", "id": conv_id}


@router.post(
    "/{conv_id}/messages", response_model=Message, summary="Append a message to a conversation"
)
async def append_message(request: Request, conv_id: str, payload: AddMessageRequest) -> Message:
    """Directly append a message to an existing conversation."""
    store = get_conversation_store()
    if not store.get_conversation(conv_id, owner_id=_session_id(request)):
        raise HTTPException(status_code=404, detail=f"Conversation '{conv_id}' not found")
    return store.add_message(
        conversation_id=conv_id,
        role=payload.role,
        content=payload.content,
        token_count=payload.token_count or 0,
        owner_id=_session_id(request),
    )


@router.delete("/{conv_id}/messages", summary="Clear all messages in conversation")
async def clear_messages(request: Request, conv_id: str) -> dict[str, Any]:
    """Clear message history in a conversation without removing the conversation session."""
    store = get_conversation_store()
    if not store.get_conversation(conv_id, owner_id=_session_id(request)):
        raise HTTPException(status_code=404, detail=f"Conversation '{conv_id}' not found")
    store.clear_messages(conv_id, owner_id=_session_id(request))
    return {"status": "cleared", "id": conv_id}
