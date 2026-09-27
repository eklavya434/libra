"""
Libra API v1 - Chat Completions Endpoint

Supports non-streaming and Server-Sent Events (SSE) streaming chat completions
routed through Ollama, Local Transformer (PyTorch), or Mock providers,
with optional conversation memory persistence and context window management.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from packages.core.memory import (
    ContextWindowManager,
    get_conversation_store,
)
from packages.providers.errors import (
    LibraProviderError,
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderQuotaExceededError,
    ProviderRateLimitError,
)
from packages.providers.router import get_router

logger = logging.getLogger("libra")

router = APIRouter()


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(
        ..., description="Role of the author: system, user, or assistant"
    )
    content: str = Field(..., min_length=1, description="Message text content")


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, description="Conversation history")
    model: str = Field("libra-llama-tied", description="Target model ID")
    provider: Optional[str] = Field(
        None, description="Optional explicit provider: ollama, libra_lab, mock-provider"
    )
    conversation_id: Optional[str] = Field(
        None, description="Optional persistent conversation session ID"
    )
    system_prompt: Optional[str] = Field(None, description="Optional system prompt override")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling randomness")
    max_tokens: int = Field(512, ge=1, le=4096, description="Max generated tokens")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    top_k: int = Field(40, ge=1, description="Top-k tokens consideration")
    stop: Optional[list[str]] = Field(None, description="Optional stop sequences")
    stream: bool = Field(False, description="Whether to stream back response tokens via SSE")
    use_rag: bool = Field(
        False, description="Whether to ground the user prompt with RAG knowledge base search"
    )
    rag_top_k: int = Field(
        3, ge=1, le=10, description="Top-k chunks to retrieve if use_rag is True"
    )


@router.post("/chat/completions", summary="Create chat completion (streaming or standard)")
async def create_chat_completion(request: ChatCompletionRequest, fastapi_request: Request) -> Any:
    session_id: str = fastapi_request.state.session_id
    router_instance = get_router()
    try:
        provider = await router_instance.resolve_provider_for_model(
            model_id=request.model,
            requested_provider=request.provider,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    context_manager = ContextWindowManager(
        max_context_tokens=2048,
        reserved_completion_tokens=request.max_tokens,
    )

    store = get_conversation_store()
    active_conv_id = request.conversation_id or f"conv-{uuid.uuid4().hex[:12]}"

    # The client-supplied conversation id must belong to this session (or be a
    # legacy owner-less row). Distinguish "does not exist" from "not yours".
    existing_owner = store.conversation_owner(active_conv_id)
    if existing_owner is not None and existing_owner != session_id:
        raise HTTPException(
            status_code=403, detail="Conversation is not accessible to this session"
        )

    # Handle persistent conversation memory
    conv = store.get_conversation(active_conv_id, owner_id=session_id)
    if not conv:
        # Determine title from the first non-empty user message
        title = "New Conversation"
        for m in request.messages:
            if m.role == "user" and m.content.strip():
                clean_t = m.content.strip().split("\n")[0][:40]
                if clean_t:
                    title = clean_t
                break
        store.create_conversation(
            conv_id=active_conv_id,
            title=title,
            model=request.model,
            system_prompt=request.system_prompt,
            owner_id=session_id,
        )

    # Append incoming user turn if present
    last_req_msg = request.messages[-1]
    if last_req_msg.role == "user":
        db_messages = store.get_messages(active_conv_id, owner_id=session_id)
        # Check if this request is a "Regenerate" action from the frontend.
        # A regenerate occurs when the client re-sends the last user message, but
        # intentionally strips the trailing assistant answer that was stored in the DB.
        if (
            db_messages
            and db_messages[-1].role == "assistant"
            and len(db_messages) >= 2
            and db_messages[-2].role == "user"
            and db_messages[-2].content == last_req_msg.content
            and not any(
                m.role == "assistant" and m.content == db_messages[-1].content
                for m in request.messages
            )
        ):
            # Remove the superseded assistant message so the new generation replaces it cleanly
            store.delete_message(db_messages[-1].id)
            db_messages = store.get_messages(active_conv_id, owner_id=session_id)

        # Only append the user turn if it is not ALREADY the most recent message in the DB.
        # If the last message in DB was an assistant message, this incoming user turn is
        # unequivocally a NEW conversational turn (even if the user repeated the same text, e.g. "hii").
        if (
            not db_messages
            or db_messages[-1].role != "user"
            or db_messages[-1].content != last_req_msg.content
        ):
            store.add_message(
                conversation_id=active_conv_id,
                role=last_req_msg.role,
                content=last_req_msg.content,
                owner_id=session_id,
            )

    # Sync conversation model if user switched models in UI
    conv = store.get_conversation(active_conv_id, owner_id=session_id)
    if conv and conv.model != request.model:
        store.update_conversation(active_conv_id, model=request.model, owner_id=session_id)

    # Retrieve full conversation history from persistent store
    history = store.get_messages(active_conv_id, owner_id=session_id)
    active_system_prompt = (
        request.system_prompt
        or (conv.system_prompt if conv else None)
        or "You are Libra, an intelligent, helpful, and friendly AI assistant. Answer conversationally in Markdown."
    )
    processed_context = context_manager.prepare_context(
        messages=history,
        override_system_prompt=active_system_prompt,
    )
    messages_to_send = processed_context.messages

    # Apply RAG knowledge base context grounding if requested
    if request.use_rag and request.messages and request.messages[-1].role == "user":
        from packages.rag import RAGPromptSynthesizer, get_hybrid_retriever

        retriever = get_hybrid_retriever()
        last_query = request.messages[-1].content
        rag_matches = retriever.search(query=last_query, top_k=request.rag_top_k, mode="hybrid")
        if rag_matches and messages_to_send and messages_to_send[-1]["role"] == "user":
            synthesizer = RAGPromptSynthesizer()
            grounded_prompt = synthesizer.build_grounded_prompt(last_query, rag_matches)
            messages_to_send[-1]["content"] = grounded_prompt

    if not request.stream:
        try:
            response = await provider.chat(
                messages=messages_to_send,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                top_k=request.top_k,
                stop=request.stop,
            )
            if isinstance(response, dict):
                response["conversation_id"] = active_conv_id
                if "choices" in response:
                    assistant_text = response["choices"][0].get("message", {}).get("content", "")
                    if assistant_text:
                        store.add_message(
                            conversation_id=active_conv_id,
                            role="assistant",
                            content=assistant_text,
                            owner_id=session_id,
                        )
            return response
        except Exception as e:
            # Map provider failures to honest, non-leaking client errors. Full
            # exception details are logged server-side with the request id.
            request_id = getattr(fastapi_request.state, "request_id", None)
            if isinstance(e, ProviderAuthenticationError):
                raise HTTPException(
                    status_code=503,
                    detail="The selected provider's API key is invalid, expired, or not configured on this deployment.",
                )
            if isinstance(e, (ProviderRateLimitError, ProviderQuotaExceededError)):
                raise HTTPException(
                    status_code=429,
                    detail="The selected provider is rate-limited or its free quota is exhausted. Try another model or retry later.",
                )
            if isinstance(e, ModelNotFoundError):
                raise HTTPException(
                    status_code=404,
                    detail="The selected model is not recognized by the provider. Check the model name and provider configuration.",
                )
            if isinstance(e, LibraProviderError):
                raise HTTPException(
                    status_code=503,
                    detail=str(getattr(e, "message", e) or "The selected provider is unavailable."),
                )
            if isinstance(e, (ConnectionError, TimeoutError, OSError)):
                logger.warning(
                    "Provider connection failure (request_id=%s, model=%s): %s",
                    request_id,
                    request.model,
                    str(e),
                )
                raise HTTPException(
                    status_code=503,
                    detail="The selected provider is unreachable. Check network connectivity or try another model.",
                )
            logger.exception(
                "Non-streaming chat failed (request_id=%s, model=%s)",
                request_id,
                request.model,
            )
            raise HTTPException(
                status_code=500,
                detail="Inference error: the request could not be completed. See the server logs for details.",
            )

    # Streaming mode via Server-Sent Events (SSE)
    async def event_generator():
        accumulated_chunks = []
        try:
            # Yield conversation session metadata chunk
            meta_chunk = {
                "conversation_id": active_conv_id,
                "model": request.model,
            }
            yield f"data: {json.dumps(meta_chunk)}\n\n"

            async for token in provider.stream(
                messages=messages_to_send,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                top_k=request.top_k,
                stop=request.stop,
            ):
                accumulated_chunks.append(token)
                chunk = {
                    "choices": [
                        {
                            "delta": {"content": token},
                            "index": 0,
                            "finish_reason": None,
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # Stream finished successfully: persist assistant output
            if accumulated_chunks:
                full_text = "".join(accumulated_chunks)
                store.add_message(
                    conversation_id=active_conv_id,
                    role="assistant",
                    content=full_text,
                    owner_id=session_id,
                )

            yield "data: [DONE]\n\n"
        except Exception as e:
            request_id = getattr(fastapi_request.state, "request_id", None)
            if isinstance(e, ProviderAuthenticationError):
                user_msg = "The selected provider's API key is invalid, expired, or not configured on this deployment."
            elif isinstance(e, (ProviderRateLimitError, ProviderQuotaExceededError)):
                user_msg = "The selected provider is rate-limited or its free quota is exhausted. Try another model or retry later."
            elif isinstance(e, ModelNotFoundError):
                user_msg = "The selected model is not recognized by the provider. Check the model name and provider configuration."
            elif isinstance(e, LibraProviderError):
                user_msg = str(getattr(e, "message", e) or "The selected provider is unavailable.")
            elif isinstance(e, (ConnectionError, TimeoutError, OSError)):
                logger.warning(
                    "Provider connection failure (request_id=%s, model=%s): %s",
                    request_id,
                    request.model,
                    str(e),
                )
                user_msg = "The selected provider is unreachable. Check network connectivity or try another model."
            else:
                logger.exception(
                    "Streaming chat failed (request_id=%s, model=%s)",
                    request_id,
                    request.model,
                )
                user_msg = "Inference error: the request failed unexpectedly. See the server logs for details."
            err_chunk = {"error": user_msg}
            yield f"data: {json.dumps(err_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    headers = {"X-Conversation-Id": active_conv_id}
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)
