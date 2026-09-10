"""
Libra API v1 - Chat Completions Endpoint

Supports non-streaming and Server-Sent Events (SSE) streaming chat completions
routed through Ollama, Local Transformer (PyTorch), or Mock providers,
with optional conversation memory persistence and context window management.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from packages.core.memory import (
    ContextWindowManager,
    get_conversation_store,
)
from packages.providers.router import get_router

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
async def create_chat_completion(request: ChatCompletionRequest) -> Any:
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

    # Handle persistent conversation memory
    conv = store.get_conversation(active_conv_id)
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
        )

    # Append incoming user turn if present
    last_req_msg = request.messages[-1]
    if last_req_msg.role == "user":
        db_messages = store.get_messages(active_conv_id)
        # Avoid duplicate insertion if caller passed full history
        if (
            not db_messages
            or db_messages[-1].content != last_req_msg.content
            or db_messages[-1].role != "user"
        ):
            store.add_message(
                conversation_id=active_conv_id,
                role=last_req_msg.role,
                content=last_req_msg.content,
            )

    # Sync conversation model if user switched models in UI
    conv = store.get_conversation(active_conv_id)
    if conv and conv.model != request.model:
        store.update_conversation(active_conv_id, model=request.model)

    # Retrieve full conversation history from persistent store
    history = store.get_messages(active_conv_id)
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
                        )
            return response
        except ConnectionError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

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
                )

            yield "data: [DONE]\n\n"
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate" in msg.lower():
                user_msg = (
                    "Provider rate limit or daily free tier quota exhausted. "
                    "Please switch to 'libra-mock-v1' or an Ollama local model, or try again later."
                )
            elif "401" in msg or "api_key" in msg.lower() or "auth" in msg.lower():
                user_msg = "Provider authentication error. Please verify your API key in .env or switch to a local model."
            else:
                user_msg = f"Inference error: {msg}"
            err_chunk = {"error": user_msg}
            yield f"data: {json.dumps(err_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    headers = {"X-Conversation-Id": active_conv_id}
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)
