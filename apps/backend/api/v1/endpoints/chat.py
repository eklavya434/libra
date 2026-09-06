"""
Libra API v1 - Chat Completions Endpoint

Supports non-streaming and Server-Sent Events (SSE) streaming chat completions
routed through Ollama, Local Transformer (PyTorch), or Mock providers.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from packages.providers.router import get_router

router = APIRouter()


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the author: system, user, or assistant")
    content: str = Field(..., description="Message text content")


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, description="Conversation history")
    model: str = Field("libra-llama-tied", description="Target model ID")
    provider: Optional[str] = Field(None, description="Optional explicit provider: ollama, libra_lab, mock-provider")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling randomness")
    max_tokens: int = Field(512, ge=1, le=4096, description="Max generated tokens")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    top_k: int = Field(40, ge=1, description="Top-k tokens consideration")
    stop: Optional[list[str]] = Field(None, description="Optional stop sequences")
    stream: bool = Field(False, description="Whether to stream back response tokens via SSE")


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

    messages_raw = [{"role": m.role, "content": m.content} for m in request.messages]

    if not request.stream:
        try:
            response = await provider.chat(
                messages=messages_raw,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                top_k=request.top_k,
                stop=request.stop,
            )
            return response
        except ConnectionError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # Streaming mode via Server-Sent Events (SSE)
    async def event_generator():
        try:
            async for token in provider.stream(
                messages=messages_raw,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                top_k=request.top_k,
                stop=request.stop,
            ):
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
            yield "data: [DONE]\n\n"
        except Exception as e:
            err_chunk = {"error": str(e)}
            yield f"data: {json.dumps(err_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
