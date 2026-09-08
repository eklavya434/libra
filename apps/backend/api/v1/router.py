"""
Libra API v1 - Route Aggregator
"""

from fastapi import APIRouter

from apps.backend.api.v1.endpoints import (
    agents,
    alignment,
    arena,
    attention,
    chat,
    coder,
    conversations,
    grammar,
    health,
    models,
    peft,
    quantization,
    rag,
    routing,
    structured,
    teams,
    telemetry,
    tools,
)

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(health.router, tags=["Health"])
api_v1_router.include_router(models.router, tags=["Models"])
api_v1_router.include_router(chat.router, tags=["Chat"])
api_v1_router.include_router(arena.router, tags=["Arena"])
api_v1_router.include_router(conversations.router, prefix="/conversations", tags=["Conversations"])
api_v1_router.include_router(rag.router, prefix="/rag", tags=["RAG"])
api_v1_router.include_router(tools.router, prefix="/tools", tags=["Tools"])
api_v1_router.include_router(structured.router, prefix="/structured", tags=["Structured"])
api_v1_router.include_router(agents.router, prefix="/agents", tags=["Agents"])
api_v1_router.include_router(coder.router, prefix="/coder", tags=["Coder"])
api_v1_router.include_router(teams.router, prefix="/teams", tags=["Teams"])
api_v1_router.include_router(routing.router, tags=["Dynamic Routing"])
api_v1_router.include_router(alignment.router, tags=["Alignment & RLHF"])
api_v1_router.include_router(attention.router, tags=["Attention & KV Cache"])
api_v1_router.include_router(quantization.router, tags=["Quantization & Compression"])
api_v1_router.include_router(peft.router, tags=["PEFT & LoRA"])
api_v1_router.include_router(telemetry.router, tags=["Token Telemetry"])
api_v1_router.include_router(grammar.router, tags=["Grammar & Constrained Decoding"])
