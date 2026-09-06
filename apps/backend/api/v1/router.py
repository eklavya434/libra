"""
Libra API v1 - Route Aggregator
"""

from fastapi import APIRouter

from apps.backend.api.v1.endpoints import arena, chat, health, models

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(health.router, tags=["Health"])
api_v1_router.include_router(models.router, tags=["Models"])
api_v1_router.include_router(chat.router, tags=["Chat"])
api_v1_router.include_router(arena.router, tags=["Arena"])

