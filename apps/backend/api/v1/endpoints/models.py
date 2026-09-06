"""
Libra API v1 - Models & Registry Endpoint
"""

from typing import Any

from fastapi import APIRouter

from packages.providers.mock import MockProvider

router = APIRouter()
default_provider = MockProvider()


@router.get("/models", summary="List all available models across providers")
async def list_models() -> dict[str, Any]:
    models = await default_provider.list_models()
    return {
        "count": len(models),
        "models": [m.to_dict() for m in models],
    }
