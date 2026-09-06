"""
Libra API v1 - Models & Registry Endpoint
"""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from packages.models.catalog import get_default_registry
from packages.models.registry import HardwareTier

router = APIRouter()
registry = get_default_registry()


@router.get("/models", summary="List all available models across providers")
async def list_models(
    provider: Optional[str] = Query(None, description="Filter by provider (e.g. ollama, libra_lab, mock-provider)"),
    tier: Optional[str] = Query(None, description="Filter by hardware tier (CPU-friendly, large, etc.)"),
    cpu_friendly_only: bool = Query(False, description="Return only models verified to run on your CPU"),
) -> dict[str, Any]:
    hw_tier = None
    if tier:
        try:
            hw_tier = HardwareTier(tier)
        except ValueError:
            pass

    models = registry.list_models(provider=provider, tier=hw_tier, cpu_friendly_only=cpu_friendly_only)
    return {
        "count": len(models),
        "models": [m.to_dict() for m in models],
    }


@router.get("/models/{model_id:path}", summary="Get detailed metadata for a specific model")
async def get_model(model_id: str) -> dict[str, Any]:
    model = registry.get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in registry")
    return model.to_dict()
