"""
Libra API v1 - Models & Registry Endpoint
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from packages.models.catalog import get_default_registry
from packages.models.registry import HardwareTier

router = APIRouter()
registry = get_default_registry()


@router.get("/models", summary="List all available models across providers")
async def list_models(
    provider: str | None = Query(
        None, description="Filter by provider (e.g. ollama, libra_lab, mock-provider)"
    ),
    tier: str | None = Query(
        None, description="Filter by hardware tier (CPU-friendly, large, etc.)"
    ),
    cpu_friendly_only: bool = Query(
        False, description="Return only models verified to run on your CPU"
    ),
) -> dict[str, Any]:
    # Discover models from local Ollama runtime if reachable
    try:
        from packages.models.registry import LicenseType, ModelMetadata
        from packages.providers.router import get_router

        ollama_prov = get_router().get_provider("ollama")
        installed_ollama = await ollama_prov.list_models()
        for om in installed_ollama:
            if om.id not in registry._models:
                registry.register(
                    ModelMetadata(
                        model_id=om.id,
                        name=f"Ollama {om.name}",
                        organization="Local Ollama",
                        provider="ollama",
                        architecture=om.architecture or "Transformer",
                        parameter_count=4_000_000_000,
                        context_length=om.context_length or 4096,
                        license="Open Weights",
                        license_type=LicenseType.OPEN_WEIGHTS_COMMUNITY,
                        quantization=om.quantization or "Q4_K_M",
                        capabilities=["chat", "stream", "reasoning"],
                        description=f"Locally installed model '{om.id}' ready for CPU inference.",
                    )
                )
    except (ConnectionError, RuntimeError, OSError, ValueError, KeyError):
        pass

    hw_tier = None
    if tier:
        try:
            hw_tier = HardwareTier(tier)
        except ValueError:
            pass

    models = registry.list_models(
        provider=provider, tier=hw_tier, cpu_friendly_only=cpu_friendly_only
    )
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
