"""
Libra API v1 - Models & Registry Endpoint
"""

import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from apps.backend.core.config import settings
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
    if isinstance(tier, str) and tier:
        try:
            hw_tier = HardwareTier(tier)
        except ValueError:
            pass

    filter_provider = provider if isinstance(provider, str) else None
    filter_cpu_friendly = cpu_friendly_only if isinstance(cpu_friendly_only, bool) else False

    models = registry.list_models(
        provider=filter_provider, tier=hw_tier, cpu_friendly_only=filter_cpu_friendly
    )
    def_model_id = get_system_default_model()
    return {
        "count": len(models),
        "default_model": def_model_id,
        "models": [m.to_dict() for m in models],
    }


def get_system_default_model() -> str:
    """Resolve the highest quality available model based on active configuration."""
    # 1. Prefer Gemini 2.5 Flash if GEMINI_API_KEY is configured
    if settings.gemini_api_key or os.getenv("GEMINI_API_KEY"):
        return "gemini-2.5-flash"

    # 2. Check registered local Ollama models
    for mid in ["llama3.2:1b", "llama3.2:3b", "qwen2.5:0.5b", "gemma2:2b"]:
        if mid in registry._models:
            return mid

    # 3. Check OpenAI if configured
    if settings.openai_api_key or os.getenv("OPENAI_API_KEY"):
        return "gpt-4o-mini"

    # 4. Fallback to mock
    return "libra-mock-v1"


@router.get(
    "/models/default", summary="Get the recommended default model based on active configuration"
)
async def get_default_model() -> dict[str, Any]:
    def_id = get_system_default_model()
    model = registry.get(def_id)
    return {
        "default_model": def_id,
        "model": model.to_dict() if model else None,
    }


@router.get("/models/{model_id:path}", summary="Get detailed metadata for a specific model")
async def get_model(model_id: str) -> dict[str, Any]:
    model = registry.get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in registry")
    return model.to_dict()
