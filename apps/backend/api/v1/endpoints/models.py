"""
Libra API v1 - Models & Registry Endpoint
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from packages.models.catalog import get_default_registry
from packages.models.registry import HardwareTier
from packages.providers.router import get_router

router = APIRouter()
registry = get_default_registry()
provider_router = get_router()


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
    installed_ollama_ids: set[str] = set()
    try:
        from packages.models.registry import LicenseType, ModelMetadata

        ollama_prov = provider_router.get_provider("ollama")
        installed_ollama = await ollama_prov.list_models()
        installed_ollama_ids = {om.id for om in installed_ollama}
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

    annotated: list[dict[str, Any]] = []
    for m in models:
        d = m.to_dict()
        try:
            available, reason = await provider_router.check_model_available(
                m.model_id, installed_ollama_ids
            )
        except Exception:
            available, reason = False, "Availability could not be verified right now."
        d["available"] = available
        d["unavailable_reason"] = reason
        annotated.append(d)

    return {
        "count": len(models),
        "default_model": def_model_id,
        "models": annotated,
    }


def get_system_default_model() -> str | None:
    """Resolve the highest quality CONFIGURED real model id, or None.

    Public production must never silently fall back to MockProvider or to a
    local/experimental model. Only providers whose API key is actually
    configured (and whose representative model id exists in the registry) are
    candidates. When nothing is configured the caller must surface a clear
    configuration message instead of faking availability.
    """
    has_key = lambda pid: bool(  # noqa: E731
        getattr(provider_router.get_provider(pid), "api_key", None)
    )
    candidates = [
        ("gemini", "gemini-2.5-flash"),
        ("anthropic", "claude-3-5-haiku-20241022"),
        ("nvidia", "nvidia/llama-3.1-nemotron-70b-instruct"),
        ("openai", "gpt-4o-mini"),
        ("deepseek", "deepseek-ai/deepseek-v4-flash-0731"),
        ("kimi", "kimi-k1.5"),
    ]
    for provider_id, model_id in candidates:
        if has_key(provider_id) and registry.get(model_id) is not None:
            return model_id
    return None


@router.get(
    "/models/default", summary="Get the recommended default model based on active configuration"
)
async def get_default_model() -> dict[str, Any]:
    def_id = get_system_default_model()
    if def_id is None:
        return {
            "default_model": None,
            "configured": False,
            "message": (
                "No provider is configured. Add an API key (e.g. GEMINI_API_KEY, ANTHROPIC_API_KEY) "
                "or start a local Ollama engine, then reload. Chat will refuse to answer until a "
                "real provider is configured — MockProvider is never used in production."
            ),
            "model": None,
        }
    model = registry.get(def_id)
    return {
        "default_model": def_id,
        "configured": True,
        "model": model.to_dict() if model else None,
    }


@router.get("/models/{model_id:path}", summary="Get detailed metadata for a specific model")
async def get_model(model_id: str) -> dict[str, Any]:
    model = registry.get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in registry")
    return model.to_dict()
