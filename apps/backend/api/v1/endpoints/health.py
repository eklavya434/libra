"""
Libra API v1 - Health & Diagnostics Endpoint
"""

import os
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter

from apps.backend.core.config import settings
from packages.core.hardware import detect_hardware
from packages.core.memory import get_conversation_store
from packages.providers.router import get_router

router = APIRouter()


async def _active_providers() -> list[str]:
    """Honestly report which providers can actually serve requests right now.

    Cloud providers count only when their API key is configured; the local
    Ollama runtime only when the daemon answers; the lab model only when a real
    trained checkpoint exists. MockProvider is a test artifact and is never
    reported as an active production provider.
    """
    router_instance = get_router()
    active: list[str] = []

    cloud_ids = {
        "openai",
        "gemini",
        "anthropic",
        "deepseek",
        "groq",
        "kimi",
        "openrouter",
        "nvidia",
    }
    for pid in cloud_ids:
        prov = router_instance._providers.get(pid)
        if prov and getattr(prov, "api_key", None):
            active.append(pid)

    try:
        ollama_prov = router_instance._providers.get("ollama")
        if ollama_prov:
            health_resp = await ollama_prov.health()
            if health_resp.get("connected"):
                active.append("ollama")
    except Exception:
        pass

    libra_prov = router_instance._providers.get("libra_lab")
    if libra_prov and getattr(libra_prov, "is_ready", lambda: False)():
        active.append("libra_lab")

    if router_instance._providers.get("huggingface") is not None:
        active.append("huggingface")

    return sorted(active)


@router.get("/health", summary="Get system health and hardware diagnostics")
async def get_health() -> dict[str, Any]:
    hw_report = detect_hardware()
    commit = os.getenv("RENDER_GIT_COMMIT", "local")
    store = get_conversation_store()
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "commit": commit,
        "environment": settings.libra_env,
        "learning_mode": settings.libra_learning_mode,
        "gemini_configured": bool(settings.gemini_api_key),
        "secret_file_exists": os.path.exists("/etc/secrets/.env"),
        "timestamp": datetime.now(UTC).isoformat(),
        "hardware": hw_report.to_dict(),
        "store_backend": type(store).__name__,
        "active_providers": await _active_providers(),
    }
