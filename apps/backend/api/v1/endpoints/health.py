"""
Libra API v1 - Health & Diagnostics Endpoint
"""

import os
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter

from apps.backend.core.config import settings
from packages.core.hardware import detect_hardware

router = APIRouter()


@router.get("/health", summary="Get system health and hardware diagnostics")
async def get_health() -> dict[str, Any]:
    hw_report = detect_hardware()
    commit = os.getenv("RENDER_GIT_COMMIT", "local")
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
        "active_providers": ["gemini"] if settings.gemini_api_key else ["mock-provider"],
    }
