"""
Libra Backend - Main Application Entrypoint
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.backend.api.v1.router import api_v1_router
from apps.backend.core.config import settings
from apps.backend.core.logging import logger
from apps.backend.middleware.identity import SessionIdentityMiddleware
from apps.backend.middleware.observability import RequestIdMiddleware, exception_handlers
from apps.backend.middleware.security import (
    RateLimitingMiddleware,
    RequestSizeLimiterMiddleware,
    SecurityHeadersMiddleware,
)
from apps.backend.middleware.tracing import TracingMiddleware
from packages.core.hardware import detect_hardware
from packages.core.memory import get_conversation_store
from packages.core.network import enable_ipv4_preference

enable_ipv4_preference()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Initializing Libra Backend...")
    hw = detect_hardware()
    logger.info(
        f"Hardware detected: CPU={hw.cpu_model} ({hw.cpu_physical_cores}C/{hw.cpu_logical_cores}T), "
        f"RAM={hw.ram_available_gb}/{hw.ram_total_gb}GB, "
        f"DiskFree={hw.disk_free_gb}GB, CUDA={hw.has_cuda} (Tier: {hw.device_tier})"
    )
    if not hw.has_cuda:
        logger.info(
            "Running in CPU-first mode. All computations optimized for local CPU execution."
        )
    # Housekeeping for guest sessions (30-day default TTL).
    expired = get_conversation_store().cleanup_expired_sessions()
    logger.info(f"Expired sessions cleaned: {expired}")
    logger.info(f"Public code execution enabled: {settings.libra_public_code_exec_enabled}")
    yield
    logger.info("Shutting down Libra Backend...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend API powering the Libra Conversational Assistant and LLM Laboratory.",
    lifespan=lifespan,
)

# Register error handlers explicitly so the server-level middleware honors them
# (constructor-passed handlers are not applied to the ServerErrorMiddleware).
for _exc, _handler in exception_handlers().items():
    app.add_exception_handler(_exc, _handler)

# Configure CORS for local frontend communication. In production the frontend
# and backend share one origin, so this list is typically empty.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestSizeLimiterMiddleware, max_bytes=10 * 1024 * 1024)
app.add_middleware(
    RateLimitingMiddleware,
    requests_per_minute=settings.libra_rate_limit_per_minute,
    burst_capacity=settings.libra_rate_limit_burst,
)
app.add_middleware(TracingMiddleware)
app.add_middleware(
    SessionIdentityMiddleware,
    ttl_days=settings.libra_session_ttl_days,
)
# Outermost: attaches request IDs used by logging, tracing, and error handlers.
app.add_middleware(RequestIdMiddleware)

# Mount API routes
app.include_router(api_v1_router)


@app.get("/", summary="Root status")
async def root():
    return {
        "message": "Welcome to Libra AI Laboratory & Assistant API",
        "docs_url": "/docs",
        "health_url": "/api/v1/health",
        "learning_mode": settings.libra_learning_mode,
        "public_code_exec_enabled": settings.libra_public_code_exec_enabled,
    }


@app.get("/healthz", summary="Liveness probe")
async def healthz() -> dict:
    """Liveness: the process is up and serving requests."""
    return {"status": "alive"}


@app.get("/readyz", summary="Readiness probe")
async def readyz() -> JSONResponse:
    """Readiness: core dependencies (SQLite store) are reachable."""
    try:
        store = get_conversation_store()
        store.list_conversations(limit=1, offset=0)
        return JSONResponse({"status": "ready"})
    except Exception as exc:  # noqa: BLE001 - readiness must not raise
        logger.exception("Readiness probe failed")
        return JSONResponse(content={"status": "not_ready", "detail": str(exc)}, status_code=503)


@app.get("/livez", summary="Liveness probe (alias)")
async def livez() -> dict:
    """Alias for /healthz used by some orchestrators."""
    return {"status": "alive"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "apps.backend.main:app",
        host=settings.libra_host,
        port=settings.libra_port,
        reload=True,
    )
