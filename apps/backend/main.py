"""
Libra Backend - Main Application Entrypoint
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.backend.api.v1.router import api_v1_router
from apps.backend.core.config import settings
from apps.backend.core.logging import logger
from apps.backend.middleware.security import (
    RateLimitingMiddleware,
    RequestSizeLimiterMiddleware,
    SecurityHeadersMiddleware,
)
from apps.backend.middleware.tracing import TracingMiddleware
from packages.core.hardware import detect_hardware
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
    yield
    logger.info("Shutting down Libra Backend...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend API powering the Libra Conversational Assistant and LLM Laboratory.",
    lifespan=lifespan,
)

# Configure CORS for local frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestSizeLimiterMiddleware, max_bytes=10 * 1024 * 1024)
app.add_middleware(RateLimitingMiddleware)
app.add_middleware(TracingMiddleware)

# Mount API routes
app.include_router(api_v1_router)


@app.get("/", summary="Root status")
async def root():
    return {
        "message": "Welcome to Libra AI Laboratory & Assistant API",
        "docs_url": "/docs",
        "health_url": "/api/v1/health",
        "learning_mode": settings.libra_learning_mode,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "apps.backend.main:app",
        host=settings.libra_host,
        port=settings.libra_port,
        reload=True,
    )
