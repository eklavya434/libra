"""
Libra Backend - Capstone System Status Endpoint (Phase 36)
Provides aggregated release readiness and system status for the complete 36-phase stack.
"""

from __future__ import annotations

import os

from fastapi import APIRouter
from pydantic import BaseModel

from packages.core.hardware import detect_hardware

router = APIRouter(prefix="/capstone", tags=["Capstone & Release"])


class CapstoneStatusResponse(BaseModel):
    app_name: str
    version: str
    status: str
    total_phases: int
    completed_phases: int
    curriculum_complete: bool
    pillars: list[str]
    hardware_tier: str
    cpu_model: str
    disk_free_gb: float
    disk_quota_used_mb: float
    disk_quota_max_mb: float
    cost_policy: str
    learning_mode: bool


@router.get("/status", response_model=CapstoneStatusResponse)
async def get_capstone_status() -> CapstoneStatusResponse:
    """Returns the comprehensive Project Libra Capstone Release status."""
    hw = detect_hardware()

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
    )
    tracked = ["models", "data", "checkpoints", ".venv"]
    total_bytes = 0
    for d in tracked:
        dp = os.path.join(repo_root, d)
        if os.path.exists(dp):
            for root, _, files in os.walk(dp):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        total_bytes += os.path.getsize(fp)
                    except OSError:
                        pass
    disk_used_mb = total_bytes / (1024 * 1024)

    pillars = [
        "1. Architecture & Modeling (Modern Transformer, RoPE, GQA, PagedAttention, VLM)",
        "2. Inference Acceleration (KV Cache, Speculative Decoding, Structured Grammars)",
        "3. Provider Routing & Cost Tracking (Multi-Provider, Semantic Router, Cost Tracker)",
        "4. Memory & Context (SQLite Persistent Store, Sliding Window Context Manager)",
        "5. Knowledge & Hybrid RAG (Dense Embeddings, BM25 Lexical, RRF Fusion)",
        "6. Agents & Tools (Python Sandbox, Tool Registry, ReAct & Plan-and-Solve)",
        "7. Training, Alignment & Deployment (ChatML Packing, DPO Loss, Docker Compose, CI Matrix)",
    ]

    return CapstoneStatusResponse(
        app_name="Project Libra - Educational LLM Lab & ChatGPT-Like Assistant",
        version="1.0.0",
        status="operational",
        total_phases=36,
        completed_phases=36,
        curriculum_complete=True,
        pillars=pillars,
        hardware_tier=hw.device_tier,
        cpu_model=hw.cpu_model,
        disk_free_gb=hw.disk_free_gb,
        disk_quota_used_mb=round(disk_used_mb, 2),
        disk_quota_max_mb=15360.0,
        cost_policy="$0 / ₹0 development (Zero-Cost Policy strictly enforced)",
        learning_mode=True,
    )
