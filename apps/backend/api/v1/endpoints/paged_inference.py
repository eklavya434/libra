"""
Libra Backend API - PagedAttention & Continuous Batching Endpoints (Phase 31)
Provides simulation, physical page audits, and multi-sequence continuous batching execution.
"""

from __future__ import annotations

import time
from typing import Any
from fastapi import APIRouter
from pydantic import BaseModel, Field

from packages.models.components.paged_cache import PagedKVCache
from packages.models.inference.continuous_batching import ContinuousBatchingEngine

router = APIRouter(prefix="/paged", tags=["PagedAttention & Continuous Batching"])

# Global shared paged cache instance for simulation / inspection
_global_paged_cache = PagedKVCache(
    num_layers=2,
    num_blocks=64,
    block_size=16,
    num_kv_heads=4,
    head_dim=64,
)


class SimulationRequest(BaseModel):
    batch_size: int = Field(default=8, ge=1, le=64)
    avg_prompt_tokens: int = Field(default=120, ge=10, le=4096)
    max_output_tokens: int = Field(default=256, ge=16, le=4096)
    block_size: int = Field(default=16, ge=4, le=128)
    head_dim: int = Field(default=64)
    num_kv_heads: int = Field(default=4)
    num_layers: int = Field(default=2)


class SimulationResponse(BaseModel):
    contiguous_reservation_bytes: int
    contiguous_waste_percent: float
    paged_allocation_bytes: int
    paged_internal_frag_percent: float
    memory_savings_percent: float
    max_concurrency_multiplier: float
    summary: str


class BatchRunRequest(BaseModel):
    prompts: list[str] = Field(
        default=[
            "Explain quantum superposition.",
            "Write a Python function for binary search.",
            "Describe the life cycle of a massive star.",
            "What causes gravitational time dilation?",
        ]
    )
    max_tokens: int = Field(default=12, ge=4, le=64)


class BatchRunResponse(BaseModel):
    total_requests: int
    total_tokens_generated: int
    total_iterations: int
    total_time_ms: float
    throughput_tokens_per_sec: float
    timeline: list[dict[str, Any]]
    completed_requests: list[dict[str, Any]]


@router.post("/simulate", response_model=SimulationResponse)
async def simulate_memory_savings(req: SimulationRequest) -> SimulationResponse:
    """Simulates memory footprint and fragmentation comparing standard contiguous vs PagedAttention."""
    bytes_per_element = 4  # FP32
    kv_state_bytes_per_token = req.num_layers * req.num_kv_heads * req.head_dim * 2 * bytes_per_element

    # In contiguous allocation, every sequence reserves (avg_prompt + max_output) tokens upfront
    max_tokens_per_seq = req.avg_prompt_tokens + req.max_output_tokens
    contiguous_bytes = req.batch_size * max_tokens_per_seq * kv_state_bytes_per_token

    # Real tokens actually used if sequence generates on average 50% of max_output_tokens
    actual_tokens_used = req.batch_size * (req.avg_prompt_tokens + (req.max_output_tokens // 2))
    contiguous_used_bytes = actual_tokens_used * kv_state_bytes_per_token
    contiguous_waste_percent = ((contiguous_bytes - contiguous_used_bytes) / contiguous_bytes) * 100.0

    # In PagedAttention, memory is allocated in blocks of size B dynamically
    # Average internal fragmentation per sequence is at most (block_size / 2) tokens
    tokens_per_seq_paged = (req.avg_prompt_tokens + (req.max_output_tokens // 2))
    blocks_per_seq = (tokens_per_seq_paged + req.block_size - 1) // req.block_size
    allocated_tokens_paged = req.batch_size * blocks_per_seq * req.block_size
    paged_bytes = allocated_tokens_paged * kv_state_bytes_per_token
    internal_frag_tokens = (allocated_tokens_paged - actual_tokens_used)
    paged_frag_percent = (internal_frag_tokens / allocated_tokens_paged) * 100.0

    savings_percent = ((contiguous_bytes - paged_bytes) / contiguous_bytes) * 100.0
    concurrency_mult = contiguous_bytes / max(1, paged_bytes)

    return SimulationResponse(
        contiguous_reservation_bytes=contiguous_bytes,
        contiguous_waste_percent=round(contiguous_waste_percent, 2),
        paged_allocation_bytes=paged_bytes,
        paged_internal_frag_percent=round(paged_frag_percent, 2),
        memory_savings_percent=round(savings_percent, 2),
        max_concurrency_multiplier=round(concurrency_mult, 2),
        summary=(
            f"PagedAttention reduces memory waste from {contiguous_waste_percent:.1f}% down to "
            f"{paged_frag_percent:.1f}%, enabling {concurrency_mult:.2f}x higher request concurrency."
        ),
    )


@router.post("/batch_run", response_model=BatchRunResponse)
async def run_continuous_batch(req: BatchRunRequest) -> BatchRunResponse:
    """Simulates continuous iteration batching interleaving arrivals and completions."""
    cache = PagedKVCache(
        num_layers=2,
        num_blocks=128,
        block_size=16,
        num_kv_heads=4,
        head_dim=64,
    )
    engine = ContinuousBatchingEngine(paged_cache=cache)

    for idx, prompt_text in enumerate(req.prompts):
        tokens = [hash(w) % 1000 for w in prompt_text.split()] or [1, 2, 3]
        engine.add_request(
            request_id=f"req-{idx+1}",
            prompt=prompt_text,
            prompt_token_ids=tokens,
            max_new_tokens=req.max_tokens,
        )

    t0 = time.perf_counter()
    timeline = engine.run_until_complete()
    total_time_ms = (time.perf_counter() - t0) * 1000.0

    total_tokens = sum(r.tokens_generated_in_step for r in timeline)
    throughput = (total_tokens / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0

    return BatchRunResponse(
        total_requests=len(req.prompts),
        total_tokens_generated=total_tokens,
        total_iterations=len(timeline),
        total_time_ms=round(total_time_ms, 2),
        throughput_tokens_per_sec=round(throughput, 1),
        timeline=[
            {
                "iteration": r.iteration_idx,
                "running": r.num_running,
                "waiting": r.num_waiting,
                "finished": r.num_finished,
                "allocated_blocks": r.allocated_blocks,
                "free_blocks": r.free_blocks,
                "elapsed_ms": r.elapsed_ms,
            }
            for r in timeline
        ],
        completed_requests=[
            {
                "request_id": c.request_id,
                "prompt": c.prompt,
                "tokens_generated": len(c.generated_token_ids),
                "latency_ms": round(c.latency_ms, 2),
            }
            for c in engine.completed_requests
        ],
    )


@router.get("/memory_stats")
async def get_paged_memory_stats() -> dict[str, Any]:
    """Returns memory metrics from the global physical block pool."""
    return _global_paged_cache.memory_metrics()
