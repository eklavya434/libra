"""
Libra API v1 - High-Throughput Batch Inference & Async Workers
Provides non-blocking job creation, SSE progress streaming, dynamic sequence binning analysis,
and job management.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from packages.core.batch.sequence_binner import (
    BinningStrategy,
    calculate_padding_waste,
)
from packages.core.batch.worker_queue import (
    BatchJob,
    BatchJobPriority,
    BatchJobStatus,
    get_batch_scheduler,
)
from packages.models.inference.batch_generator import BatchGenerator

router = APIRouter(prefix="/batch", tags=["Batch Inference & Async Workers"])

# Initialize scheduler and batch generator
scheduler = get_batch_scheduler(max_concurrency=1)
generator = BatchGenerator()


async def default_job_processor(job: BatchJob, progress_cb: Any) -> None:
    await generator.execute_job(job, progress_cb)


scheduler.register_processor(default_job_processor)


class BatchJobCreateRequest(BaseModel):
    prompts: list[str] = Field(
        ..., min_length=1, max_length=200, description="List of prompts to process"
    )
    name: str = Field("Batch Workload", description="Human-readable job label")
    model_name: str = Field("libra-llama-tied", description="Target model name")
    max_new_tokens: int = Field(32, ge=1, le=256, description="Max tokens generated per prompt")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    binning_strategy: str = Field(
        "dynamic_length", description="Binning strategy: naive, dynamic_length, fixed_buckets"
    )
    priority: str = Field("NORMAL", description="Priority level: HIGH, NORMAL, LOW")


class BatchAnalyzeRequest(BaseModel):
    prompts: Optional[list[str]] = Field(
        None, description="List of prompts to analyze for length waste"
    )
    sequence_lengths: Optional[list[int]] = Field(None, description="Direct list of token lengths")
    max_batch_size: int = Field(16, ge=1, le=64)
    max_tokens_per_bucket: int = Field(1024, ge=64, le=8192)


@router.post("/jobs")
async def create_batch_job(req: BatchJobCreateRequest) -> dict[str, Any]:
    """Submits a new batch job to the asynchronous worker queue."""
    if not scheduler._running:
        await scheduler.start()

    strategy_map = {
        "naive": BinningStrategy.NAIVE,
        "dynamic_length": BinningStrategy.DYNAMIC_LENGTH,
        "fixed_buckets": BinningStrategy.FIXED_BUCKETS,
    }
    strategy = strategy_map.get(req.binning_strategy.lower(), BinningStrategy.DYNAMIC_LENGTH)

    priority_map = {
        "HIGH": BatchJobPriority.HIGH,
        "NORMAL": BatchJobPriority.NORMAL,
        "LOW": BatchJobPriority.LOW,
    }
    prio = priority_map.get(req.priority.upper(), BatchJobPriority.NORMAL)

    job = await scheduler.submit_job(
        prompts=req.prompts,
        name=req.name,
        model_name=req.model_name,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        binning_strategy=strategy,
        priority=prio,
    )

    return {
        "status": "submitted",
        "job": job.to_dict(),
    }


@router.get("/jobs")
async def list_batch_jobs() -> dict[str, Any]:
    """Lists all batch jobs and worker queue metrics."""
    jobs = scheduler.list_jobs()
    active_count = sum(
        1 for j in jobs if j.status in [BatchJobStatus.QUEUED, BatchJobStatus.PROCESSING]
    )
    completed_count = sum(1 for j in jobs if j.status == BatchJobStatus.COMPLETED)
    failed_count = sum(1 for j in jobs if j.status == BatchJobStatus.FAILED)

    return {
        "total_jobs": len(jobs),
        "active_jobs": active_count,
        "completed_jobs": completed_count,
        "failed_jobs": failed_count,
        "worker_concurrency": scheduler.max_concurrency,
        "jobs": [j.to_dict() for j in jobs],
    }


@router.get("/jobs/{job_id}")
async def get_batch_job(job_id: str) -> dict[str, Any]:
    """Fetches full state, results, and telemetry for a specific batch job."""
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Batch job '{job_id}' not found.")
    return job.to_dict()


@router.delete("/jobs/{job_id}")
async def cancel_batch_job(job_id: str) -> dict[str, Any]:
    """Cancels a pending or currently processing batch job."""
    success = await scheduler.cancel_job(job_id)
    if not success:
        job = scheduler.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Batch job '{job_id}' not found.")
        raise HTTPException(
            status_code=400,
            detail=f"Job '{job_id}' in state '{job.status.value}' cannot be cancelled.",
        )
    return {"status": "cancelled", "job_id": job_id}


@router.get("/jobs/{job_id}/stream")
async def stream_batch_job_progress(job_id: str) -> StreamingResponse:
    """Streams real-time Server-Sent Events (SSE) detailing job progress and completion."""
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Batch job '{job_id}' not found.")

    async def event_generator():
        q = scheduler.subscribe(job_id)
        try:
            # Send initial state
            yield f"data: {json.dumps({'event': 'initial', 'job': job.to_dict()})}\n\n"

            if job.status in [
                BatchJobStatus.COMPLETED,
                BatchJobStatus.FAILED,
                BatchJobStatus.CANCELLED,
            ]:
                return

            while True:
                try:
                    data = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {json.dumps(data)}\n\n"
                    if data.get("event") in ["finished", "cancelled"]:
                        break
                except asyncio.TimeoutError:
                    # Keepalive ping
                    yield ": ping\n\n"
        finally:
            scheduler.unsubscribe(job_id, q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/analyze")
async def analyze_batch_efficiency(req: BatchAnalyzeRequest) -> dict[str, Any]:
    """
    Computes comparative FLOP and padding waste between naive uniform batching
    and dynamic sequence length binning.
    """
    lengths = req.sequence_lengths or []
    if req.prompts:
        # Approximate tokens by whitespace/character length heuristic if lengths not given
        lengths = [max(1, len(p.split())) for p in req.prompts]

    if not lengths:
        raise HTTPException(
            status_code=400, detail="Provide either prompts or sequence_lengths to analyze."
        )

    analysis = calculate_padding_waste(
        sequence_lengths=lengths,
        max_batch_size=req.max_batch_size,
        max_tokens_per_bucket=req.max_tokens_per_bucket,
    )
    return analysis
