"""
Libra API v1 - Observability, Distributed Tracing & Telemetry Endpoints

Provides REST APIs for:
1. GET /api/v1/observability/traces: List recent distributed traces.
2. GET /api/v1/observability/traces/{trace_id}: Detailed span hierarchy with waterfall offsets.
3. GET /api/v1/observability/metrics: System-wide latency percentiles and token velocities.
4. POST /api/v1/observability/traces/clear: Reset trace collector buffer.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from packages.core.observability.metrics import get_metrics_aggregator
from packages.core.observability.tracer import get_trace_collector

router = APIRouter(prefix="/observability", tags=["Observability & Tracing"])


@router.get("/traces", summary="List recent distributed traces")
async def list_traces(
    limit: int = Query(50, ge=1, le=200, description="Max traces to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> dict[str, Any]:
    collector = get_trace_collector()
    traces = collector.list_traces(limit=limit, offset=offset)
    return {
        "total": len(traces),
        "limit": limit,
        "offset": offset,
        "traces": traces,
    }


@router.get("/traces/{trace_id}", summary="Get detailed trace span hierarchy and latency waterfall")
async def get_trace_detail(trace_id: str) -> dict[str, Any]:
    collector = get_trace_collector()
    trace = collector.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found")
    return trace


@router.get("/metrics", summary="Get system-wide latency percentiles and token velocity metrics")
async def get_metrics() -> dict[str, Any]:
    aggregator = get_metrics_aggregator()
    collector = get_trace_collector()
    summary = aggregator.get_summary()
    summary["buffered_traces_count"] = len(collector.list_traces(limit=200))
    return summary


@router.post("/traces/clear", summary="Clear trace buffer and reset telemetry counters")
async def clear_traces() -> dict[str, Any]:
    collector = get_trace_collector()
    collector.clear()
    aggregator = get_metrics_aggregator()
    aggregator.reset()
    return {"status": "cleared", "message": "Trace collector and metrics reset successfully."}
