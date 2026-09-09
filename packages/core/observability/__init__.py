"""
Libra Core Observability Package

Provides OpenTelemetry-compatible distributed tracing and token metrics:
- Tracer, Span, TraceCollector
- W3CTraceContext
- TokenVelocityMetrics, MetricsAggregator
"""

from packages.core.observability.metrics import (
    MetricsAggregator,
    TokenVelocityMetrics,
    get_metrics_aggregator,
)
from packages.core.observability.tracer import (
    Span,
    SpanEvent,
    TraceCollector,
    Tracer,
    W3CTraceContext,
    generate_span_id,
    generate_trace_id,
    get_trace_collector,
    get_tracer,
)

__all__ = [
    "MetricsAggregator",
    "Span",
    "SpanEvent",
    "TokenVelocityMetrics",
    "TraceCollector",
    "Tracer",
    "W3CTraceContext",
    "generate_span_id",
    "generate_trace_id",
    "get_metrics_aggregator",
    "get_trace_collector",
    "get_tracer",
]
