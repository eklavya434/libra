"""
Libra Backend - Distributed Tracing Middleware

Intercepts incoming HTTP requests, establishes OpenTelemetry span context,
propagates W3C traceparent headers, and attaches X-Trace-Id and Server-Timing response headers.
"""

from __future__ import annotations

from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from packages.core.observability.metrics import get_metrics_aggregator
from packages.core.observability.tracer import (
    W3CTraceContext,
    generate_trace_id,
    get_tracer,
)


class TracingMiddleware(BaseHTTPMiddleware):
    """Establishes a root distributed tracing span for every incoming HTTP request."""

    def __init__(self, app) -> None:
        super().__init__(app)
        self.tracer = get_tracer()
        self.metrics = get_metrics_aggregator()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Don't trace internal trace retrieval endpoints to prevent query pollution
        if request.url.path.startswith("/api/v1/observability/traces"):
            return await call_next(request)

        traceparent = request.headers.get("traceparent")
        parsed = W3CTraceContext.parse(traceparent)

        if parsed:
            trace_id, parent_span_id, _ = parsed
        else:
            trace_id = generate_trace_id()
            parent_span_id = None

        route_name = f"{request.method} {request.url.path}"
        span = self.tracer.start_span(
            name=route_name,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            attributes={
                "http.method": request.method,
                "http.url": str(request.url),
                "http.path": request.url.path,
                "http.client_ip": request.client.host if request.client else "127.0.0.1",
            },
        )

        request.state.trace_id = span.trace_id
        request.state.span_id = span.span_id

        try:
            with self.tracer.start_as_current_span(
                name=span.name,
                trace_id=span.trace_id,
                parent_span_id=span.parent_span_id,
                attributes=span.attributes,
            ) as active_span:
                response = await call_next(request)
                active_span.set_attribute("http.status_code", response.status_code)
                if response.status_code >= 400:
                    active_span.set_status("ERROR", f"HTTP {response.status_code}")

                # Attach X-Trace-Id and Server-Timing headers
                response.headers["X-Trace-Id"] = active_span.trace_id
                response.headers["Server-Timing"] = f"total;dur={active_span.duration_ms:.2f}"

                # Update global latency rolling metrics
                self.metrics.record_request(
                    duration_ms=active_span.duration_ms,
                    is_error=response.status_code >= 500,
                )
                return response
        except Exception as exc:
            span.set_status("ERROR", str(exc))
            self.metrics.record_request(duration_ms=span.duration_ms, is_error=True)
            raise
