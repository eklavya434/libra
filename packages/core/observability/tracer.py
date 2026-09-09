"""
Libra Core Observability - Distributed Tracing & OpenTelemetry Core

Provides first-principles, zero-cost distributed tracing:
1. 128-bit TraceId & 64-bit SpanId generation compliant with W3C TraceContext.
2. Parent-child span hierarchy tracking with thread-safe contextvars.
3. OpenTelemetry span semantics (attributes, events, status codes, duration).
4. In-memory bounded TraceCollector ring buffer for zero-dependency local storage.
"""

from __future__ import annotations

import contextvars
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Generator, Optional


def generate_trace_id() -> str:
    """Generates a random 128-bit hex string for W3C TraceId."""
    return os.urandom(16).hex()


def generate_span_id() -> str:
    """Generates a random 64-bit hex string for W3C SpanId."""
    return os.urandom(8).hex()


class W3CTraceContext:
    """Parses and serializes W3C traceparent headers (00-{trace_id}-{span_id}-{flags})."""

    _PATTERN = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")

    @classmethod
    def parse(cls, header: Optional[str]) -> Optional[tuple[str, str, str]]:
        if not header:
            return None
        match = cls._PATTERN.match(header.strip().lower())
        if not match:
            return None
        return match.group(1), match.group(2), match.group(3)

    @classmethod
    def serialize(cls, trace_id: str, span_id: str, sampled: bool = True) -> str:
        flags = "01" if sampled else "00"
        return f"00-{trace_id}-{span_id}-{flags}"


@dataclass
class SpanEvent:
    name: str
    timestamp_ns: int
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timestamp_ns": self.timestamp_ns,
            "attributes": self.attributes,
        }


@dataclass
class Span:
    """Represents a single timed unit of work within a distributed trace."""

    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    start_time_ns: int = field(default_factory=time.perf_counter_ns)
    end_time_ns: Optional[int] = None
    status: str = "OK"  # "OK" or "ERROR"
    status_description: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.end_time_ns is None:
            now = time.perf_counter_ns()
            return round((now - self.start_time_ns) / 1_000_000, 3)
        return round((self.end_time_ns - self.start_time_ns) / 1_000_000, 3)

    def set_attribute(self, key: str, value: Any) -> Span:
        self.attributes[key] = value
        return self

    def set_attributes(self, attrs: dict[str, Any]) -> Span:
        self.attributes.update(attrs)
        return self

    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        self.events.append(
            SpanEvent(
                name=name,
                timestamp_ns=time.perf_counter_ns(),
                attributes=attributes or {},
            )
        )

    def set_status(self, status: str, description: str = "") -> None:
        self.status = status.upper()
        self.status_description = description

    def end(self, end_time_ns: Optional[int] = None) -> None:
        if self.end_time_ns is None:
            self.end_time_ns = end_time_ns or time.perf_counter_ns()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "status_description": self.status_description,
            "attributes": self.attributes,
            "events": [e.to_dict() for e in self.events],
        }


# Context variable for propagation across async execution flows
_current_span_var: contextvars.ContextVar[Optional[Span]] = contextvars.ContextVar(
    "current_span", default=None
)


class TraceCollector:
    """Thread-safe, bounded ring-buffer storage for completed traces."""

    def __init__(self, capacity: int = 100) -> None:
        self.capacity = capacity
        # Maps trace_id -> list of completed spans in that trace
        self._traces: dict[str, list[Span]] = {}
        # Order of trace_ids for ring-buffer eviction
        self._trace_order: list[str] = []
        self._lock = Lock()

    def record_span(self, span: Span) -> None:
        with self._lock:
            tid = span.trace_id
            if tid not in self._traces:
                if len(self._trace_order) >= self.capacity:
                    oldest_tid = self._trace_order.pop(0)
                    self._traces.pop(oldest_tid, None)
                self._traces[tid] = []
                self._trace_order.append(tid)
            self._traces[tid].append(span)

    def get_trace(self, trace_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            spans = self._traces.get(trace_id)
            if not spans:
                return None

            # Sort spans by start_time_ns
            sorted_spans = sorted(spans, key=lambda s: s.start_time_ns)
            root_span = next((s for s in sorted_spans if s.parent_span_id is None), sorted_spans[0])
            total_duration_ms = root_span.duration_ms
            root_start_ns = root_span.start_time_ns

            span_dicts = []
            for s in sorted_spans:
                d = s.to_dict()
                # Calculate relative offset from root start for visual Gantt waterfall
                d["offset_ms"] = round(max(0, s.start_time_ns - root_start_ns) / 1_000_000, 3)
                span_dicts.append(d)

            return {
                "trace_id": trace_id,
                "root_name": root_span.name,
                "status": "ERROR" if any(s.status == "ERROR" for s in spans) else "OK",
                "span_count": len(spans),
                "total_duration_ms": total_duration_ms,
                "spans": span_dicts,
            }

    def list_traces(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            summaries = []
            # Traverse in reverse order (most recent first)
            for tid in reversed(self._trace_order[offset : offset + limit]):
                spans = self._traces.get(tid, [])
                if not spans:
                    continue
                sorted_spans = sorted(spans, key=lambda s: s.start_time_ns)
                root_span = next(
                    (s for s in sorted_spans if s.parent_span_id is None), sorted_spans[0]
                )
                summaries.append(
                    {
                        "trace_id": tid,
                        "root_name": root_span.name,
                        "status": "ERROR" if any(s.status == "ERROR" for s in spans) else "OK",
                        "span_count": len(spans),
                        "total_duration_ms": root_span.duration_ms,
                        "start_time_ns": root_span.start_time_ns,
                    }
                )
            return summaries

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()
            self._trace_order.clear()


# Global singletons
_global_trace_collector = TraceCollector(capacity=100)


def get_trace_collector() -> TraceCollector:
    return _global_trace_collector


class Tracer:
    """Manages active span creation, hierarchy, and context scoping."""

    def __init__(self, collector: Optional[TraceCollector] = None) -> None:
        self.collector = collector or _global_trace_collector

    def get_current_span(self) -> Optional[Span]:
        return _current_span_var.get()

    def start_span(
        self,
        name: str,
        parent_span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> Span:
        parent = self.get_current_span()
        active_trace_id = trace_id or (parent.trace_id if parent else generate_trace_id())
        active_parent_id = parent_span_id or (parent.span_id if parent else None)
        span_id = generate_span_id()

        return Span(
            name=name,
            trace_id=active_trace_id,
            span_id=span_id,
            parent_span_id=active_parent_id,
            attributes=attributes or {},
        )

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        parent_span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        span = self.start_span(
            name=name,
            parent_span_id=parent_span_id,
            trace_id=trace_id,
            attributes=attributes,
        )
        token = _current_span_var.set(span)
        try:
            yield span
        except Exception as exc:
            span.set_status("ERROR", str(exc))
            span.add_event(
                "exception", {"exception.message": str(exc), "exception.type": type(exc).__name__}
            )
            raise
        finally:
            span.end()
            self.collector.record_span(span)
            _current_span_var.reset(token)


_global_tracer = Tracer()


def get_tracer() -> Tracer:
    return _global_tracer
