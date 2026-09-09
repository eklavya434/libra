"""
Tests for Tracer & TraceCollector (packages/core/observability/tracer.py)
"""

import time

from packages.core.observability.tracer import (
    TraceCollector,
    Tracer,
    W3CTraceContext,
    generate_span_id,
    generate_trace_id,
)


def test_id_generation():
    tid = generate_trace_id()
    assert len(tid) == 32
    assert int(tid, 16) >= 0

    sid = generate_span_id()
    assert len(sid) == 16
    assert int(sid, 16) >= 0


def test_w3c_traceparent_parsing_and_serialization():
    tid = "4bf92f3577b34da6a3ce929d0e0e4736"
    sid = "00f067aa0ba902b7"
    header = W3CTraceContext.serialize(tid, sid, sampled=True)
    assert header == f"00-{tid}-{sid}-01"

    parsed = W3CTraceContext.parse(header)
    assert parsed is not None
    assert parsed[0] == tid
    assert parsed[1] == sid
    assert parsed[2] == "01"

    # Invalid header
    assert W3CTraceContext.parse("invalid-header") is None
    assert W3CTraceContext.parse(None) is None


def test_tracer_span_nesting_and_context():
    collector = TraceCollector(capacity=10)
    tracer = Tracer(collector=collector)

    with tracer.start_as_current_span("root_task", attributes={"component": "pipeline"}) as root:
        assert tracer.get_current_span() == root
        assert root.parent_span_id is None

        # Simulate small work
        time.sleep(0.01)

        with tracer.start_as_current_span(
            "child_task", attributes={"subsystem": "retrieval"}
        ) as child:
            assert tracer.get_current_span() == child
            assert child.parent_span_id == root.span_id
            assert child.trace_id == root.trace_id
            child.set_attribute("docs.retrieved", 5)
            child.add_event("chunks_scored", {"score_max": 0.95})

        # Context reverts to root
        assert tracer.get_current_span() == root

    # Both spans should be recorded in collector
    assert tracer.get_current_span() is None
    trace = collector.get_trace(root.trace_id)
    assert trace is not None
    assert trace["span_count"] == 2
    assert trace["root_name"] == "root_task"
    assert trace["status"] == "OK"
    assert trace["total_duration_ms"] >= 10.0


def test_trace_collector_ring_buffer_eviction():
    collector = TraceCollector(capacity=3)
    tracer = Tracer(collector=collector)

    trace_ids = []
    for i in range(5):
        with tracer.start_as_current_span(f"job_{i}") as s:
            trace_ids.append(s.trace_id)

    # Capacity is 3, so first 2 should be evicted
    assert len(collector.list_traces()) == 3
    assert collector.get_trace(trace_ids[0]) is None
    assert collector.get_trace(trace_ids[1]) is None
    assert collector.get_trace(trace_ids[4]) is not None
