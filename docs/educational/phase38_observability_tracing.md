# Phase 38: Observability, Distributed Tracing & OpenTelemetry Logging

## 1. WHAT Was Built
Phase 38 introduces an OpenTelemetry-compatible distributed tracing and telemetry engine built from first principles for Project Libra.

Key modules implemented:
- **`Tracer` & `Span` (`packages/core/observability/tracer.py`)**:
  - 128-bit `TraceId` and 64-bit `SpanId` generation conforming to W3C standards.
  - Thread-safe context management with `contextvars` for hierarchical parent-child span nesting.
  - OpenTelemetry span semantics (attributes, events, status codes, microsecond-accurate durations).
  - W3C `traceparent` header parser and serializer (`00-{trace_id}-{span_id}-{flags}`).
  - In-memory bounded `TraceCollector` ring buffer (default 100 traces) for zero-dependency local analysis.
- **`TokenVelocityMetrics` & `MetricsAggregator` (`packages/core/observability/metrics.py`)**:
  - Time to First Token (TTFT) in milliseconds.
  - Inter-Token Latency (ITL) distribution: mean, median (p50), and 95th percentile (p95).
  - Autoregressive generation velocity (tokens per second).
  - Rolling request latency percentiles (p50, p95) and error rate tracking.
- **`TracingMiddleware` (`apps/backend/middleware/tracing.py`)**:
  - Intercepts all incoming HTTP requests, establishes a root span, and attaches OpenTelemetry attributes.
  - Injects `X-Trace-Id` and W3C `Server-Timing: total;dur=...` headers on all responses.
- **Observability REST Endpoints (`apps/backend/api/v1/endpoints/observability.py`)**:
  - `GET /api/v1/observability/traces`: List recent request traces.
  - `GET /api/v1/observability/traces/{trace_id}`: Fetch detailed span hierarchy with timeline offsets for Gantt rendering.
  - `GET /api/v1/observability/metrics`: Summary percentiles and token velocities.
  - `POST /api/v1/observability/traces/clear`: Reset buffer.
- **Frontend Observability & Waterfall UI (`apps/frontend/src/components/ObservabilityView.tsx`)**:
  - Interactive Gantt chart latency waterfall visualization.
  - Span detail drawer displaying OpenTelemetry attributes and events.
  - Real-time telemetry cards (TTFT, TPS, p50/p95 latency).

---

## 2. WHY It Exists
In modern conversational AI and multi-step agent systems, a single user interaction is not a monolithic database query. It is a multi-stage distributed workflow:
$$\text{HTTP Ingress} \longrightarrow \text{Prompt Guard} \longrightarrow \text{Dynamic Router} \longrightarrow \text{RAG Embedding/BM25} \longrightarrow \text{Transformer Prefill} \longrightarrow \text{Cached Decode} \longrightarrow \text{Secret Redactor}$$

When a user perceives high latency, traditional log lines provide disconnected timestamps that make root-cause analysis difficult.

Distributed tracing solves this by:
1. **Unifying the Execution Context**: A single 128-bit `trace_id` correlates every intermediate step across background tasks, retrieval modules, and LLM forward passes.
2. **Exposing the Critical Path**: Waterfall Gantt charts visually isolate which span took the majority of the time (e.g. 80% in external API vs 10% in dense vector retrieval).
3. **Quantifying Token Velocity**: Separates initial prompt processing latency (TTFT) from autoregressive generation rate (tok/s), which is crucial for CPU hardware performance tuning.

---

## 3. HOW It Works (The Math & Mechanics)

### A. W3C TraceContext Specification
A standard W3C `traceparent` header has the format:
$$\text{version}-\text{trace\_id}-\text{parent\_id}-\text{trace\_flags}$$
- `version`: Always `00`.
- `trace_id`: 16 bytes (32 hex characters) uniquely identifying the distributed trace.
- `parent_id`: 8 bytes (16 hex characters) identifying the caller span.
- `trace_flags`: 8-bit bitmap (`01` indicates sampled).

### B. High-Precision Span Durations
Durations are measured using the CPU's monotonic hardware clock via `time.perf_counter_ns()`, ensuring immunity against system clock updates or NTP drift:
$$\Delta t_{\text{ms}} = \frac{t_{\text{end\_ns}} - t_{\text{start\_ns}}}{1{,}000{,}000}$$
Relative waterfall offset from the root span start:
$$\text{Offset}_{\text{ms}} = \frac{t_{\text{span\_start\_ns}} - t_{\text{root\_start\_ns}}}{1{,}000{,}000}$$

### C. Token Velocity & Inter-Token Latency (ITL)
For a generation sequence producing $N$ tokens at arrival timestamps $T_0, T_1, \dots, T_{N-1}$:
- $\text{TTFT} = T_0 - T_{\text{request\_start}}$
- $\text{ITL}_i = T_i - T_{i-1} \quad \text{for } i \in [1, N-1]$
- $\text{Velocity} = \frac{N}{(T_{N-1} - T_{\text{request\_start}}) / 1000} \quad (\text{tokens/second})$

---

## 4. TEST Verification
The observability subsystem is verified with unit and endpoint test suites:
- `tests/observability/test_tracer.py`: Span lifecycle, W3C traceparent parsing, context manager nesting, ring buffer eviction.
- `tests/observability/test_metrics.py`: TTFT calculation, ITL percentiles, rolling metrics aggregation.
- `tests/api/test_observability_endpoint.py`: `/api/v1/observability/traces`, `/traces/{trace_id}`, `/metrics`, and middleware `X-Trace-Id` / `Server-Timing` headers.

---

## 5. NEXT Steps
With Phase 38 complete, Project Libra has full production observability and latency profiling. Phase 39 will focus on **High-Throughput Batch Processing & Asynchronous Queue Workers**.
