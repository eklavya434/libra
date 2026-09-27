# Libra Bug Report — Reproducible Bugs Found & Fixed

Audit 2026-09-27. Each bug has a reproduction, root cause, fix, and a regression test.

---

## BUG-01 · Local provider fabricates output when no checkpoint exists

**Severity:** P0 · **Repro:**
1. Start the API with no `checkpoints/best_engine_model.pt` (e.g. fresh clone).
2. `POST /api/v1/chat/completions` with `model="libra-llama-tied"`.
3. Old behavior: HTTP 200 with a canned/pretender completion — a fake success masked as real inference.

**Root cause:** `LocalTransformerProvider` fell back to a lightweight placeholder path instead of refusing.

**Fix:** `packages/providers/local_transformer.py` now raises `ProviderOfflineError` unless `checkpoint_path` AND architecture `config_path` exist; `is_ready()` returns the same disk truth; the API maps the error to an honest 503.

**Verify:** `tests/unit/test_local_provider_integrity.py` (5 tests: missing checkpoint, missing config, incompatible checkpoint, health disk-truth, stream guard), `tests/providers/test_local_transformer_provider.py::test_local_transformer_offline_when_checkpoint_missing`; live `/api/v1/models` shows `available` for `libra-llama-tied` only because the checkpoint + `configs/models/tiny_modern_tied.yaml` exist on this machine.

---

## BUG-02 · Dynamic router could select a mock tier when real tiers "failed"

**Severity:** P0 · **Repro:**
1. Configure no real provider.
2. Any request routed through `router.generate(...)`.
3. Old behavior: silently returned mock-tier output labelled as the selected tier.

**Root cause:** router included a mock tier and a fallback that treated it as a real engine.

**Fix:** `packages/routing/dynamic_router.py` has a real tier map only. All tiers off → `RuntimeError` → generic 503 in `routing.py`. No mock anywhere in the path.

**Verify:** `tests/unit/test_dynamic_router_honesty.py` (6 tests: real-only tier map, no mock class referenced, exhausted chain raises, no-available-attempt message, health gate precedes chat, no tier/policy selects a mock), `tests/api/test_routing_endpoint.py::test_endpoint_routing_generate_fails_loudly_when_no_provider`.

---

## BUG-03 · `runtime.generate()` AttributeError crashes long-context & context views on real models

**Severity:** P0 · **Repro:**
1. `POST /api/v1/long-context/generate` with any cloud/ollama model id.
2. Old behavior: `AttributeError: 'ModelRouter' object has no attribute 'generate'` (500) because the router exposes `chat()`/`stream()`, not `generate()`.

**Root cause:** `long_context.py` and `context.py` (context needle path) called a method that does not exist, then a "fix" injected a fabricated generator for real models.

**Fix:** `apps/backend/api/v1/deps.py` provides `run_provider_completion` — an async→sync bridge that calls the router's real `chat()` and extracts the completion text via `extract_provider_text`. On provider failure it returns an honest 503, never a fake string.

**Verify:** `tests/api/test_long_context_endpoint.py`, `tests/api/test_context_endpoint.py`.

---

## BUG-04 · Needle-in-a-haystack claimed ~100% accuracy by cheating

**Severity:** P0 · **Repro:**
1. `POST /api/v1/context/needle` with any real model.
2. Old behavior: every trial "correct" because retrieval matched the target key without reading the context.

**Root cause:** `context.py` had a hard-coded correct answer for the target key on the real-model path.

**Fix:** The cheat path is reachable only when `model="mock"` (explicitly a mock). Real models are scored on actual retrieval.

**Verify:** `tests/api/test_context_endpoint.py`.

---

## BUG-05 · OCR files readable by any guest session

**Severity:** P1 · **Repro:**
1. Guest A uploads via `POST /api/v1/document/parse` (file mode).
2. Guest B issues `GET /api/v1/document/files/<key>`.
3. Old behavior: B fetched A's file.

**Root cause:** `document_ocr.py` tracked no upload ownership.

**Fix:** `_file_owner` map {key → session}. Non-owner access → 404. Storage exceptions are also sanitized.

**Verify:** `tests/api/test_document_endpoint.py::test_uploaded_file_cannot_be_read_by_other_guest`.

---

## BUG-06 · `/readyz` leaked exception internals

**Severity:** P1 · **Repro:**
1. Cause a conversation-store failure.
2. `GET /readyz`.
3. Old behavior: 503 body contained the raw traceback/exception string.

**Fix:** `main.py` returns a generic message plus `store_backend`, `database_url_configured`, and `degraded` status when applicable. No `str(exc)` in the response.

**Verify:** `tests/api/test_hardening_middleware.py::test_readyz_never_leaks_exception_details`, `test_readyz_degraded_reflects_store_backend`.

---

## BUG-07 · Chunked-encoding bodies bypassed the request-size limit

**Severity:** P1 · **Repro:**
1. Send a body with `Transfer-Encoding: chunked` larger than the configured limit (no Content-Length).
2. Old behavior: processed (or crashed) instead of 413.

**Root cause:** the limiter was a `BaseHTTPMiddleware` that inspected `request._body` — which is only populated by the Content-Length path; BaseHTTPMiddleware builds a fresh downstream Request.

**Fix:** `apps/backend/middleware/security.py` rewrote the limiter as pure ASGI middleware that owns `receive`, buffers bounded chunks, returns 413 when the budget is exceeded, and replays the buffered body downstream otherwise.

**Verify:** `tests/api/test_hardening_middleware.py::test_chunked_body_over_limit_rejected`, `test_chunked_body_under_limit_replayed_to_app`, `test_content_length_over_limit_rejected`.

---

## BUG-08 · Frontend error surface dumped raw bodies and suggested a mock

**Severity:** P2 · **Repro:**
1. Trigger a failed chat request.
2. Old behavior: the error bubble contained the full raw response body and the guidance "select Libra Mock v1".

**Fix:** `api.ts::extractBackendError` yields the backend `detail` (generated to be honest and generic server-side); `ChatArea.tsx` copy now instructs checking backend/provider configuration.

**Verify:** `npm run typecheck`, `npm run build`, component inspection.