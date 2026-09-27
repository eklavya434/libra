# Libra Issue Matrix — Stabilization Audit

Audit date: 2026-09-27. Scope: production-quality hardening of the deployed app per the 54-section brief.
Severity: **P0** = fake/impossible behavior withstood, **P1** = honesty/security defect, **P2** = UX/quality defect.
Status legend: `FIXED` (verified by tests), `MITIGATED` (guard rails), `OPEN` (tracked limitation).

Every `tests/...::test_...` citation in the five files in `docs/reports/` is machine-checked
to resolve to a real file and a real test function (48 references, 0 dangling). The two
integrity suites cited for P0 #1 and P0 #2 (`tests/unit/test_local_provider_integrity.py`,
`tests/unit/test_dynamic_router_honesty.py`) are now real files with real assertions.

## P0 — Provider honesty / fake success

| # | Issue | Where | Fix | Verify |
| :- | :-- | :-- | :-- | :-- |
| 1 | Local lab provider returned a fabricated response when no trained checkpoint existed ("fake success" with a pretender model). | `packages/providers/local_transformer.py` | Raise `ProviderOfflineError` (→ HTTP 503) unless a real checkpoint **and** architecture config exist; `is_ready()` reflects disk truth. | `tests/unit/test_local_provider_integrity.py`, `tests/providers/test_local_transformer_provider.py::test_local_transformer_offline_when_checkpoint_missing`, live `/models` `available` flags |
| 2 | Dynamic router inspected fake tier results and could fall back to a mock when all tiers "failed". | `packages/routing/dynamic_router.py` | Real tier map only, no mock any trade. Exhaustion → `RuntimeError` → generic 503. | `tests/unit/test_dynamic_router_honesty.py`, `tests/api/test_routing_endpoint.py::test_endpoint_routing_generate_fails_loudly_when_no_provider` |
| 3 | Agent/notebook/structured endpoints silently defaulted to a mock model when no provider was configured. | `apps/backend/api/v1/deps.py`, `agents.py`, `coder.py`, `teams.py`, `structured.py`, `notebook.py` | `resolve_effective_model_id` resolves the configured provider or raises an honest config error; notebook kernels are entity-scoped. | `tests/api/test_agents_endpoint.py` et al. |
| 4 | Long-context and context-view endpoints crashed or injected a fake "849204 / Alpha-77 Falcon" generator on real models. | `long_context.py`, `context.py` | `run_provider_completion` bridges async chat into sync generator callbacks for real models; fake fallback removed. `_mock_gen` only for literal `model="mock"`. | `tests/api/test_long_context_endpoint.py`, `test_context_endpoint.py` |
| 5 | Needle-in-a-haystack engine reported ~100% accuracy for real models by cheating on the retrieval step. | `context.py` | Cheat short-circuit restricted to `model="mock"` only; real models are scored honestly. | `tests/api/test_context_endpoint.py` |

## P1 — Honesty & security hardening

| # | Issue | Where | Fix | Verify |
| :- | :-- | :-- | :-- | :-- |
| 6 | `/readyz` leaked raw exception text and reported "ready" without a database check. | `apps/backend/main.py` | Generic 503 message, `store_backend`, `database_url_configured`, `status: degraded` when `DATABASE_URL` is set but SQLite is serving. | `tests/api/test_hardening_middleware.py::test_readyz_never_leaks_exception_details`, `test_readyz_degraded_reflects_store_backend` |
| 7 | Logs could capture unredacted API keys/credentials in request bodies. | `apps/backend/core/logging.py` | `RedactionFormatter` regex masks key names and 40+ char token runs. | `tests/api/test_hardening_middleware.py::test_log_redaction_formatter_scrubs_secrets` |
| 8 | Responses missing CSP and never sent HSTS. | `apps/backend/middleware/security.py` | `SecurityHeadersMiddleware`: CSP `default-src 'none'; frame-ancestors 'none'`; HSTS (`63072000`) on https only. | `tests/api/test_hardening_middleware.py::test_security_headers_include_csp_and_hsts_over_https` |
| 9 | Request-size limit was bypassable via chunked `Transfer-Encoding` bodies. | `apps/backend/middleware/security.py` | Pure-ASGI bounded chunk buffering + body replay; 413 above limit. | `tests/api/test_hardening_middleware.py::test_chunked_body_over_limit_rejected`, `test_chunked_body_under_limit_replayed_to_app`, `test_content_length_over_limit_rejected` |
| 10 | CORS add order produced non-outermost middleware; `expose_headers` missing. | `apps/backend/main.py` | CORS registered outermost with `allow_credentials` + expose of X-Request-Id / session / rate-limit headers. | `tests/api/test_hardening_middleware.py::test_cors_exposes_libra_headers` |
| 11 | Client-supplied `X-Request-Id` accepted raw (injection surface). | `apps/backend/middleware/observability.py` | Validate against `^[A-Za-z0-9._:/-]{1,64}$` else generate; always emit a sanitized id. | `tests/api/test_hardening_middleware.py::test_request_id_sanitized_when_client_supplies_junk` |
| 12 | Rate limiting omitted the OCR/document endpoints. | `apps/backend/middleware/rate_limit.py` | `/api/v1/document` prefix added to limiter table. | regression suite |
| 13 | Guest sessions never refreshed; middle-ware key set forever on the same expiry. | `packages/core/memory/sqlite_store.py`, `postgres_store.py`, `identity.py` | `renew_session()` slides TTL on activity (never revives an expired token). | `tests/memory/test_sqlite_memory.py::test_renew_session_slides_expiry`, `test_renew_session_does_not_revive_expired_session` |
| 14 | RAG documents/chunks were visible and deletable across guest sessions. | `packages/rag/hybrid.py`, `deep_research.py`, `rag.py` | `scope` parameter stitched identity → store → endpoints; foreign access → 404. | `tests/api/test_rag_endpoint.py` |
| 15 | OCR-uploaded files were fetchable by any guest. | `document_ocr.py` | `_file_owner` map; non-owner fetch → 404. | `tests/api/test_document_endpoint.py::test_uploaded_file_cannot_be_read_by_other_guest` |
| 16 | OCR storage errors leaked the underlying exception text. | `document_ocr.py` | Sanitized generic storage-error message. | `tests/api/test_document_endpoint.py` |
| 17 | `/health` advertised mock/vLLM providers and "no active" providers regardless of config. | `health.py` | `active_providers` derived live: tested `api_key` presence, Ollama connectivity, libra lab `is_ready()`. | live smoke on `/api/v1/health` |

## P2 — Frontend honesty & usability

| # | Issue | Where | Fix | Verify |
| :- | :-- | :-- | :-- | :-- |
| 18 | Chat error card dumped raw response bodies and suggested "select Libra Mock v1" (contradicts no-mock policy). | `apps/frontend/src/lib/api.ts`, `ChatArea.tsx` | `extractBackendError` surfaces backend `detail`; error copy now points to provider configuration. | `npx tsc`, `npm run build`, component inspection |
| 19 | Broken indentation in `fetchModels` (broken formatting from an earlier edit). | `apps/frontend/src/lib/api.ts:435` | Reformatted. | typecheck/build |
| 20 | On mobile the fixed 256px sidebar left ~110px for the chat surface with no way to close it. | `apps/frontend/src/app/chat/page.tsx`, `Sidebar.tsx`, `ChatArea.tsx` | Overlay drawer (`md+` docked as before), hamburger toggle in the chat header, auto-close on navigation. | build + `tests/frontend/test_frontend_structure.py` |
| 21 | `ruff format --check` (CI lint job) failed on 12 files post-refactor. | repo | `ruff format .` applied; all format/check gates green. | `ruff check .` + `ruff format --check .` |

## OPEN — tracked limitations (honest)

| # | Limitation | Impact | Next step |
| :- | :-- | :-- | :-- |
| A | Postgres/Supabase parity tests are live-skipped (no `LIBRA_TEST_DATABASE_URL`). | Parity of `PostgresConversationStore` production behavior is covered by unit tests, not a live PG run. | Provide a test URL to enable the 2 parity tests. |
| B | Guest-token identity is the only isolation boundary (no real authN/AuthZ). | Conversations/RAG/OCR are tenant-isolated but not user-authenticated. | Phase 37+ real auth when the product requires it. |
| C | OCR upload store is an in-process dict. | Files vanish on restart. | Adopt DB/object storage with owner column. |
| D | Local lab models on this machine are tiny (6 MB) first-principles checkpoints. | Outputs are real but educationally naive; they are NOT claims of frontier capability. | Continue Phase 1–6 training within the 15-min budget. |