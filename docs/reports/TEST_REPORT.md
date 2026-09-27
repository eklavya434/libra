# Libra Test Report — Stabilization Verification

Audit 2026-09-27. All numbers are from actual runs on this machine (Intel i5-12450H, CPU-only, Windows).

## Command suite

| Command | Result |
| :-- | :-- |
| `pytest tests/ --ignore=tests/frontend` | **712 passed, 2 skipped** (Postgres parity, live-URL only) |
| `pytest tests/frontend/` | **5 passed** (structural invariants) |
| `ruff check .` | All checks passed |
| `ruff format --check .` | 520 files already formatted |
| `npm run typecheck` (apps/frontend) | clean |
| `npm run build` (apps/frontend) | compiled, 5 static pages preredendered |
| report reference audit | 48 `tests/...` citations across the 5 reports verified to resolve to real files and real test functions |

## New regression coverage added during this audit

| Area | File | What it proves |
| :-- | :-- | :-- |
| Chat plumbing | `tests/api/test_chat_endpoint.py` | honest errors, no fake streaming data |
| Provider offline | `tests/unit/test_local_provider_integrity.py` (5) | 503 when checkpoint **or** config is missing, incompatible checkpoint, `health()` matches disk truth, stream is not a bypass |
| Gemini adapter honesty | `tests/unit/test_gemini_honesty.py` (15) | `finishReason` mapped not collapsed; empty completion raises instead of faking `stop`; blocked/whitespace responses rejected; stream does not truncate silently |
| Router honesty | `tests/unit/test_dynamic_router_honesty.py` (6) | tier map is real-providers-only, no mock class referenced, exhausted chain raises, unavailable-attempt message, health gate precedes `chat()`, no tier/policy selects a mock |
| Routing 503 | `tests/api/test_routing_endpoint.py` | generic message, no internals |
| Agent/model resolution | `tests/api/test_agents_endpoint.py` | configured-provider resolution only |
| Notebook scoping | `tests/api/test_notebook_endpoint.py` | entity-scoped kernels, 404 on foreign |
| Middleware hardening | `tests/api/test_hardening_middleware.py` | chunked 413, under-limit replay, content-length 413, request-id sanitize, CSP/HSTS, CORS expose, readyz degraded, readyz no-leak, log redaction |
| Session TTL | `tests/memory/test_sqlite_memory.py` | sliding renew, expired-never-revived |
| RAG isolation | `tests/api/test_rag_endpoint.py` | A/B guest isolation, public-scope invisible to guests |
| OCR isolation | `tests/api/test_document_endpoint.py` | cross-tenant file 404, sanitized storage errors |
| Long-context | `tests/api/test_long_context_endpoint.py` | real-model generation via sync bridge; mock only for `model="mock"` |
| Context/NIAH | `tests/api/test_context_endpoint.py` | needle cheat restricted to mock; RoPE/continuity paths |
| Lab labs without checkpoints | `tests/api/test_telemetry_endpoint.py`, `tests/api/test_grammar_endpoint.py` | fall back to an explicitly-flagged untrained in-memory transformer instead of 503 on a machine with no trained checkpoint |
| Frontend invariants | `tests/frontend/test_frontend_structure.py` | no hardcoded model/mock leaks; stream guard preserved |

## Live smoke tests (running server, port 8000)

| Probe | Result |
| :-- | :-- |
| `GET /api/v1/health` | ok; `store_backend=SQLiteConversationStore`; `active_providers=[gemini, huggingface, libra_lab, nvidia]` |
| `GET /readyz` | ready; `database_url_configured=false`; no leakage |
| `GET /api/v1/models/default` | `gemini-2.5-flash` (configured); `configured=true` |
| `GET /api/v1/auth/session` | minted `sess-*` token + 30-day TTL |
| `POST /chat/completions` (bogus model) | 400, generic detail, `X-Request-Id` present |
| `POST /chat/completions` (`libra-llama-tied`, stream=false, max_tokens=16) | 200, real tokens, `finish=stop` (real first-principles inference, not fabricated) |
| `POST /chat/completions` (`gemini-2.5-flash`, stream=false, max_tokens=1024) | 200, real answer, `finish=stop`, 35 completion tokens |
| `POST /chat/completions` (`gemini-2.5-flash`, stream=true, max_tokens=1024) | SSE: metadata chunk, token delta `"Hello there!"`, `[DONE]` |
| `POST /chat/completions` (`gemini-2.5-flash`, max_tokens=1/2/4/8) | 503 honest error naming the exhausted budget (was 200 + empty + `finish=stop` before BUG-09) |

## Skips (2) — honest

`PostgresConversationStore` parity tests require `LIBRA_TEST_DATABASE_URL`/live `DATABASE_URL`. Without it they skip rather than fake a success. Enable by setting the URL in the test environment.

## Reproducibility notes

- Run backend tests with `.\\.venv\\Scripts\\python.exe -m pytest tests/ --ignore=tests/frontend -q`.
- Tests mint guest sessions internally and use them for isolation assertions.
- Deprecation warnings present: `starlette.testclient` httpx alias and anyio `BlockingPortal` alias (dependency-side, not project code).