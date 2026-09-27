# Libra Feature Matrix — Honest Status

Legend: **PROD** = verified working on the deployed app; **FAIL-HONEST** = works when configured, fails loudly/honestly when not; **LAB** = educational simulator with real, unfabricated math/data; **N/A** = intentional absent (documented).

## Core assistant surface

| Feature | Status | Evidence |
| :-- | :-- | :-- |
| Chat completions (stream + non-stream) | PROD | live smoke: `libra-llama-tied` → 200, real tokens, `finish=stop` |
| Default-model resolution (`/api/v1/models/default`) | PROD | returns `gemini-2.5-flash` (configured key present), never mock |
| Multi-turn conversation persistence (SQLite WAL) | PROD | `tests/memory/`, `tests/api/test_conversations*.py` |
| Streaming markdown / code copy / abort | PROD | frontend build + component inspection |
| Visitor guest-session minting | PROD | live: `GET /api/v1/auth/session` → `sess-*` + token + TTL |
| Sliding session TTL (refresh on activity) | PROD | `tests/memory/test_sqlite_memory.py` |
| Cross-session isolation (conversations/RAG/OCR) | PROD | `tests/api/test_rag_endpoint.py`, `test_document_endpoint.py` |
| Chat with unconfigured/unavailable model | FAIL-HONEST | generic 4xx/503, never fake; smoke: bogus model → 400 + `X-Request-Id` |
| Chat error UX | PROD | backend `detail` surfaced, no raw body, no mock suggestion |

## Providers & models

| Feature | Status | Evidence |
| :-- | :-- | :-- |
| Cloud providers (gemini/anthropic/openai/nvidia/deepseek/kimi/groq/openrouter) | PROD | registered; default resolution requires real key |
| Ollama local runtime | PROD | auto-registered on `/models` when reachable |
| `libra_lab` first-principles models | PROD | checkpoint+config present on this machine; 503 elsewhere |
| No mock used in production paths | PROD | `dynamic_router`, chat, agents, context hardened; tests assert absence |
| `/health` active providers | PROD | live: `[gemini, huggingface, libra_lab, nvidia]`, no mock/vLLM |
| `/readyz` | PROD | live: `ready`, `store_backend`, `database_url_configured=false` |

## Lab sections (educational, honest data)

| Feature | Status | Evidence |
| :-- | :-- | :-- |
| Model Arena (compare/battle/tournament/Elo) | LAB | real metrics from real provider calls; no synthetic scores |
| RAG lab (BM25+dense RRF, rerank, dedupe) | PROD/LAB | `packages/rag/hybrid.py`; scoping verified by tests |
| Security lab (scan/redact/stats) | LAB | phase 37 simulators; honest risk scores |
| Observability (traces/metrics) | LAB | `TracingMiddleware` + buffered trace store |
| Batch inference + paged memory | LAB | `packages/batch/` simulator with real math |
| OCR parse/chunk/QA | PROD/LAB | `document_ocr.py`; tenant-scoped files (see OPEN C) |
| Notebook kernel sandbox | PROD/LAB | entity-scoped jails; `test_notebook_jail.py` |
| Long context / NIAH, MCTS, distillation, MoE, Medusa, KTO, verifiable search, self-rewarding, grand capstone | LAB | real implementations of the underlying math; NIAH cheats restricted to mock model only |

## App infrastructure

| Feature | Status | Evidence |
| :-- | :-- | :-- |
| Next.js build + typecheck | PROD | `npm run build`, `npm run typecheck` green |
| CSP / HSTS / security headers | PROD | `tests/api/test_hardening_middleware.py` |
| Request size limit (content-length + chunked) | PROD | `tests/api/test_hardening_middleware.py` |
| Rate limiting across chat + OCR | PROD | regression suite |
| X-Request-Id tracing | PROD | live smoke (present on every response) |
| Log redaction | PROD | `tests/api/test_hardening_middleware.py::test_log_redaction` |
| CORS (outermost, credentials, exposed headers) | PROD | `tests/api/test_hardening_middleware.py::test_cors_expose_headers` |
| Mobile responsive chat (drawer + toggle) | PROD | build + structural tests |
| CI (lint, backend, frontend, deploy audit, regression) | PROD | workflow always-checks; format gate now passes |

## Explicitly NOT shipped

| Feature | Status | Evidence |
| :-- | :-- | :-- |
| vLLM on this machine | N/A | AGENTS.md: document as GPU-only, run Ollama locally |
| Docker before Phase 34 | N/A | AGENTS.md Prime Directive 5 |
| Paid-API auto-invocation | N/A | zero-cost policy; only free-tier/local usage |
| Mock-as-production fallback | N/A | issue #2/#3 fixed and tested |