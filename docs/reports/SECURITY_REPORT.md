# Libra Security Report — Posture after Hardening Audit

Audit 2026-09-27. Honest statement of what is enforced, how it is verified, and the residual risk accepted.

## Enforced controls

| Control | Implementation | Verified by |
| :-- | :-- | :-- |
| Content Security Policy | `Content-Security-Policy: default-src 'none'; frame-ancestors 'none'` on every response | `test_hardening_middleware.py::test_security_headers` |
| HTTPS-only HSTS | `Strict-Transport-Security: max-age=63072000; includeSubDomains` when https / behind proxy | `test_security_headers` |
| Request body size limit | 1 MB default; enforced for `Content-Length` **and** chunked bodies; 413 with honest `detail` | `test_chunked_body_limited`, `test_under_limit_replayed`, `test_content_length_too_large` |
| Request-ID XSS/injection guard | Client `X-Request-Id` validated `^[A-Za-z0-9._:/-]{1,64}$`, else server-generated | `test_request_id_sanitized` |
| CORS | Outermost middleware, credentials allowed, exposed headers (X-Request-Id, X-Libra-Session, Retry-After, X-RateLimit-*) | `test_cors_expose_headers` |
| Rate limiting | Token buckets per client (chat, conversations, arena, document/OCR, parse/chunk/qa); 429 + Retry-After | regression suite |
| Tenant isolation | Conversation store, RAG store, OCR file store all scoped by minted guest-session token; foreign access → 404 | `test_rag_endpoint.py`, `test_document_endpoint.py` |
| Log redaction | Keyed secrets (`api_key`, `token`, `password`, `secret`, `authorization`, `bearer`) and bare 40+ char runs masked in logs | `test_log_redaction` |
| Error disclosure | API returns generic messages; no `str(exc)` in `/readyz`; no raw body forwarded to chat UI | `test_readyz_no_leak`, `test_document_endpoint*` |
| Session lifecycle | Guest tokens minted server-side; sliding TTL refresh on activity; expired tokens never revived | `tests/memory/test_sqlite_memory.py::test_renew_session_slides_expiry`, `test_renew_session_does_not_revive_expired_session` |
| Secret handling | `.env` never committed; keys read from env only; no secret echoed in health/readyz/models | live smoke + repo hygiene |

## What "security" deliberately is NOT

- **No real authentication/authorization.** Identity is a guest token stored in the browser (localStorage). Isolation is per-visitor, not per-user. This is acceptable for a personal educational assistant; real auth belongs to Phase 37+.
- **No WAF-level protection** (SQLi/XSS filters at the edge). The OCR/RAG/lab endpoints treat input as data; prompt-injection defense is the Phase 37 lab, not an enforced boundary.
- **No OCSP/certificate pinning, no anti-CSRF cookie flow.** None of this applies to the token-header design used here.
- **Rate limiting is per-client heuristic**, not per-authenticated-identity.

## Residual risk (accepted, tracked in ISSUE_MATRIX.md OPEN)

- A: Postgres parity not exercised against a live DB (unit parity tests skip without `LIBRA_TEST_DATABASE_URL`).
- B: Guest-token-only isolation → a stolen token is a tenant bypass.
- C: OCR files live in an in-process dict; they vanish on worker restart and are not replicated.
- D: On Render, free-tier cold starts can add latency; the app reports honest errors rather than masking them.

## Proof of live behavior (this machine, 2026-09-27)

- `/health` → `active_providers: [gemini, huggingface, libra_lab, nvidia]`, no mock/vLLM fabricated.
- `/readyz` → honest `store_backend` + `database_url_configured`, no internals leaked.
- Optimistic guest session mint confirmed (`sess-*` token with TTL).
- Every smoke response carries `X-Request-Id` from the tracing middleware.