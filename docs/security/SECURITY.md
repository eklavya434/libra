# Libra Security Model

Security hardening summary for the public (zero-cost, guest-only) Libralab
deployment. This is an **educational monorepo**, so every control below is kept
simple enough to audit line-by-line, and every residual risk is explicit.

## Deployment reality (why controls look this way)

- Public service, **no user accounts**. Visitors are `guest` sessions with
  per-browser conversation isolation, not tenants with credentials.
- **Zero budget**: no commercial WAF, no auth SaaS (Clerk/Auth0), no managed
  Postgres. All controls are code-level or platform-proxy-level.
- Code execution and RAG/document features exist because they are the mission;
  security is implemented by **feature gates** rather than sandboxing magic we
  could not audit honestly.

## Threat model

| Asset | Threat | Primary control |
| :-- | :-- | :-- |
| API availability | Abuse floods, compute exhaustion | Token-bucket rate limit per client IP (120/min, burst 30) over 15 compute prefixes. Platform edge throttling. |
| Memory | Oversized request bodies | `RequestSizeLimiterMiddleware` → 413 over 10 MB. |
| Cross-site | Browser CSRF, clickjacking, MIME sniffing | Custom header-only secret model (`X-Libra-Session`, no cookies) + CORS allowlist + headers (`X-Frame-Options: DENY`, `nosniff`, CSP adjacents, `Referrer-Policy`, `Permissions-Policy`). |
| Conversation privacy | Guest A reads guest B | Sessions keyed by random 256-bit tokens (`secrets.token_hex(32)`), SHA-256 at rest; every conversation/message query is owner-scoped (`owner_id`); foreign IDs expose 404/403. |
| Data leakage | Secrets in errors/logs | Exception handlers return `{"detail":"Internal server error","request_id":...}` (no stack/PII); `X-Request-Id` correlation; logs redact key patterns. |
| Arbitrary code | RCE via notebook/coder/tools | **`LIBRA_PUBLIC_CODE_EXEC=false` → notebook `/execute` returns 503 by default.** Enabling requires the subprocess sandbox roadmap (below). |
| Injection | Prompt injection exfil | Recognized LAB risk; documented in `docs/research`; no tool-autonomy paths enabled in public. |
| Data tampering | SQLite manipulation | Only write path is the API layer via the store; single-owner files under `data/`. |
| Disk exhaustion | Fill `/app/data` | `MAX_DISK_USAGE_PERCENT` guard; ephemeral FS on the free platform resets on redeploy (no runaway-growth surface). |

## Controls by layer (where they live)

- **Middleware chain** (`apps/backend/main.py`, outermost→inner):
  CORS → SecurityHeaders → RequestSizeLimiter → RateLimiting →
  Tracing/RequestId → SessionIdentity.
- **Identity** (`middleware/identity.py`, `endpoints/auth.py`):
  header `X-Libra-Session`; token minted only by `POST /api/v1/auth/session`;
  `GET` resolves info without minting; `POST /logout` revokes; TTL cleanup runs
  on startup lifespan. No cookie, no password, no PII.
- **Errors** (`middleware/observability.py`): reliable JSON per status class;
  500s never leak internals; probe endpoints `/healthz`, `/readyz`, `/livez`.
- **Rate limiting** (`middleware/security.py`): config-driven
  `LIBRA_RATE_LIMIT_PER_MINUTE`/`BURST`, `X-Forwarded-For` trusted only per
  `LIBRA_TRUSTED_PROXIES`; `Retry-After` + `X-RateLimit-*` headers.
- **CORS** (`core/config.py`): `CORS_ORIGINS` JSON allowlist; no wildcards in
  prod compose/render blueprint.

## Known boundaries (accepted, documented)

1. **No true multi-tenant auth.** Guests are pseudonymous; a token holder's
   data is only as secret as the token. Do not store PII. Adding real users =
   out of scope of the current zero-cost mission; the `owner_id`/session schema
   is the clean upgrade point.
2. **XFF spoofing on direct exposure.** If you ever bind the backend directly
   to the internet without the platform edge, set `LIBRA_TRUSTED_PROXIES=0`,
   accepting that rate limiting keys then on `request.client` (edge still sees
   the real IP) — otherwise an attacker who can set `X-Forwarded-For` bypasses
   the bucket.
3. **Code execution is off, not sandboxed.** Re-enabling requires the subprocess
   jail (rlimit/fork sandbox or gVisor) with unit tests proving no host escape —
   until then the gate stays closed.
4. **Knowledge/upload persistence is ephemeral on the free tier** (single
   instance, no free persistent disk). Persistent multi-instance storage
   requires Postgres/S3 adapters (documented, not shipped; paid disk works on a
   single instance).
5. **No secrets rotation automation** yet; keys are changed manually via the
   platform dashboard.

## Verification (all in CI / test suite)

- `tests/api/test_security.py` — headers, size limiter, rate limit 429 paths.
- `tests/api/test_session_identity.py` — isolation, foreign conv 403/404,
  invalid-token fallback, cookie-less guest model, no stack leak on 500.
- `tests/api/test_notebook_endpoint.py` — default 503 gate; kernel/preset paths
  still pass with the flag on.
- `ruff check` + frontend `tsc` clean; full suite 610 passing.

## Incident runbook (operator)

1. Symptom 429 storm → scale reduces to: check edge IPs, raise limits by one
   unit, inspect logs for one source IP pattern.
2. `Internal server error` + `request_id` → pull that `request_id` from logs;
   stack traces are server-side only.
3. Keys leak → rotate in dashboard, touch nothing in git (they are absent).
4. Data restore → `data/conversations.db` snapshot from Render disk backup.

## Sandbox roadmap (when authorization + tokens exist)

1. `packages/tools/` executor moves to a `subprocess` jail (uid drop, rlimits,
   network namespace, no writable shared FS) — required `LIBRA_PUBLIC_CODE_EXEC=true`.
2. Provenance log for every executed cell (who/what/interal UUIDs).
3. Postgres adapter + secret rotation via a proper KMS-style secret store.