# Libra Mission Workstream — Deployment & Readiness Report

Status snapshot for the public-deployment workstream (the "ONE recruiter URL"
mission). Backend regression: **665 passed, 2 skipped**; frontend `tsc --noEmit`
clean and `next build` succeeds locally.

## 1. What is ready (implemented + tested, deploy-path safe)

| Area | State |
| :-- | :-- |
| Landing page `/` | Public, recruiter-facing hero + features + live health card; CTA → `/chat`. |
| Assistant app `/chat` | Guest-first chat shell (moved from `/`); model resolved from backend, never hardcoded; no mock fallbacks. |
| Same-origin API | `api.ts` calls `/api/v1/*` on the same origin by default (`API_BASE_URL` → `""`); `next.config.mjs` rewrites `/api/*` → backend (Vercel → Render, local → `127.0.0.1:8000`). No CORS on the shared origin. |
| Mock-free backend | `get_system_default_model()` selects only configured providers; no silent mock fallbacks; `/models/default` is honest. |
| Durable conversations | `PostgresConversationStore` (psycopg3) auto-selected via `DATABASE_URL`; schema `supabase/migrations/0001_conversation_store.sql`; graceful SQLite fallback. |
| Object storage | `SupabaseStorageStore` (bucket) / local disk / memory (tests only); document upload returns storage info + `GET /api/v1/document/files/{key}`. |
| Identity | Guests default; JWT sessions layered on via `LIBRA_SUPABASE_AUTH_ENABLED` (HS256, aud/sub/exp checks). |
| Code execution | Default OFF exact mission message; opt-in `LIBRA_CODE_SANDBOX=jail` subprocess worker (timeouts + POSIX rlimits) or `inprocess`. |
| CI/CD | Python CI matrix (lint/format/tests/frontend typecheck+build/audit) already green; `vercel.json` (Vercel auto-deploy); `render.yaml` updated (DATABASE_URL/SUPABASE/CORS); `supabase/config.toml` for `supabase db push`. |

## 2. Provider report (live inference)

- Configured keys in `.env` (dev only): `GEMINI_API_KEY`, `NVIDIA_API_KEY`.
- Default-model priority when configured: `gemini-2.5-flash` → Anthropic →
  NVIDIA → OpenAI → DeepSeek → Kimi. First configured, healthy provider wins.
- Unconfigured/missing provider: `/models/default` returns `configured:false`
  with guidance; chat falls back to the local lab model `libra-llama-tied`,
  **no fabricated metrics** anywhere.
- Local: Ollama on the owner's PC only. vLLM documented as GPU-only future.
- Cost: $0 (free tiers only).

## 3. Security report

- Secret scan CI gate PASS; keys gitignored; no endpoint echoes keys
  (tested). 15 GB storage quota gate PASS.
- Notebook code exec: 503 by default with the exact mission message;
  opt-in jail = process boundary + wall-clock kill + POSIX rlimits; `inprocess`
  explicitly trusted-operator-only.
- Guest sessions hashed at rest; per-user isolation tested (`supabase:<sub>`
  scoping under optional JWT mode).
- Rate limiting (120/min, burst 30) + `X-Libra-Session` + CORS allowlist.
- Known accepted boundaries (no true multi-tenant auth, ephemeral disk without
  Supabase) documented in `docs/security/SECURITY.md`.

## 4. Deployment report & workspace classification

- **Public**: `apps/frontend` (Vercel, root `/` landing → `/chat`), `apps/backend`
  (Render free, Docker), `supabase/` (Postgres migration + storage + optional
  Auth), CI/CD + deploy workflows, docs.
- **Local lab (owner PC only)**: `packages/models`, `packages/training`,
  `packages/evaluation`, local model code + checkpoints, Ollama.
- **Provider adapters**: `packages/providers` (no app logic inside).
- Classifier summary: no monoliths; chat app / LLM lab / provider adapters stay
  separated per AGENTS.md.

## 5. Manual owner actions (browser-auth boundary — cannot be automated)

1. **Supabase**: create free project → `supabase login` + `supabase link` +
   `supabase db push` (applies `0001_conversation_store.sql`) → create storage
   bucket `libra-files` (public-read bucket policy or signed URLs).
2. **Render**: sign up → New → Blueprint → import repo → on `libra-backend` set
   `GEMINI_API_KEY`, `NVIDIA_API_KEY`, `DATABASE_URL`, `SUPABASE_URL`,
   `SUPABASE_SERVICE_ROLE_KEY` (secrets, never in git).
3. **Vercel**: sign up → import repo → project root `apps/frontend` →
   Vercel auto-deploys `main` (optionally set `NEXT_PUBLIC_API_URL` if the
   rewrite destination changes).
4. **DNS**: point the recruiter domain at the Vercel origin; the backend stays
   on `*.onrender.com` behind the same-origin `/api/*` rewrite.
5. **Post-launch validation**: run the "Verify" checklist in
   `docs/deployment/PRODUCTION.md` (phone + desktop, second-private-window
   isolation, 429 rate-limit, notebook 503 mission message).
6. Optional: `LIBRA_SUPABASE_AUTH_ENABLED=true` + `SUPABASE_JWT_SECRET` to add
   real user sessions on top of guests.

## Verification lineage (final regression)

`pytest -p no:warnings` → **665 passed, 2 skipped** (2 live-Postgres tests
skipped honestly: gated on `LIBRA_TEST_DATABASE_URL`/`DATABASE_URL`).
`ruff check .` clean; `ruff format --check .` clean; frontend `npm run build`
+ `npx tsc --noEmit` clean; CI matrix configured to reproduce all of the above.