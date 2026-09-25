# Libra Production Deployment Guide

Zero-cost public deployment for the Libra AI Laboratory & Assistant.

## Architecture (zero-cost, CPU-first)

```
                        Browser (laptop / phone)
                                 |
                                 | single HTTPS origin (the ONE recruiter URL)
                                 v
                     +---------------------------+
                     |  Frontend (Next.js 14)     |   Vercel, free hobby tier
                     |  `/` landing page          |   auto-deploys on push to main
                     |  `/chat` assistant app     |
                     +---------------------------+
                        | same-origin `/api/*` proxy
                        | (vercel.json + next.config rewrites -> no CORS)
                        v
                     +---------------------------+
                     |  Backend (FastAPI, non-root,)|  Render free web service
                     |  CPU-first, rate limited    |  (Dockerfile.backend)
                     +---------------------------+
                      |  |  |  |
                      |  |  |  +--> Providers: Gemini / NVIDIA (free tier, keys
                      |  |  |        are secrets, never stored in-repo)
                      |  |  |
                      |  |  +--> Supabase Postgres (free tier): durable
                      |  |        conversations via PostgresConversationStore
                      |  |        when DATABASE_URL is set (recommended)
                      |  |
                      |  +--> Supabase Storage bucket: uploaded document bytes
                      |
                      +---------> Ollama: NOT deployed on the free platform.
                                   Local models remain for the machine-learning
                                   lab runs on YOUR PC (CPU budget < 15 min).
```

Total recurring cost: **$0**. Vercel hobby + Render free + Supabase free. Free
instances sleep after inactivity and cold-start in ~1 min.

## Why this shape

- **One recruiter URL**: the Vercel origin hosts both the landing page and the
  assistant app. All `/api/*` traffic is proxied same-origin by rewrites, so
  the browser never needs CORS and users never see the backend host.
- **One account, no billing**: Vercel and Render free tiers + Supabase free tier;
  no CLI or tokens required. Owner-only actions are browser signups.
- **Guest identity instead of accounts**: each browser gets an opaque session
  token (`X-Libra-Session`, SHA-256 hashed at rest) so conversations are
  isolated per visitor without login friction or email plumbing (mission §16).
- **Durability without paying**: `DATABASE_URL` → psycopg3
  `PostgresConversationStore` (migration `supabase/migrations/0001_conversation_store.sql`).
  Uploads persist to a Supabase Storage bucket.
- **Code execution stays OFF publicly**: `LIBRA_PUBLIC_CODE_EXEC=false` (default)
  makes the notebook executor return the exact mission message. Enabling it
  requires an operator flag AND a chosen backend (`LIBRA_CODE_SANDBOX=jail`
  recommended, see SECURITY.md); the reference deployment keeps it OFF.
- **Honest limits**: free 0.1 CPU / 512 MB RAM / ephemeral local filesystem.
  Suitable for a demo/lab (<10 concurrent users). Scaling out is documented below.

## Data persistence (free vs paid — read this before deploying)

Render free web services have an **ephemeral filesystem**: anything written
locally (`data/`) is lost on redeploy or restart. Durable data uses external
free services instead:

1. **Durable conversations (recommended)**: create a free Supabase project, run
   the migration, and set `DATABASE_URL` (plus `SUPABASE_URL` /
   `SUPABASE_SERVICE_ROLE_KEY` for uploads) on the backend service. This makes
   guest session chats survive cold-starts and redeploys at $0.
2. **Demo acceptable (fallback)**: leave `DATABASE_URL` unset. The backend
   degrades to SQLite under `data/`, recreated from schema on first request —
   honest about the ephemeral host, fine for a throwaway demo.
3. **Paid plan**: uncomment the `disk:` block in `render.yaml` to persist local
   files across deploys (~$0.25/mo for 1 GB).

## Step-by-step (owner actions required)

### 1. Set up the GitHub repo + CI (already done)
`git origin` is `https://github.com/eklavya434/libra`, branch `main`, CI green.

### 2. Create the services (manual, browser signup)
1. **Frontend (Vercel)**: import the repo at https://vercel.com -> Project
   (framework Next.js, build `npm run build`). Vercel reads `vercel.json`
   (rewrites `/api/*` -> backend) and auto-deploys pushes to `main`.
2. **Backend (Render)**: sign up at https://render.com (free, no card) ->
   New -> **Blueprint** -> connect GitHub and select `libra`. Render reads
   `render.yaml` and provisions `libra-backend` (Docker, health `/healthz`).
3. **Durability (Supabase)**: create a free project, run
   `supabase db push` (or apply `supabase/migrations/`), create the
   `libra-files` storage bucket, then on the Render service set:
   - `GEMINI_API_KEY`, `NVIDIA_API_KEY` (real values, Render secrets).
   - `DATABASE_URL` (durable conversations), `SUPABASE_URL` +
     `SUPABASE_SERVICE_ROLE_KEY` (uploads bucket).
   - Optional Supabase Auth: `LIBRA_SUPABASE_AUTH_ENABLED=true` +
     `SUPABASE_JWT_SECRET` (JWT sessions layered over guests).
4. Both services auto-deploy on every push to `main`.

### 3. Verify (before touching DNS)
1. Open the Vercel FQDN in a phone browser -> landing page -> "Start chatting".
2. Send a chat -> expect streaming through the configured provider
   (`gemini-2.5-flash` or whatever the backend defaults to; never a mock).
3. Refresh (sessions persist) and confirm the sidebar still lists the round.
4. Open a second private window on another device -> conversation list is EMPTY
   there (isolation working).
5. Backend smoke checks:
   - `curl https://libra-backend.onrender.com/healthz` -> `{"status":"alive"}`
   - `curl https://libra-backend.onrender.com/api/v1/models` -> model list with
     `available: true/false` and reasons.
   - Notebook execution:
     `curl -X POST https://libra-backend.onrender.com/api/v1/notebook/sessions/x/execute -H 'Content-Type: application/json' -d '{"code":"print(1)"}'`
     -> `503` with the exact mission message (public-safe gate).

### 4. Custom domain (GitHub Student domain)
You own `libra.<subdomain>.students.github.workers.dev`? There is no such
service; the GitHub Students "domain" is their package, not DNS. Use one of:
- **Render custom domain (recommended)**: a `*.onrender.com` is enough for the
  demo. For a custom apex (`libra.me`), add the custom domain in Render
  (`Services -> libra-frontend -> Settings -> Custom Domain`) and point DNS:
  - `CNAME  www -> libra-frontend.onrender.com`
  - `CNAME  @  -> libra-frontend.onrender.com`   (or ALIAS/ANAME where supported)
- Vercel for frontend + Render for backend: move DNS (or subdomain) to Vercel
  (`cname.vercel-dns.com`), add the frontend there, keep the backend on Render.
Then set `NEXT_PUBLIC_API_URL=https://<backend-host>` and re-deploy.

> Note: whatever DNS host you pick, point only the FRONTEND domain. The backend
> can stay on `*.onrender.com` (no auth depends on it; locality is irrelevant).

### 5. Final validation
Run through the mission checklist:
- Browser on phone + desktop -> responsive, streaming, no console errors.
- REST smoke tests through the public URLs (chat via configured provider, models, health).
- Rate limit: >120 rapid `/api/v1/chat/completions` calls -> 429.
- Secret scan: `GEMINI_API_KEY` never appears in response bodies/headers.
- Cost check: Gemini free tier daily quota; zero other spend.

## Operations runbooks

### Logs
`Services -> libra-backend -> Logs`. Search `request_id` to correlate
`/healthz`-proxy errors (each JSON error carries `X-Request-Id`).

### Restart / redeploy
Render `Deploy -> Deploy latest commit`, or push to `main`. Data persists on the
disk mount (`/app/data`).

### Waking from sleep
Free services sleep after ~15 min idle; first request triggers a cold start
(~1 min). A scheduler (cron in CI) can ping `/healthz` every 10 min to keep it
warm, or accept the latency. `LIBRA_DEPLOY_ENV=render` is set for telemetry.

### Backups
On a paid plan, SQLite lives on the Render disk (`/app/data`) with daily
snapshots. On the free plan the filesystem is ephemeral, so backup by polling
`/api/v1/conversations` (export endpoint) or accept the demo reset behavior.

## Scaling-out path (documented, not built — avoid untested code on $0)
When more than single-replica is needed:
1. `SQLiteConversationStore` gains a Postgres adapter (`owner_id` schema already
   portable; sessions/unique hash translate 1:1). Postgres: Neon/Supabase
   free tier.
2. RAG vectors move to pgvector (or an S3/R2 file store adapter).
3. Frontend/Backend split to two render services is already done; add a
   reverse-proxy (Render proxies HTTPS) if single-domain sharing is preferred.
Defer all three until tokens + free instances exist (RULE: never ship untested
storage switches).

## Related
- `docs/deployment/ENV_INVENTORY.md` — every environment variable.
- `docs/security/SECURITY.md` — threat model and controls.
- `.github/workflows/ci.yml` + `.github/workflows/deploy.yml` — CI/CD.
- `docker-compose.prod.yml` — self-hosted single-vm alternative.