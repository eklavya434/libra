# Libra Production Deployment Guide

Zero-cost public deployment for the Libra AI Laboratory & Assistant.

## Architecture (zero-cost, CPU-first)

```
                        Browser (laptop / phone)
                                 |
                                 | HTTPS + X-Libra-Session header (guest identity)
                                 v
                     +---------------------------+
                     |  Frontend (Next.js 14,     |   Render free web service
                     |  standalone Node server)   |   (Dockerfile.frontend)
                     +---------------------------+
                                 |
                                 | NEXT_PUBLIC_API_URL (absolute backend URL)
                                 v
                     +---------------------------+
                     |  Backend (FastAPI, non-root,)|  Render free web service
                     |  CPU-first, rate limited    |  (Dockerfile.backend)
                     +---------------------------+
                      |  |  |  |
                      |  |  |  +--> Providers: Gemini / NVIDIA (free tier, keys
                      |  |  |        are secrets, never stored in-repo)
                      |  |  |
                      |  |  +-----> SQLite (on Render disk mount /app/data) —
                      |  |            conversations + guest sessions
                      |  |
                      |  +---------> RAG index (in-memory; re-seeded by document
                      |               uploads; persisted on disk mount)
                      |
                      +------------> Ollama: NOT deployed on the free platform.
                                     Local models remain for the machine-learning
                                     lab runs on YOUR PC (CPU budget < 15 min).
```

Total recurring cost: **$0**. Free-tier sleeps after inactivity; cold-starts ~1 min.

## Why this shape

- **One account, no billing**: Render free tier hosts both Docker services and
  gives a free `*.onrender.com` subdomain + TLS. No CLI needed.
- **Guest identity instead of accounts**: each browser gets an opaque session
  token (`X-Libra-Session`, SHA-256 hashed at rest) so conversations are
  isolated per visitor without login friction or email plumbing (mission §16).
- **Code execution stays OFF publicly**: `LIBRA_PUBLIC_CODE_EXEC=false` (default)
  makes the notebook executor return 503. Enabling it requires an explicit
  operator decision and a sandboxed subprocess runner (see SECURITY.md).
- **Honest limits**: single-replica SQLite on a small disk. Suitable for a
  demo/lab (<10 concurrent users). Scaling out is documented below.

## Step-by-step (owner actions required)

### 1. Set up the GitHub repo + CI (already done)
`git origin` is `https://github.com/eklavya434/libra`, branch `main`, CI green.

### 2. Create the Render services (manual, browser signup)
1. Sign up at https://render.com (free, no card).
2. New -> **Blueprint** -> connect your GitHub org and select `libra`.
3. Render reads `render.yaml` and provisions:
   - `libra-backend` (Docker, health `/healthz`)
   - `libra-frontend` (Docker)
4. Create the two screens first, then revisit the Dashboard:
   - Services -> libra-backend -> Environment:
     - set `GEMINI_API_KEY` and `NVIDIA_API_KEY` (your real values, stored as
       Render secrets, never in git), or leave blank to run 100% offline/mock.
     - set `CORS_ORIGINS` to the frontend FQDN
       (default `["https://libra-frontend.onrender.com"]`).
   - Services -> libra-frontend -> Environment:
     - learn the backend FQDN, e.g. `https://libra-backend.onrender.com`
     - set `NEXT_PUBLIC_API_URL=https://libra-backend.onrender.com`.
   - Both services auto-deploy on every push to `main` (Blueprinted).

### 3. Verify (before touching DNS)
1. Open `https://libra-frontend.onrender.com` in a phone browser.
2. Send a chat -> expect streaming with `gemini-2.5-flash` or `libra-mock-v1`.
3. Refresh (sessions persist) and confirm the sidebar still lists the round.
4. Open a second private window (or `libra-frontend.onrender.com` on another
   device) -> conversation list is EMPTY there (isolation working).
5. Backend smoke checks:
   - `curl https://libra-backend.onrender.com/healthz` -> `{"status":"alive"}`
   - `curl https://libra-backend.onrender.com/api/v1/models` -> 41 models,
     `available: true/false` with reasons.
   - Notebook execution:
     `curl -X POST https://libra-backend.onrender.com/api/v1/notebook/sessions/x/execute -H 'Content-Type: application/json' -d '{"code":"print(1)"}'`
     -> `503` (public-safe gate).

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
- REST smoke tests through the public URLs (chat mock + Gemini, models, health).
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
SQLite lives on the Render disk. Download `data/conversations.db`
periodically, or add an optional cron that dumps it to an R2/Drive destination
(owner's storage keys required; not in the default flow).

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