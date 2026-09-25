# Libra Environment Variable Inventory

Every variable the backend, frontend, and deployment surface read, with the
default, source (authoritative file), and behavior. Backend variables are
defined in `apps/backend/core/config.py` (pydantic-settings).

## Backend (`apps/backend`)

| Variable | Default | Meaning |
| :-- | :-- | :-- |
| `LIBRA_ENV` | `development` | Runtime environment tag (`production` in prod). |
| `LIBRA_LEARNING_MODE` | `true` | Keeps `LIBRA_LEARNING_MODE=true` markers for the educational protocol. |
| `LIBRA_HOST` | `127.0.0.1` | Bind host (Docker/Render set `0.0.0.0`). |
| `LIBRA_PORT` | `8000` | Bind port. |
| `LIBRA_LOG_LEVEL` | `INFO` | uvicorn log level. |
| `CORS_ORIGINS` | localhost dev list | JSON list (`["https://front.onrender.com"]`) of allowed browser origins. Empty `[]` only when frontend+backend share one origin. |
| `LIBRA_TRUSTED_PROXIES` | `1` | Trust count for `X-Forwarded-For` in rate limiting (`0` = never trust, for direct exposure). Matches uvicorn `--forwarded-allow-ips`. |
| `LIBRA_SESSION_TTL_DAYS` | `30` | Guest session lifetime before expiry cleanup. Range 1–365. |
| `LIBRA_PUBLIC_CODE_EXEC` | `false` | **Public-safety gate.** When `false`, notebook `/execute` returns 503. Only turn on with a subprocess sandbox. |
| `LIBRA_RATE_LIMIT_PER_MINUTE` | `120` | Token bucket refill (per client IP) on compute endpoints. |
| `LIBRA_RATE_LIMIT_BURST` | `30` | Token bucket burst capacity. |
| `LIBRA_CACHE_DIR` | `./models` | Model cache/artifacts quota root (subject to `MAX_DISK_USAGE_PERCENT`). |
| `LIBRA_DATA_DIR` | `./data` | Persistence root (RAG index, uploaded docs). |
| `MAX_DISK_USAGE_PERCENT` | `90.0` | Hard guard preventing disk-fill. |
| `LIBRA_DB_PATH` | `$LIBRA_DATA_DIR/conversations.db` (package default `data/conversations.db`) | SQLite file for conversations + sessions. Read via `os.getenv`. |
| `OPENAI_API_KEY` | *(blank)* | Optional provider key (free/paid, never committed). |
| `ANTHROPIC_API_KEY` | *(blank)* | Optional provider key. |
| `GEMINI_API_KEY` | *(blank)* | **Real key in your `.env` (dev) / Render secret (prod).** Powers `gemini-2.5-flash`/`-pro`. |
| `GROQ_API_KEY` | *(blank)* | Optional provider key. |
| `DEEPSEEK_API_KEY` | *(blank)* | Optional provider key. |
| `KIMI_API_KEY` | *(blank)* | Optional provider key (Moonshot). |
| `MOONSHOT_API_KEY` | *(blank)* | Alias provider key. |
| `NVIDIA_API_KEY` | *(blank)* | **Real key in your `.env` / Render secret.** Powers the NVIDIA `deepseek-*`/`kimi-*` set. |
| `NVIDIA_DEEPSEEK_API_KEY` | *(blank)* | Legacy split key (kept for compat). |
| `NVIDIA_KIMI_API_KEY` | *(blank)* | Legacy split key (kept for compat). |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local model runtime (your PC only; not deployed). |
| `VLLM_BASE_URL` | `http://localhost:8000` | Reserved GPU runtime (documented, not installed). |
| `LIBRA_DEPLOY_ENV` | *(blank)* | Set to `render` by render.yaml for telemetry/runbook hints. |

## Frontend (`apps/frontend/.env.local`, baked at build time)

| Variable | Default | Meaning |
| :-- | :-- | :-- |
| `NEXT_PUBLIC_API_URL` | *(unset → Next dev proxy / same origin)* | Absolute base URL the **browser** calls for `/api/v1/*`. In Render set to `https://libra-backend.onrender.com`. |

## Session identity (no env needed — designed-in)

- Header `X-Libra-Session` carries an opaque token per browser.
- Only SHA-256 hashes are stored in `sessions`; `conversations.owner_id` is the
  session id. No cookies, no PII, no passwords.
- Mints only via `POST /api/v1/auth/session`; missing header → shared `public`
  scope kept for legacy/curl access (isolation applies to token holders).

## Secrets policy

- `GEMINI_API_KEY` / `NVIDIA_API_KEY` live ONLY in your untracked `.env` (dev)
  and Render Environment (prod). They are gitignored, excluded from responses,
  and never echoed by any endpoint (verified by test).
- `.env.example` documents keys with blank values only.