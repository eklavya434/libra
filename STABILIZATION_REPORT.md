# Project Libra — Phase 50 Full Stabilization, Debugging & Repair Report

**Date:** September 10, 2026  
**Status:** **STABLE & VERIFIED (PASS)**  
**Environment:** Windows 10/11 x64, Intel Core i5-12450H CPU, 16GB RAM, Python 3.13 (.venv), Node.js v20 (Next.js 14)

---

## 1. Executive Summary

During Phase 50 capstone validation, a critical defect was detected: user messages sent through the chat interface could fail to return an answer, leaving empty assistant bubbles or terminating with connection errors. 

Through deep root-cause diagnostics, **three interrelated failure modes** were discovered and resolved:
1. **Windows Anycast IPv6 DNS Blackhole**: On Windows systems with virtual or dormant network interfaces (e.g. Tailscale, Hyper-V/WSL, virtual LANs), Python's default `socket.getaddrinfo(family=0)` systematically queried dead `fec0:0:0:ffff::1` IPv6 DNS servers first. This created a **41-second blocking hang** before falling back to IPv4 DNS. Because `httpx` default client timeouts are 30.0s, requests to external cloud endpoints (Google Generative AI / Gemini API) aborted with `All connection attempts failed`.
2. **Missing Session ID Ephemeral Drop**: When the frontend had no active session ID (e.g., initial clean state), `conversation_id` was sent as empty string or undefined. The backend chat endpoint treated this as ephemeral, omitting SQLite WAL persistence and dropping the conversation history on refresh.
3. **0-Token & Welcome Banner Dialog Rejection**:
   - In the frontend SSE parser, streams receiving 0 delta content tokens silently called `onComplete`, rendering an invisible, empty message bubble rather than an informative error status.
   - The UI welcome banner was prepended to the API dialog history as an assistant message. Downstream providers with strict turn-taking rules (Google Gemini API, Anthropic) rejected dialogs where the first turn was not a user turn (`HTTP 400: First content should be with role 'user'`).

All root causes have been resolved, verified with live end-to-end multi-turn streaming tests against both local mock and external Gemini 2.5 Flash endpoints, and backed by new automated regression tests (`tests/api/test_chat_stabilization.py`).

---

## 2. Root Cause Analysis (RCA)

| Root Cause ID | Fault Description | Impact | Resolution |
| :--- | :--- | :--- | :--- |
| **RCA-01** | `getaddrinfo` IPv6 DNS hang on Windows virtual adapters (`fec0::` site-local servers). | 41-second latency per DNS lookup; `httpx` timeouts causing `"All connection attempts failed"`. | Created `packages/core/network.py` with `enable_ipv4_preference()`. Intercepts `socket.getaddrinfo` to resolve standard domain names via `AF_INET` in **42 ms** (1000x speedup). |
| **RCA-02** | Gemini API candidate fallback list contained nonexistent models (`gemini-3.5-flash`). | 404 error cascades if fallback models were triggered. | Cleaned candidate models in `packages/providers/gemini.py` to only valid endpoints (`gemini-2.5-flash`, `gemini-2.5-pro`). |
| **RCA-03** | Gemini message converter lacked leading assistant turn stripping. | `HTTP 400: First content should be with role 'user'` if welcome message or greeting was first in history. | Added `while gemini_contents and gemini_contents[0]["role"] == "model": gemini_contents.pop(0)`. |
| **RCA-04** | Chat endpoint omitted SQLite WAL persistence if `conversation_id` was empty. | Chats disappeared upon page reload or navigation; multi-turn context was lost. | `apps/backend/api/v1/endpoints/chat.py` now auto-generates a UUID session (`conv-<uuid>`), creates the DB record, stores turns, and sends `X-Conversation-Id`. |
| **RCA-05** | Silent fallback to `mock-provider` when cloud models requested without API keys. | Users selecting OpenAI/Gemini/Claude saw unexpected mock responses rather than actionable auth guidance. | Hardened `packages/providers/router.py` to raise explicit `ValueError` with clear setup instructions. |
| **RCA-06** | Frontend 0-token stream completion & error swallow. | Invisible empty bubbles rendered; errors occurring after partial token yield were masked by `msg.content || err.message`. | Updated `api.ts` to trigger `onError` on 0-token streams, and `ChatArea.tsx` to append streaming errors. |

---

## 3. Detailed Code Modifications

### A. Core Network Layer (`packages/core/network.py`)
- Created automated IPv4 DNS preference mechanism.
- Auto-executes upon package import across `apps/backend/main.py` and `packages/providers/`.
- Resolves domain queries in **42.9 ms** compared to the prior **41.2 seconds**.

### B. Unified Provider Layer (`packages/providers/router.py` & `packages/providers/gemini.py`)
- **Router**: Added explicit routing blocks for `deepseek` and `openrouter`. Replaced silent mock fallbacks with informative exceptions.
- **Gemini Adapter**:
  - Aliased `gemini`, `gemini-flash`, `gemini-1.5-flash` to `gemini-2.5-flash`.
  - Aliased `gemini-pro`, `gemini-1.5-pro` to `gemini-2.5-pro`.
  - Enforced valid dialog structure (first turn is `user`, last turn is `user`).

### C. Backend API Endpoint (`apps/backend/api/v1/endpoints/chat.py`)
- Guaranteed persistent conversation storage:
  - If `request.conversation_id` is missing or empty, auto-generates `conv-<hex>`.
  - Automatically derives the session title from the user prompt.
  - Returns `X-Conversation-Id` HTTP header and initial SSE metadata chunk.
  - Persists both user prompts and accumulated streaming assistant responses in SQLite WAL.

### D. Frontend UX Layer (`apps/frontend/src/lib/api.ts` & `ChatArea.tsx`)
- **`streamChat`**:
  - Captures `X-Conversation-Id` header and `conversation_id` chunk, notifying caller via `onConversationId`.
  - Detects 0-token completion and raises an explicit error.
- **`ChatArea.tsx`**:
  - Filters client welcome banner (`id === 'welcome'`) and error cards out of dialog history sent to the backend.
  - Links session updates to `page.tsx` via `onSelectConversation`.
  - Displays streaming errors even if partial tokens were previously accumulated.

---

## 4. Verification & Testing

### 1. Live Streaming Verification
- **Local Mock Model (`libra-mock-v1`)**:
  - Status: `200 OK`, `Content-Type: text/event-stream`
  - Output: `[MockStream] Hello from Libra! You said: 'Hello from test! Tell me what 3+3 is.'.`
- **Cloud Frontier Model (`gemini-2.5-flash`)**:
  - Status: `200 OK`, `Content-Type: text/event-stream`
  - Latency: First token received in < 1.2 seconds.
  - Output: `Hello there from test! It's nice to meet you! The answer to 3 + 3 is 6.`

### 2. Multi-Turn SQLite WAL Persistence Verification
- Executed 2-turn dialog:
  - Turn 1: *"My secret code is PHOENIX-99."* -> Assistant acknowledged.
  - Turn 2: *"What was my secret code?"* -> Assistant recalled: *"Your secret code was PHOENIX-99."*
  - SQLite WAL Store: Verified 4+ entries recorded in `conversations` and `messages` tables.

### 3. Automated Regression Tests (`tests/api/test_chat_stabilization.py`)
- `test_chat_auto_generates_and_persists_conversation`: **PASSED**
- `test_chat_streaming_includes_conversation_metadata`: **PASSED**
- `test_gemini_model_normalization`: **PASSED**
- `test_router_raises_explicit_error_for_unconfigured_cloud_provider`: **PASSED**
- Full chat suite (`tests/api/test_chat_endpoint.py`): **PASSED** (4/4 tests)
- TypeScript compile (`npx tsc --noEmit`): **PASSED** (0 errors)
- Ruff lint check: **PASSED** (0 errors)

---

## 5. System Runbook

### To run the Libra Stack:
1. **Backend**:
   ```powershell
   .\.venv\Scripts\uvicorn apps.backend.main:app --host 127.0.0.1 --port 8000
   ```
2. **Frontend**:
   ```powershell
   cd apps/frontend
   npm run dev
   ```
3. **Run Regression Tests**:
   ```powershell
   .\.venv\Scripts\pytest -v tests/api/test_chat_stabilization.py tests/api/test_chat_endpoint.py
   ```\n