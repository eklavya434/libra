# Educational Guide: Phase 12 - Multi-Turn Conversation Memory & Context Window Management

Welcome to **Phase 12** of **Libra**!

In this phase, we transformed Project Libra from an ephemeral, single-turn completion tool into a stateful, multi-turn conversational AI system. We designed a zero-dependency **SQLite conversation store** with Write-Ahead Logging (WAL) and implemented a first-principles **Context Window Manager** that manages finite model attention budgets through sliding-window truncation while strictly preserving the system prompt.

---

## 1. WHAT Was Built?

1. **SQLite Conversation Store (`packages/core/memory/sqlite_store.py`)**:
   - Zero-dependency implementation using Python's standard library `sqlite3`.
   - **WAL Mode (Write-Ahead Logging)**: Enables concurrent reads while writing, preventing lock contention.
   - **Cascading Foreign Key Integrity**: Deleting a conversation session automatically cascades and cleans up all associated messages.
   - **Auto-Titling**: New conversations automatically derive their title from the first prompt sent by the user (up to 40 characters).

2. **Context Window Manager (`packages/core/memory/context_manager.py`)**:
   - Manages token allocation between input prompt history and generated response headroom.
   - **System Prompt Preservation**: Treats the system prompt as an invariant anchor, guaranteeing persona and safety instructions are never evicted.
   - **Sliding-Window Dialogue Truncation**: Walks backwards from the newest user/assistant turns, discarding older messages only when the budget ceiling is exceeded.

3. **FastAPI Conversation Endpoints (`apps/backend/api/v1/endpoints/conversations.py`)**:
   - `GET /api/v1/conversations`: Paginated list of sessions sorted by `updated_at DESC`.
   - `POST /api/v1/conversations`: Initializes new session with custom or default model and system prompt.
   - `GET /api/v1/conversations/{id}`: Retrieves conversation metadata and chronological message history.
   - `PATCH /api/v1/conversations/{id}`: Renames title, switches active model, or modifies system prompt.
   - `DELETE /api/v1/conversations/{id}`: Cascading deletion.
   - `POST /api/v1/chat/completions`: Seamlessly takes `conversation_id`, auto-appends user turns, budgets context tokens, and persists assistant stream chunks upon completion.

4. **Frontend Integration (`apps/frontend/`)**:
   - `Sidebar.tsx`: Real-time session history drawer with active session selection, hover delete action, and "New Conversation" quick-start.
   - `ChatArea.tsx`: Dynamic message history hydration from the persistent store and multi-turn state synchronization.

5. **Interactive Laboratory Demo (`scripts/run_phase12_memory_demo.py`)**:
   - Command-line walkthrough demonstrating multi-turn history accumulation, auto-titling, sliding-window truncation, and database cascading cleanup.

---

## 2. WHY Context Window Management Is Essential

Large Language Models do not possess infinite memory. Every transformer architecture has a fixed **maximum context length** ($L_{\max}$):
- Educational Tiny Transformer: 256 or 512 tokens.
- Small Local CPU Models (e.g. Llama 3.2 1B / 3B): 2,048 to 8,192 tokens.

### The Two Major Failure Modes:
1. **Dimension Mismatch / Attention Crash**:
   If prompt tokens $N_{\text{prompt}} > L_{\max}$, the attention matrix $Q K^T$ exceeds the physical positional encoding matrix (RoPE / absolute embeddings), causing PyTorch to raise runtime shape mismatch exceptions.
2. **Output Starvation**:
   Even if the prompt fits within $L_{\max}$, if $N_{\text{prompt}} = L_{\max}$, there are **zero tokens remaining for generation** ($N_{\text{completion}} = 0$). The model immediately outputs an End-of-Sequence (`<|eot_id|>`) token without answering!

---

## 3. HOW The Context Budgeting Math Operates

For any given inference request:
$$B_{\text{input}} = L_{\max} - N_{\text{reserved}}$$

Where:
- $L_{\max}$: The maximum context capacity of the target model (e.g. 2048).
- $N_{\text{reserved}}$: The maximum tokens reserved for the model's generated answer (`max_tokens`, e.g. 512).
- $B_{\text{input}}$: The maximum allowable token budget for prompt history.

### The Allocation Algorithm:
1. Allocate $N_{\text{system}}$ for the system prompt:
   $$B_{\text{dialog}} = B_{\text{input}} - N_{\text{system}}$$
2. Iterate through turns in reverse chronological order ($t = T, T-1, \dots, 1$):
   $$\sum_{i=k}^T N_{\text{turn}_i} \le B_{\text{dialog}}$$
3. Retain turns $k \dots T$, discarding turns $1 \dots k-1$.
4. Re-assemble final prompt:
   $$\text{Prompt} = [\text{System}] \cup [\text{Turn}_k, \dots, \text{Turn}_T]$$

---

## 4. Verification & Test Coverage

1. **Memory & API Pytest Suite**:
   ```bash
   .\.venv\Scripts\pytest -v tests/memory/ tests/api/test_conversations_endpoint.py
   ```
   *Result*: **16 passed** in 5.43s.

2. **Full Repository Pytest Suite**:
   ```bash
   .\.venv\Scripts\pytest -v tests/
   ```
   *Result*: **91 passed, 0 failed** in 15.69s.

3. **Frontend Production Build**:
   ```bash
   cd apps/frontend
   npm run build
   ```
   *Result*: Successfully compiled and statically optimized with 0 TypeScript errors.

4. **Zero-Cost & Quota Compliance**:
   - Uses built-in `sqlite3` (no new pip packages, no external database daemons).
   - Entire workspace footprint remains ~1.19 GB, comfortably within the 15 GB quota.
