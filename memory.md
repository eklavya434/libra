# Project Libra — Persistent Engineering Memory

## 1. Project Overview & Directives
- **Dual-Track Objective**:
  1. ChatGPT-like conversational AI assistant with multi-provider routing, tool calling, and RAG.
  2. First-principles educational LLM laboratory (tokenizer, attention, RoPE, training engine, evaluation).
- **Core Directives**:
  - **Learning > Speed**: Beginner-friendly explanations (WHAT, WHY, HOW, TEST, NEXT).
  - **Zero-Cost ($0 / ₹0)**: Local-first development; no paid APIs or cloud billing.
  - **Hardware Budget**: Intel Core i5-12450H CPU (8C/12T), 16GB RAM, CPU-only. Full debug cycles < 15 minutes.
  - **Storage Quota**: Strict 15 GB ceiling across models, data, checkpoints, and caches.
  - **No Docker Before Phase 34**: Lightweight local `.venv` + Node.js.
  - **Inference Reality**: Ollama is the default CPU engine; vLLM is documented as future GPU-only (not installed).
  - **The Comprehension Gate**: Successfully passed between Phase 6 and Phase 7 (100% score on tokens, attention, training step, and checkpoints).

---

## 2. Phase-by-Phase Progress & Milestone Status

| Phase | Description | Status | Verification & Artifacts |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Foundation & Repository Setup | ✅ COMPLETE | FastAPI, Next.js, Hardware detection, Git remote (`e43a493`) |
| **Phase 1** | Educational Tiny LLM | ✅ COMPLETE | 477K decoder-only transformer, 19.6s training run (`fa23f83`) |
| **Phase 2** | Educational & HF Tokenizer | ✅ COMPLETE | BPE from first principles, ByteLevel BPE, 37.9% compression (`cc5142c`) |
| **Phase 3** | Data Pipeline & Quota Guard | ✅ COMPLETE | Unicode cleaner, SHA-256 deduplicator, binary uint16 sharder (`0cd8310`) |
| **Phase 4** | Modern Transformer | ✅ COMPLETE | RoPE, RMSNorm, SwiGLU, Weight Tying (`9fbcb88`) |
| **Phase 5** | Production Training Engine | ✅ COMPLETE | Cosine warmup, gradient accum, PPL, checkpoint resume (`a7a2771`) |
| **Phase 6** | Evaluation & Benchmarking | ✅ COMPLETE | Perplexity, multiple-choice log-likelihood, 17 domain probes (`f818daf`) |
| **Comprehension Gate** | 4-Pillar Milestone Gate | ✅ PASSED | 4/4 correct on tokens, attention, training steps, checkpoints |
| **Phase 7** | Model Registry & Hardware Sizing | ✅ COMPLETE | 16 models categorized, RAM formula, CPU compatibility checks (`ae3819c`) |
| **Phase 8** | Local Inference Engine | ✅ COMPLETE | Ollama adapter, Hugging Face adapter, local PyTorch server, SSE streaming (`66d83a8`) |
| **Phase 9** | External Provider Adapters | ✅ COMPLETE | OpenAI, Gemini, Claude, DeepSeek, Groq, OpenRouter, cost tracker, error normalization (`a3d7d96`) |
| **Phase 10** | Model Comparison Arena | ✅ COMPLETE | Side-by-side benchmarking, TTFT, token velocity, cost ranking, POST /api/v1/arena/compare (`67268a0`) |
| **Phase 11** | Real Libra Chat UI | ✅ COMPLETE | Next.js frontend, live SSE streaming, dynamic model picker, sampling modal, Arena UI (`bb2be8b`) |
| **Phase 12** | Multi-Turn Conversation Memory | ✅ COMPLETE | SQLite WAL storage, message cascade, ContextWindowManager sliding window, session CRUD |
| **Phase 13** | RAG: Vector Retrieval & Chunking | ⏳ NEXT | First-principles vector embeddings, cosine similarity, document chunking |

---

## 3. Phase 12 Ecosystem & Memory Status

### A. Phase 12 Outcome: Multi-Turn Conversation Memory & Context Management
- **Storage Engine**: `packages/core/memory/sqlite_store.py` (`SQLiteConversationStore`) with Write-Ahead Logging (WAL), foreign key cascade deletion, and auto-titling from initial user prompts.
- **Context Management**: `packages/core/memory/context_manager.py` (`ContextWindowManager`) prevents model context overflow by calculating token budgets and performing sliding-window truncation while preserving system prompts.
- **Backend Endpoints**: `apps/backend/api/v1/endpoints/conversations.py` (`/api/v1/conversations`) provides RESTful session CRUD.
- **Chat Endpoint Integration**: `apps/backend/api/v1/endpoints/chat.py` accepts `conversation_id`, auto-appends user turns, dynamically trims context, and persists streamed responses upon completion.
- **Frontend State**: `apps/frontend/src/components/Sidebar.tsx` renders dynamic conversations list with deletion; `ChatArea.tsx` hydrates previous session messages.
- **Interactive Script**: `scripts/run_phase12_memory_demo.py` verifies end-to-end memory accumulation, auto-titling, and context truncation.

### B. Ollama & Local Inference Status
- **Ollama Adapter**: `packages/providers/ollama.py` ready for local runtime.
- **Hugging Face Adapter**: `packages/providers/huggingface.py` running on CPU.
- **PyTorch Lab Checkpoint**: Serving `checkpoints/best_engine_model.pt` at 349.5 tok/s on CPU.
- **vLLM Status**: Not installed (Directive 6 enforced).

---

## 4. Resource Usage & Storage Quota Audit

- **Venv Size**: ~855 MB
- **Frontend node_modules**: ~281 MB
- **Models & Checkpoints**: 20.08 MB
- **Total Workspace Footprint**: **1,193.75 MB** (~1.19 GB)
- **15 GB Quota Limit**: 15,360.00 MB
- **Remaining Storage Quota**: **14,166.25 MB** (92.23% free)
- **Total Cost**: **$0 / ₹0** (100% free offline development)
- **Active Git Branch**: `main` synced with `https://github.com/eklavya434/libra.git`
- **Pytest Status**: **91 passed, 0 failed** (in 15.69s)



