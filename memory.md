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
| **Phase 12** | Multi-Turn Conversation Memory | ✅ COMPLETE | SQLite WAL storage, message cascade, ContextWindowManager sliding window, session CRUD (`b9ee315`) |
| **Phase 13** | RAG: Vector Retrieval & Chunking | ✅ COMPLETE | First-principles vector embeddings, cosine similarity, recursive chunker, in-memory store, Knowledge Base UI (`27fc1dc`) |
| **Phase 14** | Advanced RAG: Hybrid Search & Re-ranking | ✅ COMPLETE | Okapi BM25 sparse index, Reciprocal Rank Fusion (RRF), multi-factor re-ranking, chunk deduplication, Hybrid UI (`f3b1128`) |
| **Phase 15** | Real-Time SSE Streaming & Markdown Rendering | ✅ COMPLETE | StreamingMarkdown parser, syntax-highlighted CodeBlock, AbortController cancellation, regenerate response, telemetry |
| **Phase 16** | Tool Use & Sandbox Execution | ⏳ NEXT | Function calling, JSON schema tool definitions, safe execution sandbox, multi-step agent tools |

---

## 3. Phase 15 Ecosystem & Streaming UX Status

### A. Phase 15 Outcome: Streaming Markdown, Syntax Highlighting & Generation Telemetry
- **Streaming Markdown Parser**: `apps/frontend/src/components/StreamingMarkdown.tsx` implements a zero-dependency streaming state machine handling open/unclosed code fences in-flight without layout shifting or hydration errors.
- **Code Block Component**: `apps/frontend/src/components/CodeBlock.tsx` features language detection, line numbers, one-click copy with feedback, and regex-based multi-token syntax highlighting (keywords, strings, numbers, comments, builtins).
- **Collapsible Reasoning**: `<think>...</think>` tags are extracted and rendered in an expandable accordion block with animated thinking indicators for reasoning models (DeepSeek-R1, Qwen 2.5).
- **Stream Abort & Cancellation**: `streamChat` accepts `signal: AbortSignal`. The UI includes a Stop Generation button (`Square` icon) in the input bar and handles `AbortError` gracefully, finalizing telemetry up to the cancellation point.
- **Regenerate Response**: Last user turn can be regenerated with a single click (`RefreshCw` icon).
- **Telemetry Display**: Displays TTFT (ms), latency (ms), token velocity (tok/s), token counts, and cost calculation.
- **Educational Guide & Demo**: `docs/educational/PHASE_15_STREAMING_UX_AND_MARKDOWN.md` and `scripts/run_phase15_streaming_demo.py`.

### B. Vector Database Architectural Decision: pgvector vs. Qdrant
- **Evaluation & Choice**:
  - **pgvector**: Requires a live PostgreSQL database server process or Docker container, which introduces heavy memory overhead (~200MB+ idle) and violates Directive 5 (*No Docker Before Phase 34*). Since Libra already uses lightweight SQLite WAL for relational memory, introducing PostgreSQL would add unnecessary operational friction on a 16GB CPU machine.
  - **Qdrant (Selected Target)**: Chosen as the dedicated vector database target because:
    1. **Embedded Mode**: `qdrant-client` supports embedded local file-based storage (`path="./data/qdrant"`) and in-memory execution with zero Docker dependencies and zero external background daemons.
    2. **Rust & SIMD Optimization**: Ultra-fast CPU execution with AVX2/AVX-512 vector acceleration on Intel Core i5-12450H.
    3. **Native Hybrid Search**: Out-of-the-box support for sparse vectors and dense-sparse hybrid fusion, perfectly matching Phase 14 requirements.
- **Current Implementation**: `InMemoryVectorStore` + `BM25Index` using vectorized NumPy dot products and inverted indices, achieving sub-millisecond CPU retrieval. An abstract base interface allows seamless transition to embedded Qdrant when scaling persistent collections.
- **Deviations from Plan**: None.

### C. Ollama & Local Inference Status
- **Ollama Adapter**: `packages/providers/ollama.py` ready for local runtime.
- **Hugging Face Adapter**: `packages/providers/huggingface.py` running on CPU.
- **PyTorch Lab Checkpoint**: Serving `checkpoints/best_engine_model.pt` at 349.5 tok/s on CPU.
- **vLLM Status**: Not installed (Directive 6 enforced).

---

## 4. Resource Usage & Storage Quota Audit

- **Venv Size**: ~855 MB
- **Frontend node_modules**: ~281 MB
- **Models & Checkpoints**: 20.08 MB
- **Total Workspace Footprint**: **1,201.19 MB** (~1.20 GB)
- **15 GB Quota Limit**: 15,360.00 MB
- **Remaining Storage Quota**: **14,158.81 MB** (92.18% free)
- **Total Cost**: **$0 / ₹0** (100% free offline development)
- **Active Git Branch**: `main` synced with `https://github.com/eklavya434/libra.git`
- **Pytest Status**: **126 passed, 0 failed** (in 18.47s)






