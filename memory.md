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
| **Phase 14** | Advanced RAG: Hybrid Search & Re-ranking | ✅ COMPLETE | Okapi BM25 sparse index, Reciprocal Rank Fusion (RRF), multi-factor re-ranking, chunk deduplication, Hybrid UI |
| **Phase 15** | Real-Time SSE Streaming & Markdown Rendering | ⏳ NEXT | Markdown streaming parsers, code block syntax highlighting, copy-to-clipboard, token metrics UI |

---

## 3. Phase 14 Ecosystem & Advanced Hybrid RAG Status

### A. Phase 14 Outcome: Hybrid Search (BM25 + Dense), RRF, Re-Ranking & Deduplication
- **Okapi BM25 Index**: `packages/rag/bm25.py` (`BM25Index`) implements Robertson-Spärck Jones IDF and saturation/length-normalized scoring ($k_1=1.5, b=0.75$) with lightweight suffix stemming.
- **Search Fusion**: `packages/rag/fusion.py` provides Reciprocal Rank Fusion (`reciprocal_rank_fusion`, $k=60$) to bridge dense cosine similarity and sparse BM25 scores without distribution mismatch, along with normalized weighted linear interpolation.
- **Multi-Factor Re-Ranking**: `packages/rag/reranker.py` (`HeuristicReRanker`) calculates exact phrase bonuses, query keyword coverage ratios, and term proximity spans on candidate chunks.
- **Chunk Deduplication & MMR**: `packages/rag/deduplication.py` (`ChunkDeduplicator`) applies Jaccard threshold filtering ($J \ge 0.70$) and Maximal Marginal Relevance (MMR) diversification to eliminate redundant overlapping windows.
- **Unified Hybrid Retriever**: `packages/rag/hybrid.py` (`HybridRetriever`) orchestrates two-stage retrieval across dense `InMemoryVectorStore` and sparse `BM25Index`.
- **Backend & Chat Integration**: `apps/backend/api/v1/endpoints/rag.py` (`/api/v1/rag/query`) supports `mode="hybrid"|"dense"|"bm25"`, RRF, reranking, and deduplication; `apps/backend/api/v1/endpoints/chat.py` activates hybrid search for factual chat grounding.
- **Frontend UI**: `apps/frontend/src/components/KnowledgeBaseView.tsx` features interactive mode toggles (Hybrid, Dense, BM25), reranking and deduplication checkboxes, and score breakdown pills.
- **Educational Guide & Demo**: `docs/educational/PHASE_14_HYBRID_SEARCH_AND_RERANKING.md` and `scripts/run_phase14_hybrid_rag_demo.py`.

### B. Vector Database Architectural Decision: pgvector vs. Qdrant
- **Evaluation & Choice**:
  - **pgvector**: Requires a live PostgreSQL database server process or Docker container, which introduces heavy memory overhead (~200MB+ idle) and violates Directive 5 (*No Docker Before Phase 34*). Since Libra already uses lightweight SQLite WAL for relational memory, introducing PostgreSQL would add unnecessary operational friction on a 16GB CPU machine.
  - **Qdrant (Selected Target)**: Chosen as the dedicated vector database target because:
    1. **Embedded Mode**: `qdrant-client` supports embedded local file-based storage (`path="./data/qdrant"`) and in-memory execution with zero Docker dependencies and zero external background daemons.
    2. **Rust & SIMD Optimization**: Ultra-fast CPU execution with AVX2/AVX-512 vector acceleration on Intel Core i5-12450H.
    3. **Native Hybrid Search**: Out-of-the-box support for sparse vectors and dense-sparse hybrid fusion, perfectly matching Phase 14 requirements.
- **Current Phase 14 Implementation**: For educational transparency (Directive 1) and zero external dependencies, Phase 14 implements `InMemoryVectorStore` + `BM25Index` using vectorized NumPy dot products and inverted indices, achieving sub-millisecond CPU retrieval. An abstract base interface allows seamless transition to embedded Qdrant when scaling persistent collections.
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
- **Total Workspace Footprint**: **1,198.95 MB** (~1.20 GB)
- **15 GB Quota Limit**: 15,360.00 MB
- **Remaining Storage Quota**: **14,161.05 MB** (92.19% free)
- **Total Cost**: **$0 / ₹0** (100% free offline development)
- **Active Git Branch**: `main` synced with `https://github.com/eklavya434/libra.git`
- **Pytest Status**: **123 passed, 0 failed** (in 31.86s)





