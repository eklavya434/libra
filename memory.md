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
| **Phase 13** | RAG: Vector Retrieval & Chunking | ✅ COMPLETE | First-principles vector embeddings, cosine similarity, recursive chunker, in-memory store, Knowledge Base UI |
| **Phase 14** | Advanced RAG: Hybrid Search & Re-ranking | ⏳ NEXT | BM25 sparse + dense hybrid search, Reciprocal Rank Fusion (RRF), re-ranking, chunk deduplication |

---

## 3. Phase 13 Ecosystem & Knowledge Base Status

### A. Phase 13 Outcome: RAG Vector Retrieval, Chunking & In-Memory Store
- **Document Chunking**: `packages/rag/chunking.py` (`RecursiveCharacterChunker`) hierarchically splits text via natural boundaries (`\n\n`, `\n`, `. `, ` `, `""`) preserving semantic units with configurable token chunk size and sliding overlap.
- **Dense Embeddings**: `packages/rag/embeddings.py` (`EducationalDenseEmbedder`, $D=128$, L2 unit-normalized n-gram hashing projection) for zero-dependency CPU embedding, alongside `OllamaEmbeddingProvider` for local neural embeddings.
- **Vector Store**: `packages/rag/vector_store.py` (`InMemoryVectorStore`) performs sub-millisecond CPU cosine similarity search via vectorized NumPy matrix-vector multiplication without external vector DB dependencies.
- **Synthesizer**: `packages/rag/synthesizer.py` (`RAGPromptSynthesizer`) constructs grounded prompts with explicit source citation tags.
- **REST Endpoints**: `apps/backend/api/v1/endpoints/rag.py` provides `/api/v1/rag/documents` (ingest, list, delete) and `/api/v1/rag/query` (semantic search).
- **Chat Grounding**: `apps/backend/api/v1/endpoints/chat.py` integrates optional RAG grounding (`use_rag: bool`), retrieving top-$k$ relevant chunks and injecting knowledge context.
- **Frontend UI**: `apps/frontend/src/components/KnowledgeBaseView.tsx` provides full document management, chunk inspection, and semantic query testing; `ChatArea.tsx` includes real-time RAG grounding toggle.
- **Educational Guide**: `docs/educational/PHASE_13_RAG_PIPELINE.md`.
- **Demo Script**: `scripts/run_phase13_rag_demo.py`.

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
- **Total Workspace Footprint**: **1,197.13 MB** (~1.20 GB)
- **15 GB Quota Limit**: 15,360.00 MB
- **Remaining Storage Quota**: **14,162.87 MB** (92.21% free)
- **Total Cost**: **$0 / ₹0** (100% free offline development)
- **Active Git Branch**: `main` synced with `https://github.com/eklavya434/libra.git`
- **Pytest Status**: **108 passed, 0 failed** (in 18.22s)




