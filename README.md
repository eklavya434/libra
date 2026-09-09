# ♎ Libra: Personal LLM Laboratory & AI Assistant Built from First Principles

> **Project Libra** is an open-source, dual-track educational AI laboratory and ChatGPT-like assistant engineered from mathematical and code first principles. Running 100% locally on consumer-grade CPU hardware, Libra demonstrates that foundation model architectures, inference acceleration, RAG, agents, and alignment can be mastered at zero cost ($0 / ₹0).

---

## 🏆 Project Status: All 36 Phases Complete & Release Verified [PASS]

```
================================================================================
[*] PROJECT LIBRA: MASTER CAPSTONE SYSTEM AUDIT & RELEASE VERIFICATION
================================================================================
Host Hardware: Intel Core i5-12450H CPU (8C/12T) | 16 GB RAM | Windows / Linux
Device Tier:   cpu-light (Zero-Cost Local CPU Execution)
Storage Quota: 875.85 MB Used / 15.00 GB Quota (94.3% Free)
Test Suite:    394+ Tests Passing (100% Pass Rate across all 36 Phases)
Final Status:  RELEASE READY [PASS]
================================================================================
```

---

## 🎯 The Dual-Track Architecture

```
                               PROJECT LIBRA
                                     │
          ┌──────────────────────────┴──────────────────────────┐
          ▼                                                     ▼
    TRACK 1: THE AI PRODUCT                             TRACK 2: THE LLM LAB
    • ChatGPT-like Next.js UI                           • Tokenizer from Scratch (BPE)
    • FastAPI REST & SSE Streaming                      • Modern Transformer (RoPE, GQA)
    • Multi-Provider Router & Cost Engine               • PagedAttention Virtual Memory
    • Hybrid RAG (Dense + BM25 + RRF)                   • Multimodal VLM (LibraVLM)
    • Sandboxed Code Execution                          • Speculative & Grammar Decoding
    • Autonomous ReAct Agents                           • SFT, ChatML & DPO Alignment
```

---

## 🏛️ The 7 Architectural Pillars

| Pillar | Subsystems & Features | Key Components |
| :--- | :--- | :--- |
| **1. Architecture & Modeling** | Modern Transformer, RoPE, SwiGLU, GQA, PagedAttention, Multimodal VLM | `ModernTransformerLM`, `PagedKVCache`, `LibraVLM` |
| **2. Inference Acceleration** | $O(1)$ KV-Cache Generation, Speculative Decoding, Thompson NFA Grammars | `generate_with_cache`, `SpeculativeDecoder`, `RegexAutomaton` |
| **3. Providers & Routing** | Multi-Provider Router, Semantic Task Classifier, Zero-Cost Telemetry | `ProviderRouter`, `QueryClassifier`, `calculate_cost` |
| **4. Memory & Context** | SQLite Session Store with WAL, Sliding-Window Context Truncation | `SQLiteConversationStore`, `ContextWindowManager` |
| **5. Knowledge & Hybrid RAG** | Dense Vector Embeddings, Okapi BM25 Inverted Index, Reciprocal Rank Fusion | `InMemoryVectorStore`, `BM25Index`, `HybridRetriever` |
| **6. Agents & Safe Tools** | AST Whitelisted Process Sandbox, Calculator, Web Search, ReAct Loop | `SafePythonSandbox`, `ToolRegistry`, `ReActAgent` |
| **7. Training & Deployment** | ChatML Packing with Prompt Loss Masking (`-100`), DPO, Docker Compose, CI | `ChatMLFormatter`, `DeploymentValidator`, `RegressionGate` |

---

## 🗺️ Complete 36-Phase Curriculum

| Phase | Description | Status |
| :---: | :--- | :---: |
| **0** | Project Skeleton, FastAPI backend, Next.js frontend, Hardware Introspection | ✅ Completed |
| **1** | Educational Tiny Transformer (Decoder-only from scratch in PyTorch) | ✅ Completed |
| **2** | First-Principles Tokenizer (Byte-Pair Encoding, Tokenization & Encoding) | ✅ Completed |
| **3** | Data Pipeline, Synthetic Pre-training Corpora & Streaming Dataloaders | ✅ Completed |
| **4** | Modern Transformer Enhancements (RoPE, RMSNorm, SwiGLU activations) | ✅ Completed |
| **5** | CPU Training Engine, AdamW optimizer & Cosine Learning Rate Scheduler | ✅ Completed |
| **6** | Quantitative Model Evaluation, Cross-Entropy Loss & Perplexity Benchmarking | ✅ Completed |
| **7** | Educational Model Registry, Checkpoint Versioning & Metadata Tracking | ✅ Completed |
| **8** | Local Inference Engine, KV Caching & Streaming Generation | ✅ Completed |
| **9** | Unified Provider Adapters (Local, Ollama, Mock Provider) | ✅ Completed |
| **10** | ChatGPT-Like Web Interface (Next.js, Tailwind CSS, SSE Streaming) | ✅ Completed |
| **11** | Model Comparison Arena (Side-by-Side Evaluation & Generation Profiler) | ✅ Completed |
| **12** | Multi-Turn Conversation Memory & SQLite Persistent Storage | ✅ Completed |
| **13** | In-Memory Vector Store & Document Chunking Engine | ✅ Completed |
| **14** | Complete RAG Pipeline & Context-Grounded Prompt Synthesis | ✅ Completed |
| **15** | Live Web Search Integration & Grounded Citations | ✅ Completed |
| **16** | Safe Tool Execution Sandbox (AST Whitelist, Child Process Isolation) | ✅ Completed |
| **17** | Built-in Tools (Calculator, Python Code Interpreter, Web Search) | ✅ Completed |
| **18** | Autonomous Agent Orchestration (ReAct Loop: Thought, Action, Observation) | ✅ Completed |
| **19** | Deep Research Agent (Multi-Source Synthesis & Markdown Report Generation) | ✅ Completed |
| **20** | Advanced Quantization (FP32 to Dynamic INT8 CPU Quantization) | ✅ Completed |
| **21** | Grouped-Query Attention (GQA) & KV Cache Optimization | ✅ Completed |
| **22** | Structured Decoding & Grammar Enforcement (JSON Schema & Thompson NFA Regex) | ✅ Completed |
| **23** | Speculative Decoding Engine (Small Draft + Large Target Verification) | ✅ Completed |
| **24** | Hybrid Search Architecture (Dense Semantic + Sparse Okapi BM25 + RRF) | ✅ Completed |
| **25** | Multi-Modal Architecture (LibraVLM: Image Patch Embeddings + LLM Projector) | ✅ Completed |
| **26** | Direct Preference Optimization (DPO Loss, Reference Model, Preference Pairs) | ✅ Completed |
| **27** | Semantic Routing & Task Complexity Classifier | ✅ Completed |
| **28** | Cost Tracking, Token Economics & Zero-Cost Budget Enforcement | ✅ Completed |
| **29** | Advanced Context Compression, Semantic Pruning & Summarization | ✅ Completed |
| **30** | Evaluation Harness (ARC, HellaSwag, MMLU benchmarks) | ✅ Completed |
| **31** | Agent Memory, Reflection & Tool Use Telemetry | ✅ Completed |
| **32** | PagedAttention & Continuous Batching Virtual Memory Cache | ✅ Completed |
| **33** | Domain Adaptation & SFT Corpus Packing (ChatML & Prompt Loss Masking) | ✅ Completed |
| **34** | Public Deployment & Containerized Lab (Multi-Stage Docker & Compose) | ✅ Completed |
| **35** | CI/CD, Automated Pre-Commit Guards & Performance Regression Gates | ✅ Completed |
| **36** | Final Capstone Integration, System Polish & Comprehensive Release Verification | ✅ Completed |

---

## 💻 Hardware Philosophy & Zero-Cost ($0 / ₹0) Commitment

- **Target Hardware**: Intel Core i5-12450H (8 Cores, 12 Threads), 16GB RAM, Windows/Linux.
- **Zero-Cost**: Never requires paid APIs or cloud GPUs. All components are self-contained and run locally.
- **15-Minute CPU Training Budget**: All educational training runs complete a full cycle in under 15 minutes.
- **15 GB Storage Quota**: Total repository and checkpoint footprint is strictly governed (< 1 GB used).

---

## 🚀 Quick Start

### 1. Local Python Environment (Recommended)
```powershell
# Clone the repository
git clone https://github.com/eklavya434/libra.git
cd libra

# Create and activate Python virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install editable package with development dependencies
pip install -e .[dev]

# Run the Master Capstone System Audit
.\.venv\Scripts\python.exe scripts/run_libra_capstone_audit.py

# Start the FastAPI Backend
.\.venv\Scripts\uvicorn apps.backend.main:app --reload --port 8000
```

### 2. Full Test Suite & Linting
```powershell
# Run the complete Pytest suite (394+ tests)
.\.venv\Scripts\pytest -v tests/

# Run Linter & Formatter
.\.venv\Scripts\ruff check .
.\.venv\Scripts\ruff format .

# Run Pre-Commit Security & Quota Guard
.\.venv\Scripts\python.exe scripts/pre_commit_check.py
```

### 3. Docker Compose Deployment (Phase 34+)
```bash
# Build and launch containerized backend, frontend, and Ollama services
docker compose up -d --build

# Backend API: http://localhost:8000
# Frontend UI:  http://localhost:3000
```

---

## 📚 Educational Whitepaper & Documentation
- **Architectural Whitepaper**: [`docs/architecture/PROJECT_LIBRA_WHITEPAPER.md`](docs/architecture/PROJECT_LIBRA_WHITEPAPER.md)
- **Phase Retrospectives**: Explore detailed guides for every phase in [`docs/educational/`](docs/educational/)
- **API Documentation**: Interactive Swagger docs available at `http://localhost:8000/docs` when running the backend.

---

## 📄 License
Apache 2.0 License — Free for educational, research, and personal use.
