# ♎ Libra: Personal LLM Laboratory & AI Assistant

Libra is an educational large language model (LLM) laboratory and conversational assistant built from first principles. It runs locally on consumer CPU hardware under a zero-cost ($0 / ₹0) policy, demonstrating how modern transformer architectures, inference optimizations, retrieval-augmented generation (RAG), autonomous tool-using agents, and alignment loops operate from raw mathematics to working code.

---

## What This Actually Is

This project is divided into two distinct layers:

1. **Core Educational Implementations (First Principles)**  
   These components were written from scratch to develop an intuitive, mathematical understanding of how foundation models function:
   - **BPE Tokenizer**: Byte-pair encoding algorithm trained and evaluated directly from text corpora.
   - **Modern Transformer Architecture**: Implemented in PyTorch with Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU activations, and Grouped-Query Attention (GQA).
   - **Training & Alignment Engines**: Autoregressive next-token training loop with AdamW, cosine warmup scheduler, gradient accumulation, Direct Preference Optimization (DPO), and Kahneman-Tversky Optimization (KTO).
   - **Inference Algorithms**: Key-Value (KV) cache generation, speculative draft-and-verify decoding, PagedAttention block memory management, and Thompson NFA grammar-constrained decoding.

2. **Platform & Application Infrastructure**  
   These components provide the runtime scaffolding and user experience around the educational cores:
   - **Backend API**: FastAPI application exposing REST and Server-Sent Events (SSE) streaming endpoints with SQLite WAL conversation persistence.
   - **Frontend UI**: Next.js 14 (TypeScript + Tailwind CSS) chat interface with dark-mode terminal aesthetics and dedicated interactive lab visualizers.
   - **Execution Sandbox**: Process-isolated Python sandbox with AST allowlists and Windows Job Object memory limits to safely execute model-generated code.
   - **Provider Adapters**: Standardized client interfaces for local engines (Ollama) and cloud APIs (Google Gemini, OpenAI, Anthropic, DeepSeek, Groq).

> **Scale Note**: All custom PyTorch architectures in this repository are dimensioned educationally (e.g., hidden dimensions between 32 and 256, 1 to 4 layers) so that complete training, alignment, and evaluation runs finish on a standard laptop CPU in under 15 minutes without requiring cloud GPUs.

---

## Hardware Constraints & Operating Rules

The codebase is engineered to respect strict local development budgets:

- **Hardware**: Intel Core i5-12450H CPU (8 physical cores / 12 logical threads), 16 GB RAM, no dedicated GPU.
- **Zero-Cost ($0 / ₹0)**: No paid API calls or cloud compute required. All features run either locally on CPU or via free-tier API allocations.
- **15-Minute CPU Training Budget**: Any educational training loop must complete a full cycle (data loading -> training -> loss convergence -> checkpointing -> generation) in under 15 minutes.
- **15 GB Storage Quota**: Combined disk usage of `.venv`, `node_modules`, model checkpoints, datasets, and caches must stay strictly below 15 GB. The current footprint is ~1.60 GB (89.3% free quota).
- **Local Inference Engine**: Ollama is the default local model server. `vLLM` is not installed on this machine and is documented as an architectural stub for future dedicated NVIDIA GPU hardware.

---

## Model Providers

Libra unifies multiple inference backends behind a single `BaseProvider` interface:

| Provider | Type | Implementation Status | Notes |
| :--- | :--- | :--- | :--- |
| **Local Transformer** | Local CPU | Verified | First-principles PyTorch transformer running directly in-process. |
| **Ollama** | Local Daemon | Verified | Default local inference adapter targeting `http://localhost:11434`. |
| **Mock Provider** | Offline | Verified | Deterministic fixture for unit tests and offline UI development. |
| **Google Gemini** | Cloud API | Verified | Dedicated native adapter (`GeminiProvider`) with streaming support. |
| **OpenAI** | Cloud API | Verified | Official OpenAI REST & streaming adapter. |
| **Anthropic Claude** | Cloud API | Verified | Anthropic Messages API adapter with token streaming. |
| **DeepSeek** | Cloud API | Verified | DeepSeek API adapter with reasoning model support. |
| **Groq** | Cloud API | Verified | Ultra-low latency LPU inference adapter. |
| **Generic OpenAI** | Configurable | Verified | Connects to any OpenAI-compatible API base URL (including NVIDIA NIM, OpenRouter, or remote vLLM clusters). |
| **Hugging Face** | Cloud API | Verified | Hugging Face Inference API adapter. |
| **vLLM** | Future GPU | Architectural Stub | Documented specification only; inactive on CPU machine. |

---

## Project Status & Curriculum

All 50 educational milestones have been implemented, tested, and documented with dedicated retrospectives in [`docs/educational/`](docs/educational/):

### Foundations, Data & Core Modeling (Phases 0–6)
- **Phase 0**: Project skeleton, hardware introspection, FastAPI and Next.js foundations.
- **Phase 1**: Educational Tiny Transformer (477K parameter decoder-only model trained in 19.6s on CPU).
- **Phase 2**: Byte-Pair Encoding (BPE) tokenizer from scratch and Hugging Face tokenizers integration.
- **Phase 3**: Data pipeline with unicode normalization, SHA-256 deduplication, binary sharding, and quota guards.
- **Phase 4**: Modern transformer enhancements (RoPE, RMSNorm, SwiGLU activations, weight tying).
- **Phase 5**: CPU training engine with AdamW, cosine warmup schedule, gradient accumulation, and checkpoint resumption.
- **Phase 6**: Evaluation framework with cross-entropy loss, perplexity tracking, and multi-choice task probes.
- **Comprehension Gate**: Milestone evaluation ensuring mastery of tokens, attention, training steps, and checkpoints.

### Platform, Inference & Retrieval (Phases 7–15)
- **Phase 7**: Model registry, open-weight model catalog, and RAM formula sizing.
- **Phase 8**: Local inference engine, prompt formatting, KV caching, and SSE streaming.
- **Phase 9**: Unified model provider adapters, token cost tracking, and error normalization.
- **Phase 10**: Model Comparison Arena with side-by-side generation profiling (TTFT and token velocity).
- **Phase 11**: Real Libra Chat UI with live SSE token streaming and dynamic parameter tuning.
- **Phase 12**: Multi-turn conversation memory backed by SQLite in WAL mode with context window sliding.
- **Phase 13**: Vector embeddings, cosine similarity search, recursive text chunker, and in-memory store.
- **Phase 14**: Advanced RAG with Okapi BM25 sparse index, dense embeddings, Reciprocal Rank Fusion (RRF), and cross-encoder re-ranking.
- **Phase 15**: Deep Research Agent with multi-query synthesis, live search integration, and Markdown reporting.

### Tools, Agents & Alignment (Phases 16–22)
- **Phase 16**: Process-isolated Python execution sandbox with AST module allowlisting and kernel resource ceilings.
- **Phase 17**: Structured outputs and JSON schema enforcement with incremental pushdown automata.
- **Phase 18**: Autonomous ReAct agent loops and Plan-and-Solve orchestrator with real-time SSE step streaming.
- **Phase 19**: Code generation, Program-Aided Language (PAL) execution, and multi-turn auto-debugging.
- **Phase 20**: Multi-agent collaboration with shared blackboard architecture (Architect, Coder, Reviewer, Tester).
- **Phase 21**: Dynamic query routing, complexity classification, and first-principles speculative decoding.
- **Phase 22**: Alignment with Bradley-Terry reward modeling and Direct Preference Optimization (DPO).

### Advanced Architecture & Acceleration (Phases 23–36)
- **Phase 23**: Dynamic Key-Value cache optimization and Grouped-Query Attention (GQA / MQA).
- **Phase 24**: Post-training quantization (Symmetric INT8 and packed INT4 affine quantization).
- **Phase 25**: Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA).
- **Phase 26**: Streaming token telemetry (surprisal, Shannon entropy, and probability distributions).
- **Phase 27**: Grammar-constrained decoding with Thompson NFA regex automata and Earley CFG parsers.
- **Phase 28**: Automated model benchmarking arena with Bradley-Terry Elo ranking.
- **Phase 29**: Deliberative reasoning engine with Chain-of-Thought parsing, Self-Consistency voting, and Best-of-N verification.
- **Phase 30**: Long-context rotary position scaling (Linear PI, Dynamic NTK-aware RoPE, and YaRN).
- **Phase 31**: PagedAttention virtual memory management and simulated continuous batching.
- **Phase 32**: Multimodal Vision-Language Model (`LibraVLM`) with patch projection and autoregressive generation.
- **Phase 33**: Instruction fine-tuning pipeline with ChatML formatting, multi-turn packing, and prompt loss masking.
- **Phase 34**: Containerized production packaging with multi-stage Dockerfiles and Docker Compose.
- **Phase 35**: Automated CI regression gates, Ruff formatting/linting, and pre-commit security verification.
- **Phase 36**: Master capstone system audit and verification CLI.

### Frontier Systems & Autonomous Evolution (Phases 37–50)
- **Phase 37**: Security hardening with `PromptGuard` injection filtering, `SecretScanner` DLP, and token-bucket rate limiting.
- **Phase 38**: Distributed tracing with OpenTelemetry tracer, W3C trace context, and token velocity metrics.
- **Phase 39**: High-throughput batch inference with dynamic sequence binning and asynchronous worker queues.
- **Phase 40**: Multimodal document understanding (`LibraOCR`), layout-aware chunking, and grounded visual QA.
- **Phase 41**: Stateful notebook environment (`LibraNotebook`), multi-cell variable state, SVG plotting, and data analysis.
- **Phase 42**: Long-context attention compaction (StreamingLLM attention sinks, H2O heavy hitters, and multi-needle benchmark).
- **Phase 43**: Monte Carlo Tree Search (MCTS) reasoning with Process Reward Models (PRMs) and self-play preference generation.
- **Phase 44**: Knowledge distillation and model shrinking with teacher-student soft logit transfer and layer dropping.
- **Phase 45**: Sparse Mixture of Experts (MoE) with top-k noisy gating and load-balancing auxiliary loss.
- **Phase 46**: Speculative verification with Medusa multi-head residual drafting and parallel prefix tree verification.
- **Phase 47**: Direct alignment with Kahneman-Tversky Optimization (KTO) and on-policy online DPO.
- **Phase 48**: Step-level verifiable reasoning search with PRM branch pruning and automatic backtracking.
- **Phase 49**: Self-Rewarding Language Models with LLM-as-a-Judge debiasing and iterative alignment flywheels.
- **Phase 50**: Grand capstone autonomous self-evolution engine and 10-pillar master architectural audit.

---

## Test Suite & Verification

The codebase is covered by an automated test suite spanning 18 distinct test suites:

- **Total Tests**: **564 passing, 0 failing, 0 skipped**
- **Test Runtime**: ~83 seconds on Intel Core i5-12450H CPU
- **Coverage**:
  - `tests/agents/`: 29 tests (ReAct, Plan-and-Solve, PAL, multi-agent teams)
  - `tests/api/`: 132 tests (FastAPI REST and SSE endpoints across all 50 phases)
  - `tests/batch/`: 8 tests (Sequence binning, asynchronous queues)
  - `tests/document/`: 7 tests (Document OCR, layout parsing, visual QA)
  - `tests/evaluation/`: 36 tests (Perplexity, Elo rating, alignment, PRM, Capstone)
  - `tests/frontend/`: 2 tests (Component smoke checks)
  - `tests/integration/`: 4 tests (Pipeline integrations)
  - `tests/memory/`: 12 tests (SQLite WAL, foreign key cascades, sliding window)
  - `tests/models/`: 120 tests (RoPE, SwiGLU, KV cache, GQA, MoE, Medusa, VLM, Thompson NFA)
  - `tests/notebook/`: 11 tests (Stateful execution, SVG plotting)
  - `tests/observability/`: 6 tests (OpenTelemetry spans, token velocity)
  - `tests/providers/`: 22 tests (Adapter routing, cost tracking, offline fallbacks)
  - `tests/rag/`: 32 tests (BM25, dense embeddings, RRF, hybrid retrieval)
  - `tests/routing/`: 13 tests (Query classification, speculative engine)
  - `tests/security/`: 15 tests (Prompt injection, secret scanning, rate limiting)
  - `tests/tools/`: 43 tests (Process sandbox, AST allowlist, adversarial exploits)
  - `tests/training/`: 28 tests (Cosine LR, DPO, KTO, Distillation, Self-Rewarding)
  - `tests/unit/`: 44 tests (BPE tokenizer, data cleaner, sharding, quota guard)

---

## Quick Start

### 1. Local Python Environment (Recommended)

```powershell
# Clone the repository
git clone https://github.com/eklavya434/libra.git
cd libra

# Create and activate Python virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install package in editable mode with development dependencies
pip install -e .[dev]

# Run the 10-Pillar Grand Capstone Audit
.\.venv\Scripts\python.exe scripts/run_phase50_grand_capstone.py

# Start the FastAPI Backend
.\.venv\Scripts\uvicorn apps.backend.main:app --reload --port 8000
```

### 2. Run Tests & Code Quality Checks

```powershell
# Run the complete test suite (564 tests)
.\.venv\Scripts\pytest -v tests/

# Run Linter & Formatter
.\.venv\Scripts\ruff check .
.\.venv\Scripts\ruff format .

# Run Pre-Commit Security & Quota Guard
.\.venv\Scripts\python.exe scripts/pre_commit_check.py
```

### 3. Frontend Web Application

```powershell
cd apps/frontend
npm install
npm run dev
# Interface opens at http://localhost:3000
```

### 4. Docker Compose Deployment (Phase 34+)

```bash
# Build and launch containerized backend, frontend, and Ollama services
docker compose up -d --build

# Backend API: http://localhost:8000
# Frontend UI:  http://localhost:3000
```

---

## Documentation

- **Educational Retrospectives**: Detailed retrospectives for all 50 phases are located in [`docs/educational/`](docs/educational/).
- **Architecture Decisions**: Design records and rationale are preserved in [`docs/architecture/`](docs/architecture/).
- **Interactive API Documentation**: Available at `http://localhost:8000/docs` when running the FastAPI backend.

---

## License

Apache 2.0 License — Free for educational, research, and personal use.