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
| **Phase 14** | Advanced RAG & Web Search | ✅ COMPLETE | Okapi BM25 sparse index, RRF fusion, multi-factor re-ranking, DuckDuckGo & Mock Web Search providers (`f3b1128`) |
| **Phase 15** | Streaming UX & Deep Research Agent | ✅ COMPLETE | StreamingMarkdown, CodeBlock, AbortController, DeepResearchAgent multi-query workflow & synthesis (`028ee59`) |
| **Phase 16** | Tool Use & Sandbox Execution | ✅ COMPLETE | Process-isolated sandbox, AST allowlist, Windows Job Objects memory ceiling, CalculatorTool, WebSearchTool, KnowledgeBaseTool, 5 adversarial tests passing (`main`) |
| **Phase 17** | Structured Outputs & Grammar Decoders | ⏳ NEXT | JSON Schema enforcement, CFG guided generation, constrained decoding |

---

## 3. Phase 16 Ecosystem: Tool Use & Sandboxed Execution Status

### A. Tool Architecture & Registry
- **Base Abstraction**: `BaseTool` and `ToolResult` (`packages/tools/base.py`) with Pydantic validation, schema generation (`to_openai_schema()`), and timing isolation.
- **Tool Registry**: `ToolRegistry` (`packages/tools/registry.py`) with singleton `get_tool_registry()`, pre-registering:
  1. `CalculatorTool`: AST-based arithmetic and math evaluation without `eval()`.
  2. `PythonInterpreterTool`: Multi-line Python execution in a secure sandbox.
  3. `WebSearchTool`: DuckDuckGo / Mock live web search integration.
  4. `KnowledgeBaseTool`: Hybrid BM25 + Dense vector retrieval over indexed documents.
- **Tool Call Parser**: `ToolCallParser` (`packages/tools/parser.py`) extracts tool calls from XML `<tool_call>...</tool_call>`, Markdown code fences, or direct JSON structures, with conversational text cleaning (`strip_tool_calls`).
- **REST Endpoints**: `GET /api/v1/tools`, `GET /api/v1/tools/schemas`, `POST /api/v1/tools/execute`, `POST /api/v1/tools/parse` (`apps/backend/api/v1/endpoints/tools.py`).

### B. Multi-Layer Sandboxing & Adversarial Hardening
- **Out-of-Process OS Child Execution**: Child process launched via `subprocess.Popen([sys._base_executable, runner_path])` with `-I -s` isolation, completely eradicating thread abandonment and GIL starvation.
- **Kernel-Enforced Memory Limit**: Windows Job Object Extended Limit Information (`ProcessMemoryLimit` / `JobMemoryLimit`) enforces hard memory caps (64–128 MB), denying memory bombs and runaway loops.
- **Preemptive OS Timeout Termination**: `proc.kill()` calls Win32 `TerminateProcess` / POSIX `SIGKILL` on timeout, guaranteeing immediate process eradication.
- **Filesystem Chroot Isolation**: Custom `safe_open()` checks `os.path.commonpath()` against an ephemeral `tempfile.TemporaryDirectory()`, blocking directory traversal (`../../`) and absolute system paths.
- **Kernel Subprocess Blocking**: Windows Job Object `ActiveProcessLimit = 1` prevents executed code from spawning child processes or subprocesses.
- **Network Blackholing**: Proxy environment variables directed to `127.0.0.1:0`.
- **AST Security Visitor**: Prohibits non-allowlisted imports, reflection dunders (`__subclasses__`, `__class__`), and dangerous builtins.

---

## 4. Resource Usage & Storage Quota Audit

- **Venv Size**: ~855 MB
- **Frontend node_modules**: ~281 MB
- **Models & Checkpoints**: 20.08 MB
- **Total Workspace Footprint**: **1,157.31 MB** (~1.16 GB)
- **15 GB Quota Limit**: 15,360.00 MB
- **Remaining Storage Quota**: **14,202.69 MB** (92.47% free)
- **Total Cost**: **$0 / ₹0** (100% free offline development)
- **Active Git Branch**: `main` synced with `https://github.com/eklavya434/libra.git`
- **Pytest Status**: **181 passed, 0 failed** (in 25.21s)
- **Frontend Status**: Next.js 14 production build clean (0 errors)






