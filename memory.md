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
| **Phase 17** | Structured Outputs & Grammar Decoders | ✅ COMPLETE | Incremental JSON Pushdown Automaton, SchemaCompiler, ConstrainedLogitsProcessor, self-healing repair loop, POST /api/v1/structured/generate (`main`) |
| **Phase 18** | Multi-Step Agentic Loops | ✅ COMPLETE | ReAct agent loop, Plan-and-Solve orchestrator, real-time SSE step streaming, circuit breakers, budget guards (`main`) |
| **Phase 19** | Code Generation & Auto-Debugging | ✅ COMPLETE | PALAgent, CodeAgent, AutoDebugger self-correction loop, POST /api/v1/coder/pal, 20 new tests (`main`) |
| **Phase 20** | Multi-Agent Collaboration | ⏳ NEXT | Agent-to-agent communication, coordinator/worker patterns, collaborative problem solving |

---

## 3. Phase 16, 17, 18 & 19 Ecosystem: Tools, Structured Outputs, Agents & Code Execution

### A. Phase 16: Tool Use & Secure Sandboxed Execution
- **Base Abstraction**: `BaseTool` and `ToolResult` (`packages/tools/base.py`) with Pydantic validation, schema generation (`to_openai_schema()`), and timing isolation.
- **Tool Registry**: `ToolRegistry` (`packages/tools/registry.py`) with singleton `get_tool_registry()`, pre-registering `CalculatorTool`, `PythonInterpreterTool`, `WebSearchTool`, and `KnowledgeBaseTool`.
- **Multi-Layer Sandboxing**:
  - Out-of-process execution via `subprocess.Popen([sys._base_executable, runner_path])` with `-I -s` isolation.
  - Windows Job Object kernel memory limits (`ProcessMemoryLimit` / `JobMemoryLimit`).
  - Preemptive hard timeout process termination (`proc.kill()`).
  - Filesystem chroot isolation via scoped `safe_open()`.
  - Kernel subprocess blocking (`ActiveProcessLimit = 1`).
  - Network blackholing (`HTTP_PROXY=127.0.0.1:0`).
  - AST security visitor with strict module allowlist (`math`, `datetime`, `time`, `statistics`, `random`, `json`, etc.).

### B. Phase 17: Structured Outputs & Grammar-Constrained Decoders
- **Incremental JSON State Machine (PDA)**: `IncrementalJSONStateMachine` (`packages/core/grammar/json_state_machine.py`) tracks nested objects, arrays, strings, escapes, numbers, and literals character-by-character.
- **JSON Schema Compiler**: `SchemaCompiler` (`packages/core/grammar/schema_compiler.py`) compiles Pydantic models and raw schemas into validation rules and OpenAI/Ollama `response_format` schemas.
- **Constrained Logits Processor**: `ConstrainedLogitsProcessor` (`packages/core/grammar/logits_processor.py`) intercepts autoregressive logits and applies $-\infty$ masks to tokens violating the JSON grammar.
- **Structured Output Generator & Self-Healing Loop**: `StructuredOutputGenerator` (`packages/providers/structured.py`) orchestrates generation with multi-turn reflection repair.
- **REST Endpoints**: `POST /api/v1/structured/generate` and `POST /api/v1/structured/validate`.

### C. Phase 18: Autonomous Multi-Step Agent Loops
- **ReAct Agent**: `ReActAgent` (`packages/agents/react.py`) implements interleaved `Thought` -> `Action` -> `Observation` loops with scratchpad history, tool dispatching, circuit breakers (`max_tool_failures=3`), and budget limits (`max_steps=10`, `timeout_sec=60.0`).
- **Plan-and-Solve Agent**: `PlanAndSolveAgent` (`packages/agents/plan_and_solve.py`) decomposes complex inquiries into explicit milestones via `StructuredOutputGenerator`, executes tools per milestone, and synthesizes findings.
- **Real-Time Step Streaming**: `POST /api/v1/agents/react/stream` SSE endpoint emitting real-time `thought`, `action`, `observation`, and `final_answer` events.
- **REST Endpoints**: `POST /api/v1/agents/react`, `POST /api/v1/agents/react/stream`, `POST /api/v1/agents/plan-and-solve` (`apps/backend/api/v1/endpoints/agents.py`).

### D. Phase 19: Code Generation, Auto-Debugging & Program-Aided Language Models (PAL)
- **Program-Aided Language Models (PAL)**: `PALAgent` (`packages/agents/pal.py`) offloads arithmetic, combinatorics, date/time logic, and symbolic computation to Python scripts executed in `SafePythonSandbox`.
- **Code Extraction & Normalization**: `extract_code()` parses markdown blocks and raw Python text; `normalize_pal_code()` guarantees `solution()` invocation for sandboxed execution.
- **Test-Driven Auto-Debugging (Self-Correction)**:
  - `AutoDebugger` and `CodeAgent` (`packages/agents/coder.py`) execute synthesized code against unit test assertions in `SafePythonSandbox`.
  - Captures runtime crashes (`SyntaxError`, `IndexError`, `AssertionError`, `ZeroDivisionError`) and formatted tracebacks with line numbers.
  - Multi-turn reflection repair prompts LLM with line-numbered source, stack frames, and failure context to generate verified repairs.
  - `DebugIteration` and `CodeTrajectory` record complete debugging history and metrics.
- **REST Endpoints**:
  - `POST /api/v1/coder/pal`: Program-Aided Language Model solving.
  - `POST /api/v1/coder/generate`: Code generation with test validation.
  - `POST /api/v1/coder/debug`: Direct auto-debugging for user-provided broken code.

---

## 4. Resource Usage & Storage Quota Audit

- **Venv Size**: ~855 MB
- **Frontend node_modules**: ~281 MB
- **Models & Checkpoints**: 20.08 MB
- **Total Workspace Footprint**: **1,202.14 MB** (~1.20 GB)
- **15 GB Quota Limit**: 15,360.00 MB
- **Remaining Storage Quota**: **14,157.86 MB** (92.17% free)
- **Total Cost**: **$0 / ₹0** (100% free offline development)
- **Active Git Branch**: `main` synced with `https://github.com/eklavya434/libra.git`
- **Pytest Status**: **225 passed, 0 failed** (in 40.04s)
- **Frontend Status**: Next.js 14 production build clean (0 errors)






