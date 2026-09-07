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
| **Phase 20** | Multi-Agent Collaboration | ✅ COMPLETE | Architect, Coder, Reviewer, Tester collaborative team, SharedBlackboard, POST /api/v1/teams/collaborate, 8 new tests (`main`) |
| **Phase 21** | Dynamic Routing & Speculative Decoding | ✅ COMPLETE | Intent/complexity classification, 4 execution tiers, Leviathan draft-verify speculative engine, POST /api/v1/routing/*, 21 new tests (`main`) |
| **Phase 22** | RLHF & Direct Preference Optimization (DPO) | ✅ COMPLETE | Bradley-Terry reward model, DPO trainer with reference model regularization, POST /api/v1/alignment/*, 12 new tests (`main`) |
| **Phase 23** | KV Cache Optimization & Grouped-Query Attention | ⏳ NEXT | Autoregressive KV cache rolling buffer, MQA / GQA multi-head reduction |

---

## 3. Phase 16 to 22 Ecosystem: Tools, Structured Outputs, Agents, Code, Teams, Routing & Alignment

### A. Phase 16: Tool Use & Secure Sandboxed Execution
- **Registered Tools Catalog (`get_tool_registry()` in `packages/tools/registry.py`)**:
  1. `CalculatorTool` (`calculator`): AST-parsed mathematical calculation; arithmetic, trigonometric, and logarithmic expressions with zero `eval()`.
  2. `PythonInterpreterTool` (`python_interpreter`): Out-of-process sandboxed code execution via `SafePythonSandbox`.
  3. `WebSearchTool` (`web_search`): DuckDuckGo / Mock search adapter with query normalization, result parsing, and snippets.
  4. `KnowledgeBaseTool` (`knowledge_base`): BM25 + dense embedding hybrid retrieval over ingested documents via Reciprocal Rank Fusion (RRF).
- **Multi-Layer Sandboxing & Adversarial Safeguards**:
  - Out-of-process OS execution via `subprocess.Popen([sys._base_executable, runner_path])` with isolated `-I -s` flags.
  - Hard timeout preemptive kill (`proc.kill()` in <1s for infinite loops) -> verified by `test_adversarial_infinite_loop`.
  - Windows Job Object kernel memory ceiling (64–128 MB raising `MemoryError` for allocation bombs) -> verified by `test_adversarial_memory_bomb`.
  - Filesystem chroot isolation (`safe_open` rejecting traversal `../../` and absolute paths outside tempdir) -> verified by `test_adversarial_filesystem_escape_relative` & `test_adversarial_filesystem_escape_absolute`.
  - Network blackholing (`HTTP_PROXY=127.0.0.1:0`) and socket prohibitions -> verified by `test_adversarial_network_access`.
  - Kernel subprocess blocking (`ActiveProcessLimit = 1`) -> verified by `test_adversarial_subprocess_spawn`.
  - AST security visitor with strict module allowlist (`math`, `datetime`, `time`, `statistics`, `random`, `json`, etc.).
  - Built-in exception types (`ValueError`, `TypeError`, `IndexError`, `ZeroDivisionError`, `AssertionError`, `RuntimeError`, `KeyError`, `AttributeError`) safe in `safe_builtins`.

### B. Phase 17: Structured Outputs & Grammar-Constrained Decoders
- **Incremental JSON State Machine (PDA)**: `IncrementalJSONStateMachine` (`packages/core/grammar/json_state_machine.py`) tracks nested objects, arrays, strings, escapes, numbers, and literals character-by-character.
- **JSON Schema Compiler**: `SchemaCompiler` (`packages/core/grammar/schema_compiler.py`) compiles Pydantic models and raw schemas into validation rules and OpenAI/Ollama `response_format` schemas.
- **Constrained Logits Processor**: `ConstrainedLogitsProcessor` (`packages/core/grammar/logits_processor.py`) intercepts autoregressive logits and applies $-\infty$ masks to tokens violating the JSON grammar.
- **Structured Output Generator & Self-Healing Loop**: `StructuredOutputGenerator` (`packages/providers/structured.py`) orchestrates generation with multi-turn reflection repair.
- **REST Endpoints**: `POST /api/v1/structured/generate` and `POST /api/v1/structured/validate`.

### C. Phase 18: Autonomous Multi-Step Agent Loops & Budget Limits
- **ReAct Agent**: `ReActAgent` (`packages/agents/react.py`) implements interleaved `Thought` -> `Action` -> `Observation` loops with scratchpad history, tool dispatching, circuit breakers, and budget guards.
- **Plan-and-Solve Agent**: `PlanAndSolveAgent` (`packages/agents/plan_and_solve.py`) decomposes complex inquiries into explicit milestones via `StructuredOutputGenerator`, executes tools per milestone, and synthesizes findings.
- **Iteration, Timeout & Token Budget Limits**:
  - `ReActAgent`: `max_steps = 10` (clamped 1–30), `timeout_sec = 60.0` (clamped 5–300s), circuit breaker at `max_tool_failures = 3` consecutive failures.
  - `PlanAndSolveAgent`: `max_steps = 10` milestone steps, per-step validation timeouts.
  - `PALAgent`: `max_steps = 5`, `timeout_sec = 30.0`.
  - `CodeAgent` / `AutoDebugger`: `max_debug_iterations = 3` (clamped 1–10), `timeout_sec = 60.0`.
  - `CollaborativeTeam`: `max_rounds = 3` (clamped 1–10), `timeout_sec = 120.0`, `max_steps = max_rounds * 4` (12 steps).
  - `ContextWindowManager`: Sliding-window token budget pruning oldest conversation history while pinning system prompts.
- **Real-Time Step Streaming**: `POST /api/v1/agents/react/stream` SSE endpoint emitting real-time `thought`, `action`, `observation`, and `final_answer` events.
- **REST Endpoints**: `POST /api/v1/agents/react`, `POST /api/v1/agents/react/stream`, `POST /api/v1/agents/plan-and-solve`.

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

### E. Phase 20: Multi-Agent Collaboration & Orchestration
- **Specialized Agent Roles**: `ArchitectAgent` (functional specification), `CoderAgent` (implementation & revision), `ReviewerAgent` (adversarial audit & verdict), `TesterAgent` (unit test assertions & sandbox execution), `Coordinator` (consensus manager).
- **Inter-Agent Message Bus & Shared Blackboard**: `AgentMessage` and `SharedBlackboard` (`packages/agents/message_bus.py`) tracking task state, proposals, reviews, assertions, and dialogue logs.
- **Collaborative Consensus Loop**: `CollaborativeTeam` (`packages/agents/multi_agent.py`) orchestrates multi-stage iterations. Requires both Reviewer approval (`VERDICT: APPROVED`) and Sandbox test execution passing (`PASSED`) to achieve consensus.
- **REST Endpoints**:
  - `POST /api/v1/teams/collaborate`: Synchronous multi-agent collaboration returning `TeamTrajectory`.
  - `POST /api/v1/teams/collaborate/stream`: Real-time SSE streaming of agent-to-agent dialogue and round progress.

### F. Phase 21: Dynamic Model Routing & Speculative Decoding
- **Query Complexity & Intent Classifier (`QueryClassifier` in `packages/routing/classifier.py`)**:
  - Multi-factor complexity scoring ($0.0 \to 1.0$) across length, code density, reasoning markers, and constraints.
  - Intent classification (`GREETING`, `FACTUAL_QA`, `CODE_GENERATION`, `REASONING_MATH`, `CREATIVE_WRITING`, `DATA_EXTRACTION`, `MULTI_STEP_PLANNING`).
  - Tier recommendation: `FAST_LOCAL` (<50ms, $0), `BALANCED` (200-800ms), `FRONTIER_REASONING` (1.0-3.5s), `MULTI_AGENT` (3.0-10.0s).
- **Dynamic Tier Router (`DynamicRouter` in `packages/routing/dynamic_router.py`)**:
  - User policy routing (`auto`, `fast`, `balanced`, `quality`, `multi_agent`).
  - Fallback cascade chaining ensuring resilience against provider outages.
- **First-Principles Speculative Decoding Engine (`SpeculativeDecoder` in `packages/models/speculative.py`)**:
  - Leviathan et al. (2023) draft-and-verify algorithm.
  - Exact mathematical equivalence theorem guaranteed under greedy decoding.
  - Real-time telemetry: proposed draft tokens, accepted draft tokens, acceptance rate ($\alpha$), target forward passes saved, and speedup ratio ($S$).
- **REST Endpoints**:
  - `POST /api/v1/routing/classify`: Query intent and multi-factor complexity.
  - `POST /api/v1/routing/decision`: Dynamic routing plan, selected tier, and fallback chain.
  - `POST /api/v1/routing/generate`: Dynamic routing generation with automated fallback execution.
  - `POST /api/v1/routing/speculative`: Speculative decoding generation with speedup metrics.

### G. Phase 22: Reinforcement Learning from First Principles (RLHF & DPO)
- **Preference Dataset & Token Collator (`packages/training/preference_dataset.py`)**:
  - Triplet loader `(prompt, chosen, rejected)` with span masking for completion-only loss evaluation.
  - Label masking fills prompt token positions with `-100`, preventing loss pollution from prompt tokens.
- **First-Principles Reward Model (`TransformerRewardModel` in `packages/models/reward_model.py`)**:
  - Transformer backbone with scalar regression head ($d_{\text{model}} \to 1$).
  - Bradley-Terry pairwise preference ranking loss: $\mathcal{L}_{\text{RM}} = -\log \sigma(r(x, y_w) - r(x, y_l) - \text{margin})$.
  - Telemetry: chosen reward mean, rejected reward mean, reward margin ($r_w - r_l$), and ranking accuracy.
- **Direct Preference Optimization Trainer (`DPOTrainer` in `packages/training/dpo_trainer.py`)**:
  - Closed-form RLHF without PPO; trains policy $\pi_\theta$ against frozen reference $\pi_{\text{ref}}$.
  - Implicit reward calculation: $\hat{r}(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$.
  - Fast convergence: drives chosen logp higher, suppresses rejected logp, and achieves 100% preference accuracy in $< 300$ ms on CPU.
- **REST Endpoints**:
  - `POST /api/v1/alignment/reward`: Score prompt-completion pair with reward model.
  - `POST /api/v1/alignment/reward/rank`: Rank candidate completions for a prompt.
  - `POST /api/v1/alignment/dpo/step`: Run single DPO optimization step on preference batch.

### H. Multimodal Provider Status (Vision vs Generation)
- **Image Understanding (Vision)**: **ACTIVE** via unified provider abstraction (`supports_vision: True` in `packages/models/registry.py` and provider adapters `openai.py`, `gemini.py`, `anthropic.py`, `ollama.py`). Message schemas support multimodal base64 / URL image inputs for models such as `gpt-4o`, `gemini-1.5-pro`, `claude-3-5-sonnet`, and local `llava` via Ollama.
- **Image Generation (Diffusion)**: **STUBBED / DEFERRED**. Per Prime Directive #2 (Zero-Cost $0/₹0) and Prime Directive #3 & #4 (15-min CPU budget, 15 GB quota), cloud image generation APIs (DALL-E, Imagen) and multi-gigabyte local Stable Diffusion weights are not loaded. Local text-to-image is architected as an optional modular stub (`MockImageGenerator`).

---

## 4. Resource Usage & Storage Quota Audit

- **Venv Size**: ~855 MB
- **Frontend node_modules**: ~281 MB
- **Models & Checkpoints**: 20.08 MB
- **Total Workspace Footprint**: **1,202.45 MB** (~1.20 GB)
- **15 GB Quota Limit**: 15,360.00 MB
- **Remaining Storage Quota**: **14,157.55 MB** (92.17% free)
- **Total Cost**: **$0 / ₹0** (100% free offline development)
- **Active Git Branch**: `main` synced with `https://github.com/eklavya434/libra.git`
- **Pytest Status**: **266 passed, 0 failed** (in 43.55s)
- **Frontend Status**: Next.js 14 production build clean (0 errors)






