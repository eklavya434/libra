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
| **Phase 23** | KV Cache Optimization & Grouped-Query Attention | ✅ COMPLETE | Dynamic KVCache, GQA/MQA (torch.repeat_interleave), RoPE start_pos offset, generate_with_cache, POST /api/v1/attention/*, 11 new tests (`main`) |
| **Phase 24** | Quantization (INT8 / INT4 & Post-Training Quantization) | ✅ COMPLETE | Symmetric/asymmetric affine quantization, packed INT4 nibbles, QuantizedLinearINT8/INT4, PTQ engine, POST /api/v1/quantization/*, 9 new tests (`main`) |
| **Phase 25** | Parameter-Efficient Fine-Tuning (PEFT & LoRA) | ✅ COMPLETE | LoRALinear, W0 + (alpha/r)*B*A decomposition, zero-init identity, adapter save/load/merge, POST /api/v1/peft/*, 9 new tests (`main`) |
| **Phase 26** | Streaming Token Telemetry & Token-Level Metrics | ✅ COMPLETE | Surprisal, Shannon entropy, top-k alternative probabilities, SSE telemetry (`main`) |
| **Phase 27** | Constrained Decoding & Grammar Masking | ✅ COMPLETE | Thompson NFA regex compiler, Earley CFG parser, 0.00% syntax error guarantee (`main`) |
| **Phase 28** | Continuous Benchmarking & Automated Arena | ✅ COMPLETE | Bradley-Terry Elo rating engine, dual-pass referee, round-robin tournament (`4b99660`) |
| **Phase 29** | Deliberative Reasoning Engine & Test-Time Compute | ✅ COMPLETE | CoT trace parser (<think>), Self-Consistency majority voting, Best-of-N verifier (`7b7ccae`) |
| **Phase 30** | Long-Context Architecture & Rotary Position Scaling | ✅ COMPLETE | Linear PI, Dynamic NTK-Aware RoPE, YaRN, Needle-in-a-Haystack benchmark (`66de7e6`) |
| **Phase 31** | Advanced Inference Optimization (PagedAttention) | ✅ COMPLETE | Virtual memory block paging for KV caches, continuous batching simulation, zero external fragmentation (`f438e81`) |
| **Phase 32** | Multi-Modal Architecture (Vision-Language Adapter) | ✅ COMPLETE | Image patch embedder, LLaVA MLP & Perceiver adapter, end-to-end LibraVLM (`main`) |
| **Phase 33** | Domain Adaptation & Instruction Fine-Tuning Corpus | ✅ COMPLETE | ChatML formatting, multi-turn packing, loss masking, domain instruction dataset pipeline (`main`) |
| **Phase 34** | Production Packaging & Containerization | ✅ COMPLETE | Multi-stage Dockerfiles, docker-compose orchestration, non-root security, health checks (`main`) |
| **Phase 35** | CI/CD & Automated Quality Gates | ✅ COMPLETE | GitHub Actions CI workflow, Ruff lint/format, pytest matrix, Next.js build verification (`main`) |
| **Phase 36** | Capstone System Verification & Architecture Audit CLI | ✅ COMPLETE | 10-vector automated audit engine, REST endpoint, interactive CLI demo, system health verification (`main`) |
| **Phase 37** | Security Hardening & Adversarial Robustness | ✅ COMPLETE | PromptGuard, SecretScanner DLP, TokenBucketRateLimiter, OWASP middleware, Security Lab UI (`main`) |
| **Phase 38** | Observability, Distributed Tracing & OpenTelemetry | ✅ COMPLETE | OpenTelemetry Tracer, Span hierarchy, W3C traceparent, TokenVelocityMetrics, Waterfall UI (`main`) |
| **Phase 39** | High-Throughput Batch Inference & Async Workers | ✅ COMPLETE | Dynamic sequence binning, left-padding, async worker queue, Batch Lab UI (`main`) |
| **Phase 40** | Multi-Modal Document Understanding & OCR Pipeline | ✅ COMPLETE | LibraOCR, table & formula parsing, layout-aware chunker, grounded visual QA (`main`) |
| **Phase 41** | Stateful Code Interpreter & Data Analytics Sandbox | ⏳ NEXT | LibraNotebook, multi-cell state, data visualization (SVG/charts), automated table analysis |


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

### I. Phase 23: KV Cache Optimization & Grouped-Query Attention (MQA / GQA)
- **Dynamic Key-Value Cache (`KVCache` in `packages/models/components/kv_cache.py`)**:
  - Per-layer key/value rolling memory buffer managing prefill ($T$) and decode ($T=1$) steps.
  - Converts autoregressive sequence decode time complexity from $O(T^2)$ down to $O(T)$ cumulative ($O(1)$ per token).
  - Built-in `memory_bytes()` tracking and `reset()` memory flushing.
- **Grouped-Query Attention (GQA) & Multi-Query Attention (MQA)**:
  - Added `n_kv_heads` to `ModernTransformerConfig` with strict divisibility validation ($n_{\text{heads}} \pmod{n_{\text{kv\_heads}}} == 0$).
  - When $n_{\text{kv\_heads}} = n_{\text{heads}}$: Multi-Head Attention (MHA).
  - When $1 < n_{\text{kv\_heads}} < n_{\text{heads}}$: Grouped-Query Attention (GQA).
  - When $n_{\text{kv\_heads}} = 1$: Multi-Query Attention (MQA).
  - Efficient head broadcasting using `torch.repeat_interleave` across query head groups.
- **Offset RoPE & Causal Masking**:
  - `RotaryEmbedding` upgraded with `start_pos` parameter to slice sinusoidal matrices accurately during single-token decode passes.
  - Causal masking bypassed dynamically when $T=1$ during cached generation.
- **Exact Token-for-Token Equivalence**:
  - `generate_with_cache` produces 100% identical token sequences to un-cached `generate()` under greedy decoding ($T=0$).
### J. Phase 24: Quantization (INT8 / INT4 & Post-Training Quantization)
- **Quantization Mathematical Core (`packages/models/quantization/quant_core.py`)**:
  - First-principles symmetric quantization for signed weights ($Z = 0$, $S = \max(|X|)/q_{\text{max}}$).
  - Asymmetric affine quantization for activations and skewed data ($S = (\beta - \alpha)/(2^b - 1)$, calculated zero-point $Z$).
  - Bit-level INT4 packing and unpacking: packs two signed 4-bit nibbles $[-8, 7]$ into a single `torch.uint8` byte using bit-shifting (`(high << 4) | low`), halving storage over unpacked INT4.
  - Metrics: MSE, SQNR in dB, and cosine similarity.
- **Quantized Linear Layers (`packages/models/quantization/quant_linear.py`)**:
  - `QuantizedLinearINT8`: 4x weight memory reduction with per-channel scaling.
  - `QuantizedLinearINT4`: 8x weight memory reduction with bit-packed weights and on-the-fly dequantization.
- **Post-Training Quantization Engine (`packages/models/quantization/ptq.py`)**:
  - Recursive layer replacement (`quantize_model`) for transformer backbones (`ModernTransformerLM`).
  - Memory audits (`compute_model_memory`) and fidelity audits (`audit_quantization_fidelity`).
- **REST Endpoints**:
  - `POST /api/v1/quantization/benchmark`: Size, memory reduction, latency, and fidelity metrics across FP32, INT8, and INT4.
  - `POST /api/v1/quantization/convert`: Model quantization audit endpoint.

### K. Phase 25: Parameter-Efficient Fine-Tuning (PEFT & LoRA)
- **First-Principles Low-Rank Linear Layer (`LoRALinear` in `packages/models/lora/lora_linear.py`)**:
  - Decomposition: $W = W_0 + \frac{\alpha}{r} (B \cdot A)$.
  - Base weight $W_0$ frozen (`requires_grad=False`).
  - Matrix $A$ initialized with Kaiming uniform; Matrix $B$ initialized to **zeros**, ensuring exact numerical identity with base model at initialization.
  - Dynamic zero-latency weight folding (`merge_weights`) and unfolding (`unmerge_weights`).
- **Model Adapter Management (`packages/models/lora/lora_model.py`)**:
  - `apply_lora`: Converts target transformer projections (`q_proj`, `v_proj`) into LoRA layers, training $<1\%$ of total model parameters.
  - `save_lora_adapter` / `load_lora_adapter`: Lightweight serialization saving only trainable adapter tensors (~few KB).
  - Whole-model merging (`merge_lora_weights`) for production deployment.
- **CPU LoRA Trainer (`LoRATrainer` in `packages/training/lora_trainer.py`)**:
  - Memory-efficient AdamW optimizer loop updating exclusively low-rank parameters with gradient clipping.
- **REST Endpoints**:
  - `POST /api/v1/peft/apply`: Parameter count and memory audit.
  - `POST /api/v1/peft/train_step`: Single step fine-tuning optimization.
  - `POST /api/v1/peft/merge`: Zero-overhead inference merging verification.

### L. Phase 26: Streaming Token Telemetry & Token-Level Metrics
- **First-Principles Telemetry Core (`packages/models/telemetry.py`)**:
  - Exact token probability $p(w_t \mid w_{<t}) = \text{softmax}(z_t)[w_t]$ and natural log-probability $\ln p(w_t \mid w_{<t})$.
  - Shannon Surprisal: $I(w_t) = -\log_2 p(w_t \mid w_{<t})$ in bits (quantifies self-information / unexpectedness).
  - Next-token distribution entropy: $H(P_t) = -\sum_{v} p_t(v) \log_2 p_t(v)$ in bits (quantifies model uncertainty).
  - Top-$k$ alternative candidate extraction with decoded text, probability %, and logprobs.
  - Per-token step generation latency ($\Delta t$ ms) and cumulative throughput ($N / \sum \Delta t$ tok/s).
  - Sequence Perplexity identity verification: $\text{PPL} = 2^{\bar{I}} = \exp(-\frac{1}{N}\sum \ln p(w_t))$.
  - Outlier detection: identifies highest-surprisal token ($w_{\text{max}}$) and highest-confidence token ($w_{\text{min}}$).
  - Autoregressive generators: `stream_generate_with_telemetry` and async `astream_generate_with_telemetry`.
  - Teacher-forcing evaluation: `analyze_sequence_telemetry` computes surprisal and entropy without sampling.
- **REST & SSE Endpoints (`apps/backend/api/v1/endpoints/telemetry.py`)**:
  - `POST /api/v1/telemetry/generate`: Non-streaming generation returning full `SequenceTelemetry`.
  - `POST /api/v1/telemetry/stream`: Real-time SSE streaming emitting `event: token` and `event: done` chunks.
  - `POST /api/v1/telemetry/analyze`: Evaluates teacher-forcing surprisal across arbitrary input text.
- **Frontend Surprisal Heatmap & Token Inspector (`apps/frontend/src/components/TokenSurprisalHeatmap.tsx`)**:
  - Color-coded tokens based on surprisal: Green ($I < 1.0$ bit, $p > 50\%$), Amber ($1.0 \le I \le 3.0$ bits), Rose ($I > 3.0$ bits).
  - Interactive popover on click/hover displaying token ID, probability %, surprisal, entropy, latency, and top-5 alternative candidate probability bars.
  - Integrated into `ChatArea.tsx` with "Inspect Tokens" toggle for any assistant message.
- **Verification & Tests**:
  - 10 comprehensive tests in `tests/models/test_telemetry.py` and `tests/api/test_telemetry_endpoint.py`.
  - Interactive CLI demo in `scripts/run_phase26_telemetry_demo.py` with live ANSI colored token streaming.

### M. Phase 27: Constrained Decoding & Grammar Masking (CFG & Regex)
- **Thompson NFA Regex Automaton (`packages/core/grammar/regex_automaton.py`)**:
  - Compiles arbitrary regex patterns into an NFA state graph with $\epsilon$-closures.
  - Supports character classes, negation, wildcard, concatenation, alternation, repetition (`*`, `+`, `?`, `{n,m}`).
  - First-principles prefix validation `is_valid_prefix(s)` and acceptance testing `is_accepted(s)`.
- **Context-Free Grammar Earley Parser (`packages/core/grammar/cfg_parser.py`)**:
  - Jay Earley (1970) chart parser implementing Predictor, Scanner, and Completer operations.
  - Validates arbitrary BNF/EBNF context-free grammars (arithmetic, SQL, DSLs).
  - Exact next terminal symbol lookahead extraction via `get_valid_next_terminals()`.
- **Universal Grammar Logits Processor (`packages/core/grammar/grammar_processor.py`)**:
  - Filters autoregressive logits by setting disallowed continuation tokens to $-\infty$.
  - Guarantees strictly 0.00% syntax error rate across all generated outputs.
  - Enables EOS token only when the grammar reaches an accepted state.
- **REST Endpoints (`apps/backend/api/v1/endpoints/grammar.py`)**:
  - `POST /api/v1/grammar/generate`: Constrained generation with verification.
  - `POST /api/v1/grammar/validate`: Validates candidate string prefix and acceptance.
  - `POST /api/v1/grammar/next_tokens`: Inspects allowed next vocabulary tokens.
- **Verification & Tests**:
  - 10 comprehensive unit and API tests in `tests/models/test_grammar.py` and `tests/api/test_grammar_endpoint.py`.
  - Interactive CLI demo in `scripts/run_phase27_grammar_demo.py` verifying 0.00% error rate across 50 independent runs.

---

### N. Phase 28: Continuous Benchmarking & Automated Model Arena
- **Bradley-Terry Elo Rating Engine (`packages/evaluation/arena_elo.py`)**:
  - Pairwise outcome updates with logistic expected score $E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$.
  - Continuous skill tracking, confidence intervals, win-rate calculations, and leaderboard generation.
- **Position-Bias Mitigated LLM-as-a-Referee (`packages/evaluation/arena_referee.py`)**:
  - Dual-pass evaluation (evaluating candidates as both (A, B) and (B, A)) to detect and cancel out first-token and positional primacy bias.
- **Round-Robin Tournament Runner (`packages/evaluation/arena_tournament.py`)**:
  - Automated tournament running multi-turn match-ups across model pairs on benchmark test suites with JSON serialization.
- **REST Endpoints & Frontend**:
  - `GET /api/v1/arena/leaderboard`, `POST /api/v1/arena/match`, `POST /api/v1/arena/tournament`.
  - Frontend Elo Leaderboard tab with tier badges, match records, and win-rate statistics.

### O. Phase 29: Deliberative Reasoning Engine & Test-Time Compute
- **Reasoning Trace Parser & State Machine (`packages/models/reasoning/trace_parser.py`)**:
  - Real-time streaming delta parser separating `<think>...</think>` cognitive scratchpads from user answers.
  - Step extractor (`extract_reasoning_steps`) parsing numbered steps, bullet points, and paragraph transitions.
  - Static trace parser (`parse_reasoning_trace`) measuring deliberation duration and token density.
- **Self-Consistency Majority Voting (`packages/models/reasoning/self_consistency.py`)**:
  - Stochastic multi-trajectory rollout orchestrator (`SelfConsistencyEngine`).
  - Canonical answer normalization (`normalize_answer`) resolving numeric, algebraic, and boolean expressions.
  - Plurality majority voting with agreement confidence calculation.
- **Best-of-N Test-Time Compute Verifier (`packages/models/reasoning/search_verifier.py`)**:
  - Multi-candidate generation at higher sampling temperatures ($T=0.7$).
  - Multi-criteria referee scoring assessing correctness, soundness, completeness, and clarity.
  - Selects highest-scoring trajectory with fallback tie-breaking.
- **REST Endpoints & Frontend UX**:
  - `POST /api/v1/reasoning/generate`: End-to-end deliberative generation with parsed trace.
  - `POST /api/v1/reasoning/self-consistency`: Multi-path rollout with majority vote resolution.
  - `POST /api/v1/reasoning/best-of-n`: Best-of-$N$ search verification with scored candidate rankings.
  - `POST /api/v1/reasoning/parse`: Static trace analysis.
  - `ReasoningTraceAccordion.tsx`: Collapsible thinking process accordion with live elapsed timers and step markers.
  - Local Ollama auto-discovery: dynamic catalog fetching models like `qwen3:4b` with streaming thinking support.

### P. Phase 30: Long-Context Architecture & Rotary Position Scaling (YaRN & Dynamic NTK)
- **First-Principles RoPE Scaling Core (`packages/models/components/rope_scaling.py`)**:
  - `compute_freqs_linear`: Linear position interpolation ($t' = t/s$).
  - `compute_freqs_dynamic_ntk`: Base frequency scaling ($\theta_{\text{base}}' = \theta_{\text{base}} \cdot s^{d/(d-2)}$) preserving high-frequency local grammar.
  - `compute_freqs_yarn`: Band-split ramp interpolation ($r_i > 32$ extrapolate, $r_i < 1$ interpolate, mid ramp blend) + attention entropy temperature scaling ($\tau = 1 / \sqrt{0.1 \ln s + 1}$).
  - `ScaledRotaryEmbedding`: Unified drop-in RoPE module dynamically expanding buffers during inference.
- **Needle-In-A-Haystack (NIAH) Benchmark (`packages/evaluation/needle_haystack.py`)**:
  - Distractor synthesizer with configurable target word lengths ($250 \to 4000+$).
  - Depth fraction positioning ($0\%$ to $100\%$) and exact/fuzzy factual recall scoring.
- **REST Endpoints & Frontend UX**:
  - `POST /api/v1/context/scale`: Frequencies, effective wavelengths, and attention temperature factor.
  - `POST /api/v1/context/needle`: Full 2D Needle-In-A-Haystack retrieval matrix benchmark.
  - `POST /api/v1/context/perplexity`: Theoretical perplexity scaling simulation across scale factors.
  - `LongContextHeatmap.tsx`: 2D retrieval heatmap with interactive cell inspection.

### Q. Phase 31: Advanced Inference Optimization (PagedAttention & Continuous Batching)
- **Paged KV Cache Virtual Memory Manager (`packages/models/components/paged_cache.py`)**:
  - `PhysicalBlockPool`: Non-contiguous physical memory blocks for keys and values with zero external fragmentation.
  - `BlockAllocator`: Free list allocator with reference counting for prefix caching and copy-on-write sharing.
  - `SequenceBlockTable`: Per-sequence page table resolving logical token positions to physical blocks.
  - `PagedKVCache`: Multi-layer cache manager reducing internal memory waste to $< 3.5\%$.
- **PagedAttention Decode Kernel (`packages/models/components/paged_attention.py`)**:
  - Direct multi-head attention over non-contiguous physical memory blocks with exact numerical parity ($< 1.2 \times 10^{-7}$).
- **Continuous (Iteration-Level) Batching Engine (`packages/models/inference/continuous_batching.py`)**:
  - Interleaves prompt prefill steps with single-token decode iterations.
  - Evicts completed sequences immediately, freeing memory without waiting for longest request.
- **REST Endpoints & Frontend UX**:
  - `POST /api/v1/paged/simulate`: Audits memory savings and concurrency multipliers.
  - `POST /api/v1/paged/batch_run`: Simulates continuous batching runs across varying token budgets.
  - `GET /api/v1/paged/memory_stats`: Real-time block allocation metrics.
  - `PagedMemoryInspector.tsx`: UI component displaying memory savings and step timeline.

### R. Phase 32: Multi-Modal Architecture (Vision-Language Adapter: Patch Projection & Cross-Attention)
- **Vision Patch Embedding Core (`packages/models/vision/patch_embed.py`)**:
  - `ImagePatchEmbedder`: 2D image decomposition into flattened patch vectors ($P \times P \times C$) with learned 2D spatial position embeddings.
  - `generate_synthetic_image`: Deterministic image synthesis (checkerboard, gradient, solid) for zero-download CPU testing.
- **Multi-Modal Alignment Adapter (`packages/models/vision/vlm_projector.py`)**:
  - `VisionLanguageAdapter`: Projects visual feature dimensions ($d_{\text{vision}}$) into LLM token embedding space ($d_{\text{model}}$).
  - Supports LLaVA-style 2-layer GELU MLP and Flamingo-style Perceiver cross-attention latent resampling.
- **End-to-End LibraVLM (`packages/models/vision/vlm_model.py`)**:
  - Unified multi-modal model prepending visual patch tokens as visual prefixes before text prompt token IDs.
  - Autoregressive causal generation conditioning text generation on visual semantics.
- **REST Endpoints & Frontend UX**:
  - `POST /api/v1/multimodal/embed_image`: Patch grid decomposition, shape audits, and token representations.
  - `POST /api/v1/multimodal/generate`: Image-conditioned text generation with latency telemetry.
  - `VisionPlayground.tsx`: Interactive Next.js component displaying the $4 \times 4$ visual patch grid and generated descriptions.

### S. Phase 33: Domain Adaptation & Instruction Fine-Tuning Corpus
- **Instruction Data Pipelines (`packages/training/sft_dataset.py`)**:
  - ChatML formatter (`ChatMLFormatter`) supporting multi-turn dialogue with `<|im_start|>` and `<|im_end|>` delimiters.
  - Multi-turn sequence packing with loss masking (prompt tokens set to `-100`, only assistant response tokens backpropagated).
  - Domain corpus synthesis and dataset loaders for instruction fine-tuning on CPU.

### T. Phase 34: Production Packaging & Containerization
- **Container Architecture**:
  - Multi-stage backend `Dockerfile` (Python 3.11-slim, non-root user `libra`, security hardening).
  - Multi-stage frontend `Dockerfile` (Node.js 18-alpine, Next.js standalone output).
  - `docker-compose.yml` orchestrating FastAPI backend, Next.js frontend, and health check endpoints.

### U. Phase 35: Continuous Integration & Quality Gates
- **Automated CI/CD Workflow (`.github/workflows/ci.yml`)**:
  - Automated pre-commit quality gates with Ruff linting (`ruff check .`) and formatting (`ruff format --check .`).
  - Pytest full matrix run ensuring 100% test coverage across all packages.
  - Next.js production build verification (`npm run build`).

### V. Phase 36: Capstone Integration & System Verification
- **System-Wide Verification Engine (`packages/core/capstone_audit.py`)**:
  - 10-vector automated audit covering Hardware, Tokenizer, Transformer, Training, Providers, RAG, Agents, Quantization, Telemetry, and DB Integrity.
  - Interactive CLI verification tool (`scripts/capstone_audit_cli.py`).
  - REST endpoint: `POST /api/v1/capstone/audit`.

### W. Phase 37: Security Hardening & Adversarial Robustness
- **Prompt Injection & Adversarial Guard (`packages/core/security/prompt_guard.py`)**:
  - Heuristic and regex detector for direct instruction suppression ("ignore previous instructions", system prompt override).
  - Jailbreak persona detection (DAN, Developer Mode, hypothetical evil twin bypasses).
  - Delimiter spoofing prevention (ChatML `<|im_start|>`, Llama `[INST]`, `<<SYS>>`).
  - GCG adversarial suffix noise anomaly detection via non-alphanumeric token density and repetitive punctuation.
  - Canary token generator and leak detection (`CANARY-UUID`).
  - Untrusted context framing for RAG pipelines (`<untrusted_context>`).
- **Sensitive Credential & Secret Scanner DLP (`packages/core/security/secret_scanner.py`)**:
  - Pattern-based detection for Google Gemini, OpenAI, Anthropic, AWS, GitHub, JWT, and SSH private keys.
  - Shannon entropy calculator $H(s) = -\sum p_i \log_2 p_i$ flagging unstructured high-entropy credentials ($H \ge 3.8$).
  - In-place redaction engine preventing sensitive leaks in responses, logs, and tracebacks.
- **Rate Limiting & OWASP Middleware (`packages/core/security/rate_limiter.py` & `apps/backend/middleware/security.py`)**:
  - In-memory thread-safe `TokenBucketRateLimiter` per client IP.
  - OWASP headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection: 1; mode=block`, `Referrer-Policy`).
  - Payload size limiter blocking payloads $> 10$ MB.
- **REST Endpoints & Frontend UX (`apps/backend/api/v1/endpoints/security.py` & `apps/frontend/src/components/SecurityInspector.tsx`)**:
  - `POST /api/v1/security/scan`: Full security evaluation returning risk score, reasons, and sanitized text.
  - `POST /api/v1/security/redact`: Sanitizes text by replacing secrets with redacted placeholders.
  - `GET /api/v1/security/stats`: Telemetry on security events, blocked attacks, and active rate limits.
  - `SecurityInspector.tsx`: Interactive Security Lab with attack bench, secret DLP tester, and telemetry monitor.

### X. Phase 38: Observability, Distributed Tracing & OpenTelemetry
- **Distributed Tracing Core (`packages/core/observability/tracer.py`)**:
  - 128-bit `TraceId` and 64-bit `SpanId` conforming to W3C standards.
  - Thread-safe context management with `contextvars` for hierarchical parent-child span nesting.
  - OpenTelemetry span semantics: microsecond-accurate durations, attributes, events, and status codes.
  - W3C `traceparent` parser and serializer (`00-{trace_id}-{span_id}-{flags}`).
  - In-memory bounded `TraceCollector` ring buffer (default 100 traces) for zero-dependency local analysis.
- **Token Velocity & Metrics Engine (`packages/core/observability/metrics.py`)**:
  - Time to First Token (TTFT) in milliseconds.
  - Inter-Token Latency (ITL) distribution: mean, median (p50), and 95th percentile (p95).
  - Autoregressive generation velocity (tokens per second).
  - Rolling request latency percentiles (p50, p95) and error rate tracking.
- **Tracing Middleware (`apps/backend/middleware/tracing.py`)**:
  - Intercepts all incoming HTTP requests, establishes a root span, and attaches OpenTelemetry attributes.
  - Injects `X-Trace-Id` and W3C `Server-Timing: total;dur=...` headers on all responses.
- **Observability REST Endpoints (`apps/backend/api/v1/endpoints/observability.py`)**:
  - `GET /api/v1/observability/traces`: List recent request traces.
  - `GET /api/v1/observability/traces/{trace_id}`: Fetch detailed span hierarchy with timeline offsets for Gantt rendering.
  - `GET /api/v1/observability/metrics`: Summary percentiles and token velocities.
  - `POST /api/v1/observability/traces/clear`: Reset buffer.
- **Frontend Observability & Waterfall UI (`apps/frontend/src/components/ObservabilityView.tsx`)**:
  - Interactive Gantt chart latency waterfall visualization.
  - Span detail drawer displaying OpenTelemetry attributes and events.
  - Real-time telemetry cards (TTFT, TPS, p50/p95 latency).

### Y. Phase 39: High-Throughput Batch Inference & Async Workers
- **Dynamic Sequence Binner (`packages/core/batch/sequence_binner.py`)**:
  - Length-sorted clustering into compact buckets (`BinningStrategy.DYNAMIC_LENGTH`).
  - Length ratio threshold ($\le 2.5$) prevents stranding short prompts with long prompts.
  - Reduces quadratic attention padding waste by 50%–80%.
  - Left-padding tensor formatter with synchronized attention masks and 0-indexed position IDs for decoder autoregression.
- **Asynchronous Worker Queue (`packages/core/batch/worker_queue.py`)**:
  - `BatchJobScheduler` with `asyncio.PriorityQueue`, state machine (`QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED`, `CANCELLED`).
  - Priority dispatching (`HIGH` before `LOW`), cancellation, and subscriber notifications.
  - Capped to 1 concurrent CPU worker to preserve interactive UI responsiveness.
- **Vectorized Batch Generator (`packages/models/inference/batch_generator.py`)**:
  - Reconstructs original sequence ordering from binned completions.
  - Real-time throughput telemetry (tokens/sec), wall-clock time, and FLOP speedup metrics.
- **Batch REST Endpoints (`apps/backend/api/v1/endpoints/batch.py`)**:
  - `POST /api/v1/batch/jobs`: Non-blocking job creation.
  - `GET /api/v1/batch/jobs`: Lists active/completed jobs and queue statistics.
  - `GET /api/v1/batch/jobs/{job_id}`: Full item results and telemetry.
  - `DELETE /api/v1/batch/jobs/{job_id}`: Cancellation endpoint.
  - `GET /api/v1/batch/jobs/{job_id}/stream`: Server-Sent Events (SSE) progress streaming.
  - `POST /api/v1/batch/analyze`: Compares naive vs dynamic length padding waste.
- **Interactive Batch Lab UI (`apps/frontend/src/components/BatchInferenceView.tsx`)**:
  - Job runner with workload presets, live progress bar, throughput tracker, and padding matrix visualizer.

### Z. Phase 40: Multi-Modal Document Understanding & OCR Pipeline (LibraOCR)
- **Document Layout Parser (`packages/core/document/layout_parser.py`)**:
  - Deconstructs pages into `HEADING`, `PARAGRAPH`, `TABLE`, `EQUATION`, and `KEY_VALUE`.
  - Normalized 2D spatial bounding boxes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}] \in [0, 1]^4$ with Area and IoU overlap calculation.
  - Table matrix extraction into `TableGrid` with markdown and CSV serialization.
  - LaTeX equation isolation and parsing.
- **Layout-Aware Semantic Chunker (`packages/core/document/layout_chunker.py`)**:
  - Preserves table rows without splitting; re-injects table headers when an oversized table spans multiple chunks.
  - Heading breadcrumb hierarchy tracking for grounded RAG context.
- **Grounded Document QA Engine (`packages/core/document/document_qa.py`)**:
  - Answers questions over structured documents with exact citations (page number, element ID, table cell reference, and bounding box coordinates).
- **Document OCR Endpoints (`apps/backend/api/v1/endpoints/document_ocr.py`)**:
  - `GET /api/v1/document/presets`: Financial report, research paper, and commercial invoice presets.
  - `POST /api/v1/document/parse`: Full AST layout extraction with bounding boxes.
  - `POST /api/v1/document/chunk`: Layout-aware semantic chunking.
  - `POST /api/v1/document/qa`: Grounded visual question answering.
- **Interactive Document Lab UI (`apps/frontend/src/components/DocumentOCRView.tsx`)**:
  - 2D visual document page canvas with colored bounding boxes.
  - Live inspector and Grounded Q&A Assistant with active citation highlight rings.

### AA. Phase 41: Stateful Code Interpreter & Data Analytics Sandbox (LibraNotebook)
- **Stateful REPL Kernel (`packages/core/notebook/session_kernel.py`)**:
  - Persistent execution memory (`_globals`) across cells with isolated session scoping.
  - AST statement/expression splitter compiling leading statements as `exec` and trailing expression as `eval` for automatic value display.
  - Stream redirection capturing `sys.stdout` and `sys.stderr`.
  - Execution counter tracking (`In [x]`), execution duration in milliseconds, and variable diffs.
- **In-Memory Tabular Engine (`packages/core/notebook/analytics.py`)**:
  - `LibraTable`: Zero-dependency DataFrame supporting `select`, `filter`, `sort_by`, `head`, `tail`.
  - Group-by aggregation engine (`sum`, `mean`, `count`, `min`, `max`, `median`).
  - Five-number percentile summaries and Bessel-corrected sample standard deviation in `describe()`.
  - Multi-format serialization: Markdown, styled HTML, and CSV.
- **Pure-Python SVG Vector Chart Engine (`packages/core/notebook/charts.py`)**:
  - Zero-dependency vector graphics generator (`image/svg+xml`).
  - Chart types: `LibraChart.bar`, `LibraChart.line`, `LibraChart.scatter`, `LibraChart.histogram`.
  - Native Jupyter rich display hooks via `_repr_svg_()` and `_repr_html_()`.
- **AST Security Policy (`packages/core/notebook/security.py`)**:
  - Blocks dynamic execution (`eval`, `exec`), reflection attacks (`__subclasses__`, `__globals__`), and dangerous imports (`os`, `subprocess`, `socket`).
- **Notebook REST Endpoints (`apps/backend/api/v1/endpoints/notebook.py`)**:
  - `POST /api/v1/notebook/sessions`: Create or retrieve session.
  - `GET /api/v1/notebook/sessions`: List active sessions.
  - `GET /api/v1/notebook/sessions/{session_id}`: Inspect session status and namespace.
  - `DELETE /api/v1/notebook/sessions/{session_id}`: Terminate session.
  - `POST /api/v1/notebook/sessions/{session_id}/execute`: Execute code cell in stateful session.
  - `GET /api/v1/notebook/sessions/{session_id}/variables`: Namespace variable inspection.
  - `POST /api/v1/notebook/sessions/{session_id}/reset`: Reset namespace.
  - `GET /api/v1/notebook/presets`: Fetch educational analytics templates.
- **Interactive Notebook Lab UI (`apps/frontend/src/components/NotebookView.tsx`)**:
  - Jupyter-style multi-cell notebook canvas with syntax-accented code cells and markdown cells.
  - Controls: individual cell run (`Shift+Enter`), "Run All", restart kernel, cell reordering (`Up`/`Down`), and delete.
  - Rich output renderer supporting stdout, stderr, rich HTML data tables, and embedded SVG charts.
  - Live Variable Explorer sidebar inspecting active variable types, shapes, and values.

### BB. Phase 42: Long-Context Needle-in-a-Haystack (NIAH) & Dynamic Attention Compaction
- **Dynamic Compacted KV Cache (`packages/models/components/compacted_kv_cache.py`)**:
  - Fixed-budget attention memory management: $B = N_{\text{sink}} + N_{\text{recent}} + N_{\text{heavy}}$.
  - StreamingLLM Attention Sinks: Preserves initial $N_{\text{sink}}$ tokens to prevent Softmax normalization drift.
  - Rolling Recent Window: Preserves trailing $N_{\text{recent}}$ tokens for local syntax and grammar fluency.
  - Heavy-Hitter Oracle (H2O): Accumulates attention scores across layers, retaining top-$k$ influential positions while dropping transient tokens.
  - Memory savings: 60%–85% reduction in KV cache RAM on CPU.
- **Multi-Needle Associative Recall Benchmark (`packages/evaluation/multi_needle.py`)**:
  - Synthesizes complex background distractors and places multiple needles across varying document depths without collision.
  - Evaluates multi-key joint associative recall, relational synthesis, and partial/exact retrieval scoring.
- **Long-Context REST Endpoints (`apps/backend/api/v1/endpoints/long_context.py`)**:
  - `POST /api/v1/long_context/evaluate/needle`: 2D NIAH grid evaluation across context lengths and depths.
  - `POST /api/v1/long_context/evaluate/multi_needle`: Multi-key joint associative recall benchmark.
  - `POST /api/v1/long_context/compaction/simulate`: Autoregressive KV cache compaction simulation over long sequences.
  - `GET /api/v1/long_context/presets`: Pre-configured educational benchmark workloads.
- **Interactive Long-Context UI (`apps/frontend/src/components/LongContextView.tsx`)**:
  - 2D NIAH Heatmap: Interactive accuracy grid with modal trial inspection.
  - KV Cache Compaction Visualizer: Color-coded token tape (Sinks, Heavy Hitters, Recent Window, Evicted).
  - Multi-Needle Recall Console: Interactive test runner with found/missing badges.

---

## 4. Resource Usage & Storage Quota Audit

- **Venv Size**: ~856 MB
- **Frontend node_modules**: ~282 MB
- **Next.js Production Build (`.next`)**: ~104 MB
- **Python & Pytest Caches**: ~199 MB
- **Models & Checkpoints**: 20.08 MB
- **Source Code & Data**: 4.45 MB
- **Total Workspace Footprint**: **1,465.98 MB** (~1.43 GB)
- **15 GB Quota Limit**: 15,360.00 MB
- **Remaining Storage Quota**: **13,894.02 MB** (90.5% free)
- **Total Cost**: **$0 / ₹0** (100% free offline development)
- **Active Git Branch**: `main` synced with `https://github.com/eklavya434/libra.git`
- **Pytest Status**: **431 passed, 0 failed** across all 38 phases
- **Frontend Status**: Next.js 14 production build clean (0 errors, 4/4 static pages)








