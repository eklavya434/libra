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
| **Phase 9** | External Provider Adapters | ✅ COMPLETE | OpenAI, Gemini, Claude, DeepSeek, Groq, OpenRouter, cost tracker, error normalization |
| **Phase 10** | Model Comparison Arena | ⏳ NEXT | Side-by-side completions, latency/cost benchmarking |

---

## 3. Phase 8 & 9 Provider Ecosystem Status

### A. Phase 9 Outcome: Provider Abstraction & Cost Economics
- **Providers Implemented**: 11 total adapters:
  1. `ollama`: Ollama local CPU runtime
  2. `libra_lab`: PyTorch `ModernTransformerLM` checkpoint serving
  3. `huggingface`: Hugging Face `transformers` CPU pipeline
  4. `mock-provider`: Deterministic zero-cost offline test provider
  5. `openai`: Official OpenAI GPT models (GPT-4o, GPT-4o-mini)
  6. `gemini`: Google Gemini REST and streaming
  7. `anthropic`: Anthropic Claude Messages API
  8. `deepseek`: DeepSeek-V3 & DeepSeek-R1 reasoning models
  9. `groq`: Groq Cloud LPU ultra-low latency inference
  10. `openrouter`: OpenRouter multi-model aggregation gateway
  11. `vllm`: Future GPU-only architecture stub (Directive 6)
- **Token Economics Engine**: `packages/providers/cost.py` pricing tables for all major commercial models with strict zero-cost guarantee ($0.000000) for local and mock execution.
- **Normalized Error Hierarchy**: `packages/providers/errors.py` unifying 401, 403, 404, 429, 500 across providers.
- **Intelligent Fallback**: Automatic routing with graceful offline fallback to `mock-provider` or local models when API keys are unconfigured.

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
- **Total Workspace Footprint**: **1,184.38 MB** (~1.18 GB)
- **15 GB Quota Limit**: 15,360.00 MB
- **Remaining Storage Quota**: **14,175.62 MB** (92.29% free)
- **Total Cost**: **$0 / ₹0** (100% free offline development)
- **Active Git Branch**: `main` synced with `https://github.com/eklavya434/libra.git`
- **Pytest Status**: **72 passed, 0 failed** (in 25.42s)

