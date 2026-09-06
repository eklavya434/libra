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
| **Phase 9** | External Provider Adapters | ⏳ NEXT | OpenAI, Gemini, Claude, Groq with offline fallback |

---

## 3. Phase 8 Outcome & Local Inference Status

### A. Ollama Status
- **Adapter**: Implemented in `packages/providers/ollama.py` (Async REST/SSE client).
- **Daemon Status**: Offline (`http://127.0.0.1:11434` currently unreachable; user can run `ollama serve` or desktop app when desired).
- **Currently Running Models via Ollama**: None active at this moment because daemon is stopped.
- **Recommended Models for Ollama on this Machine**:
  - `llama3.2:1b` (0.8 GB RAM footprint)
  - `deepseek-r1:1.5b` (1.1 GB RAM footprint — reasoning model)
  - `qwen2.5:1.5b` (1.0 GB RAM footprint — coding/multilingual)
  - `llama3.2:3b` (2.1 GB RAM footprint)
- **Fallback Behavior**: When Ollama daemon is offline, router gracefully defaults to `libra_lab` (local PyTorch model) or `mock-provider` with informative guidance.

### B. Hugging Face Transformers Status
- **Adapter**: Implemented in `packages/providers/huggingface.py` (`HuggingFaceProvider`).
- **Dependencies**: `transformers==5.16.1` installed in `.venv`.
- **Capability**: Loads and generates from Hugging Face models on CPU with sampling controls.

### C. Local Educational Model & Checkpoint State
- **Active Checkpoint**: `checkpoints/best_engine_model.pt` (6.29 MB) and `checkpoints/modern_transformer_phase4.pt` (2.54 MB).
- **Architecture**: `ModernTransformerLM` (467,584 parameters, weight-tied, RoPE, RMSNorm, SwiGLU).
- **Tokenizer**: `EducationalBPETokenizer` loaded from `data/tokenized/libra_educational_bpe.json` (vocab size: 300).
- **Inference Speed**: **349.5 tokens/sec** on Intel Core i5-12450H CPU.
- **Serving Provider**: `LocalTransformerProvider` (`libra_lab`) serving chat and streaming completions.

### D. vLLM Status
- **Status**: **NOT INSTALLED** on this machine in strict accordance with **Directive 6** of `AGENTS.md`.
- **Placeholder**: `packages/providers/vllm_stub.py` defines the future GPU specification and returns clear disablement messages.

### E. Deviations from Plan
- **None**: All deliverables across Ollama, Hugging Face, PyTorch local transformer, prompt templating, and SSE streaming completed with zero architectural shortcuts.

---

## 4. Resource Usage & Storage Quota Audit

- **Venv Size**: 855.62 MB
- **Frontend node_modules**: 281.61 MB
- **Models & Checkpoints**: 20.08 MB
- **Total Workspace Footprint**: **1,183.99 MB** (~1.18 GB)
- **15 GB Quota Limit**: 15,360.00 MB
- **Remaining Storage Quota**: **14,176.01 MB** (>92% free)
- **Total Cost**: **$0 / ₹0** (100% free offline development)
- **Active Git Branch**: `main` synced with `https://github.com/eklavya434/libra.git`
- **Pytest Status**: **59 passed, 0 failed** (in 14.91s)
