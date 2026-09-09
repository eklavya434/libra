# Project Libra: Architectural Whitepaper
### An Educational LLM Laboratory and ChatGPT-Like AI Assistant Engineered from First Principles

**Author:** Antigravity AI & Project Libra Contributors  
**Hardware Baseline:** Everyday Consumer CPU (Intel Core i5-12450H, 16GB RAM, Windows/Linux)  
**Cost Model:** $0.00 / ₹0.00 (Strict Zero-Cost Development)  
**Storage Quota:** < 15 GB Total Footprint (Current Footprint: ~875 MB)  
**Release Milestone:** Phase 36 Master Capstone

---

## Executive Abstract
Modern Artificial Intelligence engineering has become increasingly opaque: massive foundation models, proprietary cloud APIs, and complex deployment stacks obscure the core mathematical, algorithmic, and systems engineering foundations of generative AI.

**Project Libra** bridges this gap. It is an end-to-end conversational AI assistant paired with an educational Large Language Model laboratory, built entirely from scratch in Python and TypeScript. Over 36 rigorous developmental phases, Libra establishes:
1. **Core Modeling from First Principles**: BPE tokenization, Rotary Position Embeddings (RoPE), SwiGLU activations, Grouped-Query Attention (GQA), PagedAttention virtual memory, and Vision-Language multimodal integration (LibraVLM).
2. **Inference Optimization**: O(1) step KV-Cache decoding, Speculative Decoding with verification rejection sampling, and Thompson NFA grammar-constrained decoding.
3. **Information Retrieval & Knowledge Grounding**: In-memory dense vector embeddings, Okapi BM25 sparse inverted indexing, Reciprocal Rank Fusion (RRF), heuristic re-ranking, and Jaccard chunk deduplication.
4. **Agentic Execution & Safety**: AST-whitelisted, process-isolated Python child execution sandbox with kernel memory limits and network blackholing, paired with a ReAct tool orchestration loop.
5. **Training, Alignment & Production Deployment**: ChatML dialogue formatting with prompt loss masking (`-100`), Direct Preference Optimization (DPO), containerized multi-stage Docker builds, GitHub Actions CI, and CPU performance regression gates.

Every component runs locally on consumer CPUs in under 15 minutes per educational training cycle.

---

## Architectural Taxonomy (The 7 Pillars)

### Pillar 1: Foundational Modeling & Multimodality
- **Modern Transformer Architecture**: Implements RMSNorm pre-normalization, SwiGLU activation functions ($x \cdot 	ext{Swish}(W_1 x) \odot W_2 x$), and Grouped-Query Attention (GQA) allowing flexible query-to-key-value head ratios.
- **Rotary Position Embeddings (RoPE)**: Direct complex rotational transformations applied to paired coordinate query and key features, providing natural relative position decay without learnable absolute position embeddings.
- **PagedAttention Virtual Memory**: Emulates OS virtual paging to allocate KV cache tensors in fixed physical blocks, eliminating memory fragmentation and enabling zero-copy shared prefix caching.
- **LibraVLM (Vision-Language Model)**: End-to-end multimodal architecture combining a 2D patch embedder, an MLP/Linear alignment projector, and causal transformer decoding over interleaved `[visual_tokens, text_tokens]`.

### Pillar 2: High-Performance Inference & Grammars
- **Incremental KV Caching**: Generates tokens step-by-step with $O(1)$ key-value updates rather than full quadratic sequence re-computation.
- **Speculative Decoding**: Employs a small, lightweight draft model ($M_{	ext{draft}}$) to propose $K$ candidate tokens, verified in a single batched forward pass by the target model ($M_{	ext{target}}$) with acceptance-rejection sampling.
- **Thompson NFA Regex Grammars**: Incremental prefix validation state machine parsing regular expressions into non-deterministic finite automata for structured output generation.

### Pillar 3: Semantic Routing & Zero-Cost Economics
- **Provider Router**: Abstraction layer seamlessly managing Local CPU models, Ollama inference servers, and mock/free-tier providers.
- **Task Complexity Classifier**: Analyzes prompt syntax and semantics to route queries between local tiny models and frontier reasoning tiers.
- **Cost & Budget Telemetry**: Tracks token volume and economic cost in real-time, enforcing the zero-cost ($0 / ₹0) policy.

### Pillar 4: Stateful Memory & Context Management
- **SQLite Conversation Store**: Thread-safe ACID persistence with Write-Ahead Logging (WAL) and cascading foreign key message history.
- **Sliding-Window Context Manager**: Dynamic token budgeting preserving system instructions while gracefully trimming older dialogue turns to fit CPU attention capacity.

### Pillar 5: Hybrid RAG & Deep Research
- **Dense Vector Retrieval**: Exact cosine similarity over normalized embeddings.
- **Okapi BM25 Lexical Retrieval**: Inverted index computing term frequency saturation and document length normalization.
- **Reciprocal Rank Fusion (RRF)**: Non-parametric rank merging:
  $$	ext{RRF}(d) = \sum_{m \in M} rac{1}{k + r_m(d)}$$
- **Multi-Factor Re-Ranking & Deduplication**: Lexical keyword boost, position decay, and Jaccard chunk deduplication.

### Pillar 6: Sandboxed Tools & Autonomous Agents
- **Process-Isolated Sandbox**: Multi-tier defense-in-depth featuring static AST validation, child process isolation, Windows Job Object memory limits, and network proxy blackholing.
- **ReAct Orchestrator**: Thought-Action-Observation multi-step agent reasoning loop.

### Pillar 7: Training, Alignment & Production Readiness
- **ChatML Packaging & Loss Masking**: Sequences formatted with `<|im_start|>` and `<|im_end|>` markers; loss calculation strictly masks prompt tokens with `-100`.
- **DPO (Direct Preference Optimization)**: Closed-form preference loss optimization bypassing complex RL reward modeling.
- **Automated Regression Gates**: Quantified CPU forward and backward latency checks preventing architectural degradation in CI.

---

## Benchmark & Performance Verification
The master capstone audit verifies all 7 pillars on consumer hardware:
- **Total Audit Latency**: ~320 milliseconds.
- **Tracked Disk Footprint**: 875.85 MB (strictly under 15 GB quota).
- **Unit & Integration Test Suite**: 394 passed tests across 36 phases.
- **CI Pipelines**: Clean linting, automated format checks, pre-commit secret guards.

Project Libra demonstrates that modern Large Language Model engineering can be fully mastered, executed, and understood from first principles at zero financial cost.
