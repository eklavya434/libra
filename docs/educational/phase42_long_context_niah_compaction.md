# Phase 42: Long-Context Needle-in-a-Haystack (NIAH) & Dynamic Attention Compaction

> **Project Libra — First-Principles Educational LLM Laboratory**
> *Dual-Track Architecture: Interactive AI Product & First-Principles Foundation Lab*

---

## 1. WHAT: System Architecture & Deliverables

Phase 42 engineers **LibraLongContext**, a comprehensive long-context evaluation and dynamic attention compaction engine built from first principles for zero-cost CPU execution ($0 / ₹0).

### Core Components Delivered:
1. **Dynamic Compacted KV Cache (`CompactedKVCache` in `packages/models/components/compacted_kv_cache.py`)**:
   - Fixed-budget attention memory management: $B = N_{\text{sink}} + N_{\text{recent}} + N_{\text{heavy}}$.
   - **StreamingLLM Attention Sinks**: Locks initial $N_{\text{sink}}$ positions ($0 \dots N_{\text{sink}}-1$) in cache indefinitely, stabilizing the Softmax denominator without drift.
   - **Rolling Recent Window**: Retains trailing $N_{\text{recent}}$ positions for grammatical fluency and local syntactic coherence.
   - **Heavy-Hitter Oracle (H2O)**: Dynamically accumulates attention query-key product importances $\sum_{t} \alpha_{i, t}$ across layers, evicting transient noise tokens while keeping the top-$k$ influential prompt positions.
   - Compaction telemetry: active cached tokens, total tokens ingested, eviction count, memory bytes, uncompressed baseline comparison, and memory savings percentage ($60\% - 85\%$).

2. **Multi-Needle Associative Recall Benchmark (`MultiNeedleEvaluator` in `packages/evaluation/multi_needle.py`)**:
   - Synthesizes long-context technical distractors and injects multiple distinct factoids at independent sorted depth percentiles without collisions.
   - Evaluates multi-key joint extraction, relational synthesis, and associative recall across various document lengths.
   - Telemetry: exact match, partial accuracy score, per-needle latency, and prompt token metrics.

3. **FastAPI Endpoints (`apps/backend/api/v1/endpoints/long_context.py`)**:
   - `POST /api/v1/long_context/evaluate/needle`: 2D NIAH grid evaluation across context lengths and depths.
   - `POST /api/v1/long_context/evaluate/multi_needle`: Multi-key joint associative recall benchmark.
   - `POST /api/v1/long_context/compaction/simulate`: Autoregressive KV cache compaction simulation over long sequences.
   - `GET /api/v1/long_context/presets`: Pre-configured educational benchmark templates (Vault PIN, Multi-Key Silo Operations).

4. **Interactive Long-Context UI (`apps/frontend/src/components/LongContextView.tsx`)**:
   - **2D NIAH Heatmap**: Color-coded accuracy grid (Lengths vs Depths) with click-to-inspect modal displaying prompt, target keys, retrieved text, and millisecond latency.
   - **KV Cache Compaction Visualizer**: Color-coded token tape showing Attention Sinks (blue), Heavy Hitters (purple), Recent Window (emerald), and Evicted Tokens (gray).
   - **Multi-Needle Recall Console**: Interactive runner for multi-key extraction with found/missing badges.
   - Integrated into [`Sidebar.tsx`](file:///c:/Users/eklav/Desktop/Libra/apps/frontend/src/components/Sidebar.tsx) and rendered in [`page.tsx`](file:///c:/Users/eklav/Desktop/Libra/apps/frontend/src/app/page.tsx).

---

## 2. WHY: Problem Space & Theoretical Motivation

### The KV Cache Memory Explosion
In autoregressive transformer generation, caching Keys and Values is required to avoid recomputing past representations ($O(T^2) \to O(T)$ cumulative time). However, memory consumption scales linearly with sequence length:
$$\text{Memory}_{\text{KV}} = 2 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times L \times d_{\text{head}} \times \text{bytes\_per\_elem}$$
For a long sequence ($L = 8,192$ or $32,768$ tokens), memory explodes into gigabytes, choking CPU memory bandwidth and causing cache misses.

### The Attention Sink Phenomenon
Research (StreamingLLM, Xiao et al., 2023) discovered that transformers dedicate an outsized portion of attention weights to the very first few tokens ($0 \dots 3$) regardless of their semantic value, purely as an artifact of Softmax normalization. If those first tokens are evicted (as in naive sliding-window attention), the model's perplexity catastrophically collapses.

### Heavy-Hitter Sparsity (H2O)
Not all intermediate tokens are equally important. Only a small fraction of prompt tokens ("heavy hitters") receive persistent cross-attention from downstream queries. By identifying and retaining these high-impact tokens alongside attention sinks and the recent window, memory is strictly bounded ($O(B)$) with negligible degradation in associative recall.

---

## 3. HOW: Mathematical & Algorithmic Implementation

### A. Attention Compaction Formulation
Given total sequence length $T$ and maximum budget $B$:
1. **Sink Token Set**:
   $$S = \{0, 1, \dots, N_{\text{sink}} - 1\}$$
2. **Recent Token Set**:
   $$R = \{T - N_{\text{recent}}, T - N_{\text{recent}} + 1, \dots, T - 1\}$$
3. **Candidate Middle Positions**:
   $$M = \{N_{\text{sink}}, N_{\text{sink}} + 1, \dots, T - N_{\text{recent}} - 1\}$$
4. **Heavy-Hitter Selection**:
   For each candidate token $j \in M$, accumulate attention importance:
   $$I(j) = \sum_{q} \sum_{h=1}^{H} A_{h, q, j}$$
   Select top $k = \min(|M|, B - N_{\text{sink}} - N_{\text{recent}})$ positions:
   $$H = \operatorname{argtopk}_{j \in M}(I(j), k)$$
5. **Retained Token Mask**:
   $$\Omega = \operatorname{sort}(S \cup H \cup R)$$
The cached key and value tensors are sliced along the sequence dimension:
$$K_{\text{compacted}} = K[:, :, \Omega, :], \quad V_{\text{compacted}} = V[:, :, \Omega, :]$$

---

## 4. TEST: Verification & Regression Results

1. **Phase 42 Unit & API Tests**:
   - `tests/models/test_compacted_kv_cache.py`: Verified budget constraints, sink token protection, heavy-hitter retention via attention scores, and memory calculation (4 tests).
   - `tests/evaluation/test_multi_needle.py`: Verified multi-needle distractor synthesis, collision-free ordering, partial/exact scoring, and single/grid runner (3 tests).
   - `tests/api/test_long_context_endpoint.py`: Verified presets, single-needle 2D NIAH grid, multi-needle API, and compaction simulation (4 tests).
   - **All 11 Phase 42 tests passed in 10.62s**.

2. **Full Regression Test Suite**:
   - **480 passed, 0 failed** across all 42 phases.

3. **Frontend Production Build**:
   - Next.js 14 production build compiled cleanly with zero errors.

---

## 5. NEXT: Upcoming Roadmap

With Phase 42 complete, Project Libra possesses end-to-end long-context evaluation and memory-bounded KV compaction.
The curriculum transitions to:
- **Phase 43: Reinforcement Learning via Self-Play & Monte Carlo Tree Search (MCTS / LibraReason)**:
  - Step-by-step reasoning verification with tree search.
  - Process Reward Models (PRMs) scoring intermediate reasoning steps.
  - Test-time compute scaling across branching exploration paths.
