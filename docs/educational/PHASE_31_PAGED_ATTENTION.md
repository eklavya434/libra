# Educational Guide: Phase 31 — Advanced Inference Optimization (PagedAttention & Continuous Batching)

Welcome to **Phase 31** of Project Libra! In this phase, we address the primary bottleneck of Large Language Model deployment and high-concurrency serving: **Key-Value (KV) Cache Memory Fragmentation & Static Batching Waste**.

---

## 1. WHAT: The Memory Bottleneck in LLM Serving

When generating tokens autoregressively, a transformer model saves previous Keys and Values in memory to avoid recalculating past token representations (converting per-token decoding complexity from $O(T)$ down to $O(1)$).

However, in traditional systems, the KV cache is managed as a contiguous tensor. This introduces two catastrophic inefficiencies:
1. **Internal Fragmentation (Over-Allocation)**: A system must pre-allocate contiguous memory for the maximum possible sequence length (e.g. 2,048 or 32,768 tokens) because dynamic contiguous array growth (`torch.cat`) re-allocates memory and causes memory copying overhead. When requests only generate 50 or 200 tokens, $60\% - 80\%$ of the reserved memory sits completely unused.
2. **External Fragmentation**: Contiguous allocations require contiguous physical address space. Even if total free memory is 4 GB, if it is scattered in small chunks, a new request requiring 1 GB of contiguous memory will trigger an Out-Of-Memory (OOM) crash.
3. **Static Batching ("Long-Tail Curse")**: In traditional batching, all requests in a batch run together. If one request outputs 500 tokens while three others output 20 tokens, the short requests finish early but the compute units must continue running padded iterations until the slowest request finishes.

Phase 31 implements:
- **PagedAttention** (Kwon et al., 2023, vLLM architecture)
- **Continuous (Iteration-Level) Batching** (Yu et al., 2022, Orca architecture)

---

## 2. WHY: Virtual Memory Paging from Operating Systems to LLMs

In traditional operating systems (OS), virtual memory divides physical RAM into fixed-size **page frames** (e.g., 4 KB) and uses a **page table** to map contiguous virtual addresses to non-contiguous physical pages.

PagedAttention brings this exact OS concept to transformer inference:
- The KV cache for a sequence is broken down into **blocks** of fixed size $B$ (default: 16 tokens).
- Keys and values are written into non-contiguous physical block pools.
- A **Block Table** maps logical token indices to physical block IDs.
- Result: **Zero external fragmentation**, and internal fragmentation is bounded to at most $B - 1$ tokens on the very last block ($< 3.5\%$ total memory waste).

---

## 3. HOW: Algorithms & Core Components

### A. Block Allocator & Physical Block Pool (`packages/models/components/paged_cache.py`)
- `PhysicalBlockPool`: Pre-allocates fixed memory tensors:
  $$\text{Shape: } (\text{num\_blocks}, \text{num\_kv\_heads}, \text{block\_size}, \text{head\_dim})$$
- `BlockAllocator`: Uses a free list to allocate and free physical block IDs with reference counting (enabling instant zero-copy prefix sharing).

### B. PagedAttention Decode Kernel (`packages/models/components/paged_attention.py`)
During decoding for a single query token $q_t$:
$$\text{Logical Token } j \implies \text{Block ID } b = \text{Table}[j // B], \quad \text{Offset } o = j \pmod B$$
$$k_j = \text{PhysicalPool}[b, \text{head}, o, :]$$

Attention scores are computed across the resolved blocks without gathering or allocating contiguous tensors:
$$\text{score}(t, j) = \frac{q_t \cdot k_j^T}{\sqrt{d_k}}$$

### C. Continuous (Iteration-Level) Batching (`packages/models/inference/continuous_batching.py`)
Instead of batching at the request level, continuous batching operates at the **iteration level**:
1. **Prefill Interleaving**: Arriving waiting requests have their prompts processed and are admitted whenever free blocks are available.
2. **Synchronous Decode Step**: All currently running requests generate exactly 1 token in parallel.
3. **Immediate Eviction**: Finished sequences release their physical blocks back to the free pool immediately, allowing new requests to begin without waiting for the longest request in the batch.

---

## 4. TEST: Empirical Verification

- **7 Dedicated Tests**: `tests/models/test_paged_attention.py`, `tests/models/test_continuous_batching.py`, `tests/api/test_paged_inference_endpoint.py`.
- **Numerical Parity**: `paged_attention_decode` matches standard unpaged attention to $< 1.2 \times 10^{-7}$ precision.
- **Full Suite**: **362 tests passing, 0 failing**.
- **CLI Demo**: `scripts/run_phase31_paged_attention_demo.py` verifies 1.47x concurrency capacity boost and dynamic request eviction on CPU.

---

## 5. NEXT: Advancing to Phase 32

With PagedAttention and Continuous Batching complete, Libra possesses frontier-grade inference infrastructure.

In **Phase 32**, we explore **Multi-Modal Integration (Vision-Language Adapter: Projection & Cross-Attention)**:
- Building an educational CLIP-style vision encoder and linear projection adapter.
- Feeding image patches alongside text tokens into `ModernTransformerLM`.
