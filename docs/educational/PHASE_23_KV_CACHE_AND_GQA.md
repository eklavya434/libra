# Phase 23: Key-Value (KV) Cache Optimization & Grouped-Query Attention (MQA / GQA)

## 1. WHAT: What Was Built
Phase 23 upgrades Project Libra's modern autoregressive Transformer architecture (`ModernTransformerLM`) with state-of-the-art inference acceleration and memory bandwidth reduction techniques used in production open-weights models like Llama 3, Mistral, and Gemma:

1. **Dynamic Key-Value (KV) Cache (`KVCache`)**:
   - Stores past Key ($K$) and Value ($V$) projection states across transformer blocks.
   - Converts per-token decode step time complexity from quadratic $O(T^2)$ recomputation down to linear $O(T)$ cumulative ($O(1)$ operations per decoding step).
   - Dynamic tensor concatenation across decoding steps with precise memory tracking and zero-overhead cache flushing (`reset()`).

2. **Grouped-Query Attention (GQA) & Multi-Query Attention (MQA)**:
   - Configurable $n_{\text{kv\_heads}}$ ($1 \le n_{\text{kv\_heads}} \le n_{\text{heads}}$), allowing query heads to share key/value projections.
   - When $n_{\text{kv\_heads}} = n_{\text{heads}}$: Standard Multi-Head Attention (MHA).
   - When $1 < n_{\text{kv\_heads}} < n_{\text{heads}}$: Grouped-Query Attention (GQA, e.g. 4:1 or 8:1 ratio).
   - When $n_{\text{kv\_heads}} = 1$: Multi-Query Attention (MQA).
   - Efficient head expansion via `torch.repeat_interleave` during scaled dot-product attention.

3. **Position-Offset Rotary Positional Embeddings (RoPE)**:
   - RoPE frequency matrix slicing adapted for single-token decoding steps (`start_pos : start_pos + T`), preventing positional degradation during cached generation.

4. **Cached Autoregressive Generator (`generate_with_cache`)**:
   - Two-phase execution: prompt prefill pass ($T_{\text{prompt}}$) followed by single-token decode passes ($T=1$), yielding exact mathematical token-for-token equivalence to cacheless generation while slashing CPU latency.

5. **REST API & Telemetry**:
   - `POST /api/v1/attention/benchmark`: Live benchmark comparing latency, memory consumption, speedup, and token equivalence between MHA, GQA, and MQA.
   - `POST /api/v1/attention/generate`: Autoregressive generation endpoint leveraging KV cache.

---

## 2. WHY: Why It Exists & The Math Behind It

### The Arithmetic Intensity Bottleneck in LLM Generation
During training or prompt prefill, transformers process $T$ tokens in parallel ($B \times T \times d_{\text{model}}$ matrix multiplications), achieving high arithmetic intensity (flops per byte loaded from RAM):
$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Memory Access (Bytes)}}$$

During autoregressive token generation, however, each step evaluates a single token ($T=1$). Without caching:
1. Generating token $t$ requires computing $Q, K, V$ for all preceding tokens $1 \dots t-1$.
2. Total computation for generating $N$ tokens scales quadratically:
$$\sum_{t=1}^N O(t) = O(N^2)$$

### How KV Cache Achieves $O(1)$ per Decode Step
Past tokens $1 \dots t-1$ never change. Their $K$ and $V$ vectors are identical across subsequent forward passes:
- In step $t$, compute $q_t, k_t, v_t$ only for the current token ($T=1$).
- Append $k_t, v_t$ to the cache:
$$K_{1:t} = [K_{1:t-1} \,;\, k_t], \quad V_{1:t} = [V_{1:t-1} \,;\, v_t]$$
- Compute attention scores between the single query $q_t$ and cached keys $K_{1:t}$:
$$\text{Attention}(q_t, K_{1:t}, V_{1:t}) = \text{softmax}\left(\frac{q_t K_{1:t}^\top}{\sqrt{d_k}}\right) V_{1:t}$$
Decoding complexity for token $t$ drops to $O(1)$ model projections, scaling linearly $O(N)$ for total sequence generation.

### Why MHA Memory Explodes & Why GQA Solves It
In standard Multi-Head Attention (MHA), each query head has a dedicated key and value head. For large models or long contexts:
$$\text{KV Cache Size (Bytes)} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{seq\_len} \times \text{sizeof(precision)}$$

For a model with 32 layers, 32 heads, $d_{\text{head}}=128$, and 2048 sequence length (FP16):
- **MHA ($n_{\text{kv}} = 32$)**: $2 \times 32 \times 32 \times 128 \times 2048 \times 2 \text{ bytes} = 1024 \text{ MB (1.0 GB per stream)}$
- **GQA-8 ($n_{\text{kv}} = 8$)**: $2 \times 32 \times 8 \times 128 \times 2048 \times 2 \text{ bytes} = 256 \text{ MB (4x reduction)}$
- **MQA ($n_{\text{kv}} = 1$)**: $2 \times 32 \times 1 \times 128 \times 2048 \times 2 \text{ bytes} = 32 \text{ MB (32x reduction)}$

GQA provides the optimal Pareto frontier: 99%+ of MHA's representational modeling capacity while slashing memory bandwidth bottlenecks by 4x to 8x.

---

## 3. HOW: Implementation Mechanics

### GQA Head Expansion
In `ModernCausalAttention`:
```python
# Project x into queries (n_heads), keys & values (n_kv_heads)
q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

# Dynamic KV Cache update
if kv_cache is not None:
    k, v = kv_cache.update(k, v, layer_idx=layer_idx)
    total_seq_len = k.size(2)
else:
    total_seq_len = T

# Expand KV heads to match Q heads
if self.num_queries_per_kv > 1:
    k = torch.repeat_interleave(k, repeats=self.num_queries_per_kv, dim=1)
    v = torch.repeat_interleave(v, repeats=self.num_queries_per_kv, dim=1)
```

### Offset Rotary Positional Embeddings
In `RotaryEmbedding`:
```python
def forward(self, q, k, seq_len=None, start_pos=0):
    # Slice sinusoidal frequencies starting at start_pos
    cos = self.cos_cached[:, :, start_pos : start_pos + seq_len, :]
    sin = self.sin_cached[:, :, start_pos : start_pos + seq_len, :]
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
```

---

## 4. TEST: Verification & Results

All 277 tests across the entire repository pass with zero errors:
```bash
pytest -v tests/
# Result: 277 passed, 2 warnings in 22.21s
```

Key verified test suites:
- `tests/models/test_kv_cache.py`: Cache initialization, state concatenation across prefill/decode steps, memory accounting, and reset.
- `tests/models/test_gqa.py`: Config divisibility constraints, projection dimension reduction ($W_k, W_v$), and exact mathematical equivalence between cached and cacheless autoregressive generation.
- `tests/api/test_attention_endpoint.py`: Live FastAPI testing for `/api/v1/attention/benchmark` and `/api/v1/attention/generate`.
- `scripts/run_phase23_kv_cache_demo.py`: Verified $O(1)$ decode acceleration and 100% token equivalence.

---

## 5. NEXT: Phase 24 Preview

With Phase 23 complete, Project Libra has implemented first-principles KV caching and Grouped-Query Attention.

Next Milestone: **Phase 24: Quantization (INT8 / INT4 & Post-Training Quantization)**
- First-principles symmetric & asymmetric affine quantization ($X_q = \text{round}(X / S) + Z$).
- Weight-only INT8 and INT4 quantization for linear projection layers.
- Activation quantization and per-channel scaling factors.
- Benchmarking model memory footprint reduction and perplexity degradation on consumer CPU hardware.
