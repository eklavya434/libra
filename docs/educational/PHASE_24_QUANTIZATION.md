# Phase 24: Quantization (INT8 / INT4 & Post-Training Quantization from First Principles)

## 1. WHAT: What Was Built
Phase 24 establishes Project Libra's first-principles quantization and model compression library. Without relying on external black-box libraries (`bitsandbytes`, `autoawq`), every mathematical step—from affine quantization scaling to bit-level nibble packing—is transparent, explainable, and executed natively on CPU:

1. **Quantization Mathematical Primitives (`packages/models/quantization/quant_core.py`)**:
   - **Symmetric Signed Quantization**: Maps $[-q_{\text{max}}, q_{\text{max}}]$ with zero-point $Z = 0$, ideal for zero-centered weight tensors.
   - **Asymmetric Affine Quantization**: Maps arbitrary range $[\alpha, \beta]$ into unsigned range $[0, 2^b - 1]$ with calculated zero-point $Z$.
   - **Per-Tensor and Per-Channel Scaling**: Supports granular per-channel (per-output-neuron) scale vectors.
   - **INT4 Bit Packing & Unpacking**: Packs two 4-bit signed weights into a single `uint8` byte using bitwise shifting and masking (`high << 4 | low`), halving physical memory storage over unpacked INT4.
   - **Fidelity Metrics**: Signal-to-Quantization-Noise Ratio ($\text{SQNR}_{\text{dB}}$), Mean Squared Error ($\text{MSE}$), and Cosine Similarity.

2. **Quantized PyTorch Linear Layers (`packages/models/quantization/quant_linear.py`)**:
   - `QuantizedLinearINT8`: Drop-in replacement for `nn.Linear` storing weights in `torch.int8` with per-channel `torch.float32` scaling factors (4x weight compression).
   - `QuantizedLinearINT4`: Drop-in replacement for `nn.Linear` storing weights in bit-packed `torch.uint8` tensors (8x weight compression).
   - On-the-fly dequantization during matrix multiplications ($Y = X \hat{W}^\top + b$).

3. **Post-Training Quantization (PTQ) Engine (`packages/models/quantization/ptq.py`)**:
   - `quantize_model`: Recursively inspects transformer models (`ModernTransformerLM`, `TinyTransformerLM`) and replaces standard `nn.Linear` layers (`q_proj`, `k_proj`, `v_proj`, `out_proj`, `w_gate`, `w_up`, `w_down`, `output_head`) with their INT8 or INT4 counterparts.
   - `compute_model_memory`: Audits parameter, buffer, and custom quantized weight allocations.
   - `audit_quantization_fidelity`: Quantifies logits MSE, SQNR, cosine similarity, and top-1 token agreement.

4. **REST API Endpoints (`apps/backend/api/v1/endpoints/quantization.py`)**:
   - `POST /api/v1/quantization/benchmark`: Live benchmark comparing memory footprint, latency, and reconstruction fidelity across FP32, INT8, and INT4.
   - `POST /api/v1/quantization/convert`: Converts model configurations and returns memory and fidelity audits.

---

## 2. WHY: Why It Exists & The Math Behind It

### Memory Bandwidth Bottlenecks on Consumer Hardware
During autoregressive generation, a transformer reads weights from RAM once per generated token:
$$\text{Memory Bandwidth Required} = \text{Model Size (Bytes)} \times \text{Tokens per Second}$$

For consumer CPUs (e.g. DDR4/DDR5 with ~30-50 GB/s bandwidth):
- A 7B parameter model in **FP32** requires $7 \times 10^9 \times 4 \text{ bytes} \approx 28 \text{ GB}$ (exceeds 16GB system RAM entirely).
- In **FP16/BF16**, it requires $14 \text{ GB}$.
- In **INT8**, it drops to **7 GB** (50% of 16GB RAM).
- In **INT4**, it drops to **3.5 GB** (running comfortably with ample room for OS, browser, and KV cache).

### Mathematical Derivation of Affine Quantization

#### 1. Symmetric Quantization (Signed INT8 / INT4)
For a tensor $X$ whose values are roughly zero-centered:
$$q_{\text{max}} = 2^{b - 1} - 1, \quad q_{\text{min}} = -2^{b - 1}$$
$$\text{Scale } S = \frac{\max(|X|)}{q_{\text{max}}}$$
$$\text{Quantize: } X_q = \text{clamp}\left(\left\lfloor \frac{X}{S} \right\rceil, q_{\text{min}}, q_{\text{max}}\right)$$
$$\text{Dequantize: } \hat{X} = X_q \cdot S$$

Because $Z = 0$, symmetric matrix multiplication avoids costly zero-point offset corrections:
$$X \hat{W}^\top = X (W_q S_w)^\top = S_w \cdot (X W_q^\top)$$

#### 2. Asymmetric Quantization (Arbitrary Range)
For activations or skewed distributions in $[\alpha, \beta]$:
$$S = \frac{\beta - \alpha}{2^b - 1}, \quad Z = \left\lfloor -\frac{\alpha}{S} \right\rceil$$
$$X_q = \text{clamp}\left(\left\lfloor \frac{X}{S} \right\rceil + Z, 0, 2^b - 1\right)$$
$$\hat{X} = (X_q - Z) \cdot S$$

#### 3. Signal-to-Quantization-Noise Ratio (SQNR)
$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (X_i - \hat{X}_i)^2$$
$$\text{SQNR}_{\text{dB}} = 10 \cdot \log_{10}\left( \frac{\frac{1}{N}\sum X_i^2}{\text{MSE}} \right)$$
- INT8 typically yields **40 to 60 dB SQNR** (virtually indistinguishable from FP32).
- INT4 typically yields **18 to 25 dB SQNR** (acceptable fidelity for large models with residual error compensation).

---

## 3. HOW: Implementation Details

### Bit-Level INT4 Packing
Because PyTorch lacks a native 4-bit integer dtype, two signed 4-bit integers $[-8, 7]$ are packed into one `torch.uint8` [0, 255]:
```python
# Packing:
unsigned_nibbles = (int4_tensor & 0x0F).to(torch.uint8)
low = unsigned_nibbles[..., 0::2]
high = unsigned_nibbles[..., 1::2]
packed = (high << 4) | low  # 1 byte stores 2 weights

# Unpacking:
low = (packed & 0x0F)
high = (packed >> 4) & 0x0F
# Restore signed 2's complement:
low = torch.where(low >= 8, low - 16, low)
high = torch.where(high >= 8, high - 16, high)
```

---

## 4. TEST: Verification Results

All **286 tests** across Project Libra pass with zero errors:
```bash
pytest -v tests/
# Result: 286 passed, 2 warnings in 22.74s
```

### Live Benchmark Demo Results (`scripts/run_phase24_quantization_demo.py`)
```text
======================================================================
  1. Tensor Quantization Arithmetic & Bit Packing
======================================================================
Quantization Scheme          | Bits  | Packed Shape   | SQNR (dB)  | Cosine Sim
---------------------------------------------------------------------------
Symmetric INT8               | 8-bit | (4, 16)        | 46.36      | 0.999989  
Symmetric INT4 (Packed)      | 4-bit | (4, 8)         | 21.26      | 0.996291  
Asymmetric INT8              | 8-bit | (4, 16)        | 57.10      | 0.999999  

======================================================================
  2. Post-Training Quantization on ModernTransformerLM
======================================================================
Precision Tier   | Memory (MB)  | Savings    | Cosine Sim  | Top-1 Match | Latency  
--------------------------------------------------------------------------------
FP32 Baseline    | 4.805        | Baseline   | 1.000000    | 100.0%      | 4.00   ms
INT8 Quantized   | 2.348        | 51.14%     | 0.999869    | 100.0%      | 2.74   ms
INT4 Quantized   | 1.934        | 59.75%     | 0.968060    | 62.5%       | 8.28   ms
```

---

## 5. NEXT: Phase 25 Preview

With Phase 24 complete, Project Libra has mastered post-training quantization, weight packing, and affine mathematics.

Next Milestone: **Phase 25: Parameter-Efficient Fine-Tuning (PEFT) & LoRA (Low-Rank Adaptation)**
- First-principles low-rank matrix decomposition: $W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$, where $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$ with $r \ll \min(d, k)$.
- Frozen base model weights with trainable LoRA adapters.
- Adapter serialization, loading, and dynamic merging ($W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA$).
