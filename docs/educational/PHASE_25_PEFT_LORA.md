# Phase 25: Parameter-Efficient Fine-Tuning (PEFT & LoRA from First Principles)

## 1. WHAT: What Was Built
Phase 25 implements Low-Rank Adaptation (LoRA) and Parameter-Efficient Fine-Tuning (PEFT) completely from scratch in Project Libra. Rather than treating parameter efficiency as a third-party black box (`peft`, `bitsandbytes`), every equation—from low-rank matrix decomposition to zero-latency weight folding—is explicitly coded, mathematically verified, and benchmarked on consumer CPU:

1. **First-Principles Low-Rank Linear Layer (`packages/models/lora/lora_linear.py`)**:
   - `LoRALinear`: Drop-in replacement for `torch.nn.Linear`.
   - Freezes original base model weights: $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ (`requires_grad = False`).
   - Injects low-rank trainable decomposition matrices:
     - $A \in \mathbb{R}^{r \times d_{\text{in}}}$ initialized with Kaiming uniform.
     - $B \in \mathbb{R}^{d_{\text{out}} \times r}$ initialized to **zeros**, ensuring that $\Delta W = B \cdot A = 0$ at initialization (100% exact numerical identity with the base model before training starts).
   - Scaling factor: $\text{scaling} = \frac{\alpha}{r}$.
   - **Dynamic Weight Folding (`merge_weights()`)**:
     $$W_{\text{merged}} = W_0 + \frac{\alpha}{r} (B \cdot A)$$
     Achieves exact zero-latency overhead during inference deployment.
   - **Weight Unfolding (`unmerge_weights()`)**:
     $$W_0 = W_{\text{merged}} - \frac{\alpha}{r} (B \cdot A)$$
     Permits continuing training or swapping multi-task adapters dynamically.

2. **Model-Level PEFT Injector & Adapter Management (`packages/models/lora/lora_model.py`)**:
   - `apply_lora`: Freezes base parameters and recursively swaps target projection layers (e.g. `q_proj`, `v_proj`, `out_proj`) with `LoRALinear`.
   - `get_lora_parameter_summary`: Audits trainable vs frozen parameter counts, verifying that **less than 1%** of parameters require gradients.
   - `save_lora_adapter`: Serializes **only** the low-rank tensors ($A$, $B$) and config metadata into lightweight files (~few kilobytes), saving 99%+ disk storage relative to full model checkpoints.
   - `load_lora_adapter`: Restores adapter tensors into matching `LoRALinear` layers with strict key validation.
   - `merge_lora_weights` & `unmerge_lora_weights`: Whole-model adapter merging/unmerging.

3. **Lightweight CPU LoRA Trainer (`packages/training/lora_trainer.py`)**:
   - Optimizes exclusively parameters with `requires_grad = True` using AdamW and gradient clipping.
   - Drastically cuts optimizer state memory footprint ($>95\%$ reduction), allowing fine-tuning on consumer laptops in under 1 second per epoch.

4. **REST API Endpoints (`apps/backend/api/v1/endpoints/peft.py`)**:
   - `POST /api/v1/peft/apply`: Applies LoRA to a model configuration and audits parameter reduction ratios.
   - `POST /api/v1/peft/train_step`: Executes single-step fine-tuning and returns loss and gradient norms.
   - `POST /api/v1/peft/merge`: Validates numerical equivalence between unmerged and merged weights.

---

## 2. WHY: Why It Exists & The Math Behind It

### The Intrinsic Dimension Hypothesis in LLMs
Aghajanyan et al. (2020) and Hu et al. (2021) demonstrated that the parameter changes $\Delta W$ needed to adapt a pre-trained LLM to a specific downstream task have a very low **intrinsic dimension** (rank $r \ll \min(d_{\text{in}}, d_{\text{out}})$).

For a dense linear layer with $d_{\text{in}} = d_{\text{out}} = 4096$:
- Full weight matrix $W \in \mathbb{R}^{4096 \times 4096}$: **16,777,216 parameters** (~67.1 MB in FP32).
- Full AdamW optimizer state (momentum + variance): **33,554,432 floats** (~134.2 MB).
- For a 32-layer transformer, optimizer states require tens of gigabytes of RAM during full fine-tuning.

### Mathematical Formulation of Low-Rank Adaptation
Instead of updating $W_0$ directly, LoRA constrains the update by parameterizing $\Delta W$ as the product of two low-rank matrices:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$

For rank $r = 8$:
- Matrix $A \in \mathbb{R}^{8 \times 4096}$: **32,768 parameters**.
- Matrix $B \in \mathbb{R}^{4096 \times 8}$: **32,768 parameters**.
- Total LoRA parameters: **65,536 parameters** (~262 KB in FP32).
- **Parameter Reduction**:
$$\frac{16,777,216}{65,536} = 256\times \text{ fewer parameters to train!}$$

### Why Initialize $B = 0$ and $A \sim \mathcal{N}(0, \sigma^2)$?
- If both $A$ and $B$ were initialized with random noise, $\Delta W = B A \neq 0$ at the start of training. The pre-trained capabilities would immediately be degraded by random perturbation.
- Initializing $B = 0$ guarantees that at $t = 0$:
$$\Delta W = 0 \cdot A = 0 \implies W = W_0 + 0 = W_0$$
The model starts exactly at the pre-trained baseline and gradually learns task-specific adaptations.

### The Scaling Hyperparameter $\alpha$
The scaling factor $\frac{\alpha}{r}$ stabilizes training when experimenting with different values of rank $r$. When increasing $r$, the magnitude of $\Delta W$ remains consistent without needing to retune the learning rate.

---

## 3. HOW: Implementation Mechanics

### Forward Pass in `LoRALinear`
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.merged or self.rank == 0:
        return F.linear(x, self.weight, self.bias)

    # 1. Base forward pass (frozen)
    base_out = F.linear(x, self.weight, self.bias)

    # 2. Low-rank adapter path: (x @ A.T) @ B.T * (alpha / r)
    lora_x = self.lora_dropout(x)
    lora_out = (lora_x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling

    return base_out + lora_out
```

### Zero-Overhead Weight Merging
```python
def merge_weights(self) -> None:
    if self.rank > 0 and not self.merged:
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        self.weight.data.add_(delta_w)
        self.merged = True
```

---

## 4. TEST: Verification Results

All **295 tests** across Project Libra pass with zero errors:
```bash
pytest -v tests/
# Result: 295 passed, 2 warnings in 36.37s
```

### Live Benchmark Demo Output (`scripts/run_phase25_lora_demo.py`)
```text
======================================================================
  1. Parameter Efficiency & Freezing Mechanics
======================================================================
Base Transformer Configuration:
  - Layers: 6 | d_model: 256 | Heads: 8
  - Full Parameters: 5,332,224

After Applying LoRA (r=8, alpha=16.0):
  - Trainable Parameters: 49,152 (0.91%)
  - Frozen Parameters:    5,332,224
  - Parameter Reduction:  108.5x fewer trainable weights!

======================================================================
  2. First-Principles LoRA Fine-Tuning Convergence
======================================================================
Training LoRA adapter over 5 epochs on CPU...
  Epoch 1: Avg Loss = 4.6222
  Epoch 2: Avg Loss = 4.5772
  Epoch 3: Avg Loss = 4.5383
  Epoch 4: Avg Loss = 4.5074
  Epoch 5: Avg Loss = 4.4850

Completed in 0.64 seconds (0.128 s/epoch).

======================================================================
  3. Zero-Overhead Inference: Weight Merging Equivalence
======================================================================
Pre-Merge Output Logits (Slice):  [0.10405, -0.30589, 0.21825, 0.00489, -0.19920]
Post-Merge Output Logits (Slice): [0.10405, -0.30589, 0.21825, 0.00489, -0.19920]
Max Absolute Discrepancy:         0.00000000e+00
Mathematical Equivalence:         SUCCESS (Zero Divergence)

Phase 25 PEFT & LoRA verification demo completed successfully.
```

---

## 5. NEXT: Phase 26 Preview

With Phase 25 complete, Project Libra has mastered low-rank adaptation, adapter checkpointing, and dynamic zero-latency weight merging.

Next Milestone: **Phase 26: Streaming Token Telemetry & Token-Level Metrics**
- Token-level confidence and perplexity tracking: $p(w_t \mid w_{<t})$ and surprisal $-\log_2 p(w_t \mid w_{<t})$.
- Server-Sent Events (SSE) streaming with per-token metadata (token latency in ms, top-k candidate distribution).
- Frontend visual inspection: interactive token surprisal heatmaps (color-coded tokens by confidence).
