# Educational Guide: Phase 4 — Improved Modern Transformer

Welcome to **Phase 4** of **Libra**!

In this phase, we transitioned our neural network architecture from the classic 2017 Transformer (GPT-2 style) to the **modern architectural paradigm** powering current open-weight models like **Llama 3**, **Mistral**, **DeepSeek**, and **Gemma**.

---

## 1. WHAT Was Built?

1. **Rotary Positional Embeddings (RoPE)** (`packages/models/components/rope.py`): Encodes position by rotating Query and Key vectors in 2D coordinate pairs.
2. **Root Mean Square Normalization (RMSNorm)** (`packages/models/components/rmsnorm.py`): Replaces standard LayerNorm by scaling activations by their root mean square without subtracting the mean.
3. **SwiGLU Gated Feed-Forward Network** (`packages/models/components/swiglu.py`): Replaces standard MLP with a Swish-gated activation layer.
4. **Weight Tying** (`packages/models/modern_transformer.py`): Reuses the token embedding matrix as the output classification head ($W_{out} = W_{emb}^T$), cutting parameters by ~25%.
5. **Declarative YAML Configurations** (`configs/models/`): Instantiating architectures directly from human-readable configuration files.

---

## 2. The 4 Modern Architectural Upgrades

| Component | Classic Transformer (Phase 1 / GPT-2) | Modern Transformer (Phase 4 / Llama 3) | Why the Modern Approach is Superior |
| :--- | :--- | :--- | :--- |
| **Positional Encoding** | Absolute learned lookup table ($T \le 128$) | **Rotary Positional Embedding (RoPE)** | RoPE represents *relative distance* ($m - n$) rather than fixed slots. It allows length extrapolation and naturally decays attention over long contexts. |
| **Normalization** | Standard `LayerNorm` (computes mean and variance) | **`RMSNorm`** | Eliminates mean subtraction. 10%–50% faster computation with identical training stability. |
| **Feed-Forward Activation** | Standard `GELU` or `ReLU` | **`SwiGLU`** | A multiplicative gating mechanism $(\text{Gate} \odot \text{SiLU}(\text{Up})) \text{Down}$ allows the network to dynamically suppress or amplify information. |
| **Output Projection** | Independent linear weight matrix ($d_{model} \times V$) | **Weight Tying** ($W_{out} = W_{emb}^T$) | Ties the token embedding weights to the output head, drastically reducing parameter count and preventing divergence. |

---

## 3. HOW Does the Math Work?

### A. Rotary Positional Embeddings (RoPE)
In standard attention, position is added to the token embedding at the very beginning:
$$\mathbf{x} = \mathbf{e}_{\text{token}} + \mathbf{e}_{\text{position}}$$
This is crude because once mixed, attention cannot easily isolate the position from the semantic meaning.

**RoPE's Insight**: Instead of adding position to embeddings, we rotate the Query and Key vectors by angle $m \theta$ inside the attention calculation:
$$\begin{pmatrix} q'_1 \\ q'_2 \end{pmatrix} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} q_1 \\ q_2 \end{pmatrix}$$

When we take the dot product between Query at position $m$ and Key at position $n$:
$$\langle \mathbf{q}_m, \mathbf{k}_n \rangle = \mathbf{q}^T R_{\Theta, m}^T R_{\Theta, n} \mathbf{k} = \mathbf{q}^T R_{\Theta, n - m} \mathbf{k}$$
The dot product **only depends on the relative distance $(n - m)$**! Words that are 2 tokens apart always interact identically, whether they appear at position 5 or position 5,000.

### B. Root Mean Square Normalization (RMSNorm)
Standard LayerNorm computes:
$$\mu = \frac{1}{d}\sum x_i, \quad \sigma^2 = \frac{1}{d}\sum (x_i - \mu)^2, \quad y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma$$

Researchers discovered that the mean-centering step ($x - \mu$) provides no benefit to training stability. RMSNorm simplifies this to:
$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}, \quad y = \frac{x}{\text{RMS}(x)} \odot \gamma$$
This saves valuable CPU cycles while preserving exact scale invariance.

### C. SwiGLU Gating
In standard MLP:
$$y = \text{GELU}(x W_1) W_2$$

In SwiGLU:
$$\text{Swish}(z) = z \cdot \sigma(z) = \text{SiLU}(z)$$
$$y = (x W_{\text{gate}} \odot \text{SiLU}(x W_{\text{up}})) W_{\text{down}}$$
The gate acts like an adjustable valve, allowing the model to choose which features pass forward.

---

## 4. How We Verified It

We created automated tests in `tests/models/`:
1. `test_rmsnorm.py`: Proved activations scale to unit root mean square and gradients flow smoothly.
2. `test_rope.py`: Proved mathematically that $\langle \mathbf{q}_m, \mathbf{k}_n \rangle == \langle \mathbf{q}_{m+d}, \mathbf{k}_{n+d} \rangle$ for any arbitrary position shift $d$.
3. `test_swiglu.py`: Proved feature dimension expansion and gradient propagation through both gates.
4. `test_modern_transformer.py`: Proved declarative YAML configuration loading and verified parameter savings from weight tying.

---

## 5. NEXT: What Comes Next?

In **Phase 5: Training Engine**, we will build a production-grade training engine supporting:
- Gradient Accumulation (simulating large batch sizes on our 16GB RAM CPU).
- Cosine Learning-Rate Schedule with Linear Warmup.
- Gradient Clipping and AdamW optimization.
- Real-time Perplexity tracking and training telemetry.