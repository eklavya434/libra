# Educational Guide: Phase 30 — Long-Context Architecture & Rotary Position Scaling (RoPE, YaRN & Dynamic NTK)

Welcome to **Phase 30** of Project Libra! In this phase, we explore the mathematical and architectural challenges of extending a Transformer language model's context window beyond its pre-training limit.

---

## 1. WHAT: The Long-Context Challenge

When Large Language Models are pre-trained on a fixed sequence length (e.g. $L_{\text{train}} = 512$ or $2048$ tokens), they encounter severe performance degradation when fed prompts at test time where $L > L_{\text{train}}$.

This manifests in two ways:
1. **Perplexity Explosion**: At sequence positions $m > L_{\text{train}}$, positional encoding frequencies produce rotation vectors that the model has never observed during gradient descent, leading to nonsensical activations and exploded validation loss.
2. **Attention Dilution & "Lost in the Middle"**: As the sequence length scales by factor $s$, attention logits become either overly diffuse or sharp, degrading associative recall of facts located in the middle of long documents.

Phase 30 introduces first-principles **position scaling** methods that allow an existing model to process sequences $4\times$ to $16\times$ longer without retraining from scratch:
- **Linear Position Interpolation (PI)** (Chen et al., 2023)
- **Dynamic NTK-Aware RoPE** (bloc97 / Peng et al., 2023)
- **YaRN (Yet another RoPE extensioN)** (Peng et al., 2023)
- **Needle-In-A-Haystack (NIAH)** Retrieval Benchmark

---

## 2. WHY: Why Standard Extrapolation Fails

In standard Rotary Positional Embeddings (Su et al., 2021), the 2D rotation angle for position $m$ along head dimension channel $i \in [0, d/2 - 1]$ is:

$$\theta_i = b^{-2i/d}, \quad \theta_m^{(i)} = m \cdot \theta_i, \quad b = 10000$$

The **wavelength** $\lambda_i$ of each frequency channel is:
$$\lambda_i = \frac{2\pi}{\theta_i}$$

- **High-Frequency Channels (small $i$)**: Wavelength $\lambda_i \approx 6$ tokens. These channels rotate multiple full cycles even across short sentences and encode local token-to-token syntax and punctuation.
- **Low-Frequency Channels (large $i$)**: Wavelength $\lambda_i > 10000$ tokens. These channels rotate only a tiny fraction of a radian during pre-training.

If we simply extrapolate to $m = 4096$, low-frequency channels explore angles never encountered during training.

---

## 3. HOW: Position Scaling Algorithms

### A. Linear Position Interpolation (PI)
Chen et al. proposed compressing positions linearly by scale factor $s = L_{\text{target}} / L_{\text{train}}$:

$$m' = \frac{m}{s}$$

- **Advantage**: Guarantees that $m' \le L_{\text{train}}$, so angles remain strictly within the pre-trained distribution.
- **Drawback**: Uniformly compresses high-frequency channels, degrading local token order resolution and short-range grammatical precision.

---

### B. Dynamic NTK-Aware RoPE
Instead of scaling the position $m$, Neural Tangent Kernel (NTK)-aware scaling dynamically alters the base frequency $b$:

$$b' = b \cdot s^{\frac{d}{d - 2}}$$

Because $\theta_i' = (b')^{-2i/d} = b^{-2i/d} \cdot s^{-\frac{2i}{d-2}}$:
- When $i = 0$ (highest frequency): $s^0 = 1$ $\implies$ **zero scaling**, perfectly preserving local relative grammar!
- When $i = d/2 - 1$ (lowest frequency): undergoes maximum interpolation $\approx 1/s$.

---

### C. YaRN (Yet another RoPE extensioN)
YaRN unifies the best properties by computing the wavelength ratio:

$$r_i = \frac{L_{\text{train}}}{\lambda_i} = \frac{L_{\text{train}} \cdot \theta_i}{2\pi}$$

YaRN partitions channels into three distinct frequency bands:
1. **$r_i > \beta_{\text{fast}}$ ($r_i > 32$)**: High frequencies where wavelength is short. No interpolation is applied ($s = 1.0$).
2. **$r_i < \beta_{\text{slow}}$ ($r_i < 1$)**: Low frequencies where wavelength spans beyond the context window. Full linear interpolation is applied ($s = s$).
3. **$\beta_{\text{slow}} \le r_i \le \beta_{\text{fast}}$**: Smooth linear ramp function $\gamma(r_i) = \frac{r_i - \beta_{\text{slow}}}{\beta_{\text{fast}} - \beta_{\text{slow}}}$:

$$\theta_i^{\text{YaRN}} = (1 - \gamma(r_i)) \frac{\theta_i}{s} + \gamma(r_i) \theta_i$$

#### Attention Entropy Temperature Scaling
To prevent attention softmax distribution degradation over longer token sequences, YaRN scales query-key attention logits by:

$$\tau = \frac{1}{\sqrt{0.1 \ln(s) + 1}}$$

$$\text{Scores} = \frac{Q K^T}{\sqrt{d_k}} \cdot \tau$$

---

## 4. TEST: Needle-In-A-Haystack (NIAH) Retrieval Benchmark

To empirically evaluate long-context associative memory:
1. Construct background documents (the "Haystack") across varying token lengths (e.g. 250, 500, 1000, 2000 tokens).
2. Insert a specific target fact (the "Needle", e.g. `"The security clearance code is DELTA-9182."`) at distinct relative depths ($0\%$, $25\%$, $50\%$, $75\%$, $100\%$).
3. Query the model to retrieve the needle.
4. Visualize retrieval accuracy across the 2D grid matrix using `LongContextHeatmap.tsx`.

---

## 5. NEXT: Advancing to Phase 31

With long-context architecture, Rotary Position Scaling, and NIAH evaluation in place, Libra can process extended multi-turn dialogs and long technical documents.

In **Phase 31**, we proceed to **Advanced Inference Optimization: PagedAttention & Continuous Batching Simulation**.
