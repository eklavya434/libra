# Educational Guide: Phase 32 — Multi-Modal Architecture (Vision-Language Adapter: Patch Projection & Cross-Attention)

Welcome to **Phase 32** of Project Libra! In this phase, we expand Libra beyond textual processing into **Multi-Modal AI**, teaching our Transformer model how to process and understand visual images alongside tokens.

---

## 1. WHAT: The Multi-Modal Challenge

Traditional Large Language Models (LLMs) operate strictly over a 1D discrete vocabulary of text tokens ($w_1, w_2, \dots, w_T$). An image, in contrast, is a continuous 3D tensor of pixel color channels ($C \times H \times W$).

How can an autoregressive text transformer "see" an image?

In frontier Vision-Language Models (such as **LLaVA**, **Flamingo**, and **GPT-4o**), images are not processed by complex external pipelines; instead, **images are turned into tokens**.
By cutting an image into a grid of non-overlapping 2D patches and projecting each patch into the exact continuous embedding dimension of the language model ($d_{\text{model}}$), an image simply becomes a sequence of **visual prefix tokens** ($v_1, v_2, \dots, v_N$) prepended to standard text tokens ($t_1, t_2, \dots, t_M$):

$$X = [v_1, v_2, \dots, v_N, t_1, t_2, \dots, t_M] \in \mathbb{R}^{(N + M) \times d_{\text{model}}}$$

The unified transformer then performs standard causal attention across both modalities!

---

## 2. WHY: Architectural Paradigms (LLaVA vs Flamingo)

There are two primary paradigms for connecting vision encoders to LLMs:

1. **Linear / MLP Projection (LLaVA-style)**:
   - Every patch is projected directly into a language embedding token: $W_{\text{proj}}: d_{\text{vision}} \to d_{\text{model}}$.
   - Fast, simple, and preserves full spatial granularity ($N = (H/P) \times (W/P)$ tokens).
2. **Cross-Attention Perceiver Resampler (Flamingo-style)**:
   - Uses a fixed set of learnable latent query tokens (e.g. $K = 8$ or $64$ queries).
   - Latent queries cross-attend to visual patch key-values: $\text{CrossAttention}(Q_{\text{latents}}, K_{\text{patches}}, V_{\text{patches}})$.
   - Compresses arbitrary-resolution images into a fixed number of tokens, saving context window length.

Phase 32 implements **both** approaches in `VisionLanguageAdapter`.

---

## 3. HOW: Algorithms & Core Components

### A. Vision Patch Embedder (`packages/models/vision/patch_embed.py`)
- Given an image $I \in \mathbb{R}^{B \times 3 \times H \times W}$ and patch size $P$:
  - Grid resolution: $G = H // P$.
  - Number of patches: $N = G^2$.
  - Flattened patch vectors: $x_p \in \mathbb{R}^{B \times N \times (P^2 \cdot 3)}$.
- Linear projection maps flattened patches to visual embedding dimension $d_{\text{vision}}$:
  $$E_{\text{patches}} = x_p W_{\text{patch}} + E_{\text{pos}} \in \mathbb{R}^{B \times N \times d_{\text{vision}}}$$
- $E_{\text{pos}}$ is a learned parameter tensor providing 2D spatial coordinate awareness.

### B. Vision-Language Alignment Adapter (`packages/models/vision/vlm_projector.py`)
- **LLaVA MLP Projector**:
  $$H_{\text{vision}} = W_2 \cdot \text{GELU}(W_1 E_{\text{patches}}) \in \mathbb{R}^{B \times N \times d_{\text{model}}}$$
- **Perceiver Cross-Attention**:
  $$H_{\text{vision}} = \text{Softmax}\left(\frac{Q_{\text{latents}} (E W_k)^T}{\sqrt{d}}\right) (E W_v)$$

### C. End-to-End LibraVLM (`packages/models/vision/vlm_model.py`)
- Combines `ImagePatchEmbedder`, `VisionLanguageAdapter`, and `ModernTransformerLM`.
- In `forward_embeddings(images, text_ids)`:
  - Text IDs $\to$ `tok_emb(text_ids)` $\in \mathbb{R}^{B \times T_{\text{text}} \times d_{\text{model}}}$.
  - Visual embeddings $\to$ `adapter(embedder(images))` $\in \mathbb{R}^{B \times N_{\text{patches}} \times d_{\text{model}}}$.
  - Concatenates along sequence dimension:
    $$X = [X_{\text{vision}}; X_{\text{text}}] \in \mathbb{R}^{B \times (N + T) \times d_{\text{model}}}$$
- Transformer blocks apply causal self-attention with RoPE across all tokens.
- Loss is computed strictly on text target tokens, allowing the model to train end-to-end on image captioning and Visual Question Answering (VQA).

---

## 4. TEST: Empirical Verification

- **8 Dedicated Tests**: `tests/models/test_patch_embed.py`, `tests/models/test_vlm_model.py`, `tests/api/test_multimodal_endpoint.py`.
- **End-to-End Gradients**: Verified gradient flow through language model, projection adapter, and vision patch embedder.
- **Full Test Suite**: **370 passed, 0 failed** in 33.61s.
- **Interactive CLI Demo**: `scripts/run_phase32_vlm_demo.py` executed cleanly on CPU in <10ms latency.
- **Frontend Playground**: `VisionPlayground.tsx` provides visual grid patch inspection and real-time generation.

---

## 5. NEXT: Advancing to Phase 33

With multi-modal vision-language architecture verified, we proceed to **Phase 33: Structured Data Pipelines & Domain Fine-Tuning Corpus Preparation**.
