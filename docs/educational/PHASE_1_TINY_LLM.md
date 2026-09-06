# Educational Guide: Phase 1 — Educational Tiny LLM

Welcome to **Phase 1** of **Libra**!

In this phase, we moved from system plumbing into the mathematical and computational heart of Generative AI: building an **autoregressive, decoder-only Transformer language model from scratch in PyTorch** and training it on your **CPU**.

---

## 1. WHAT Was Built?

We created a complete miniature Transformer model architecture from first principles without using high-level black-box wrappers:

1. **`TinyTransformerConfig`** (`packages/models/config.py`): The blueprint dataclass containing hyperparameters (256 vocab size, 128 context length, 128 hidden dimension, 4 attention heads, 2 transformer layers).
2. **`TokenEmbedding` & `PositionalEmbedding`** (`packages/models/transformer.py`): Converting discrete character IDs and sequence positions into continuous 128-dimensional vectors.
3. **`CausalSelfAttention`** (`packages/models/transformer.py`): The multi-head attention mechanism equipped with a strict lower-triangular causal mask preventing future information leakage.
4. **`FeedForward` (MLP)** (`packages/models/transformer.py`): Position-wise 2-layer neural network with GELU non-linear activation.
5. **`TransformerBlock`** (`packages/models/transformer.py`): Pre-LayerNorm block integrating LayerNorm, Multi-Head Attention, Residual Addition, and FeedForward MLP.
6. **`TinyTransformerLM`** (`packages/models/transformer.py`): The complete language model that projects hidden states into 256 vocabulary logits and calculates Cross-Entropy training loss.
7. **`generate()`** (`packages/models/generation.py`): Autoregressive token sampler supporting greedy decoding and temperature scaling.
8. **`train_tiny_llm()`** (`packages/training/trainer.py`): CPU training loop with AdamW optimizer, gradient clipping, checkpoint serialization, and enforcement of the <15-minute budget.

---

## 2. WHY Do We Need Each Component?

| Component | Why We Need It | What Happens Without It |
| :--- | :--- | :--- |
| **Token Embedding** | Neural networks cannot multiply words or letters; they only multiply numbers and vectors. | The model cannot read text inputs. |
| **Positional Embedding** | Attention is permutation invariant (order-agnostic). "Dog bites man" and "Man bites dog" look identical to pure attention. | The model has no sense of word order or sequence. |
| **Causal Mask** | In generative text, token $t$ must predict token $t+1$ based *only* on past context ($1 \dots t$). | The model cheats by looking ahead at the answer during training, learning nothing. |
| **Multi-Head Attention** | Allows tokens to interact across multiple representation subspaces simultaneously (e.g. one head tracks grammar, another tracks topic). | The model can only look at one relationship at a time. |
| **Residual Connections** | Adds the original input to the output ($x + f(x)$), allowing gradients to flow directly backwards during backpropagation. | Deep networks suffer from vanishing gradients and fail to train. |
| **Layer Normalization** | Keeps the mean and variance of activations stable across layers. | Numbers explode or vanish during forward passes. |
| **Feed-Forward MLP** | Expands the representation to $4 \times d_{model}$ to synthesize and memorize factual associations. | Attention alone can only route existing information, not transform it. |

---

## 3. HOW Does It Work? (From Input to Output)

### Step 1: Text to Token IDs
Computers see characters as numbers. For example:
`"Libra"` $\to$ ASCII/UTF-8 byte values $\to$ `[76, 105, 98, 114, 97]`.

### Step 2: Embedding Space ($d_{model} = 128$)
Each token ID $k$ looks up row $k$ in an embedding table of size $256 \times 128$.
Position $0, 1, 2, \dots$ looks up row $p$ in a positional embedding table.
We add them:
$$\mathbf{x} = \text{TokenEmbedding}(k) + \text{PositionalEmbedding}(p)$$

### Step 3: Queries, Keys, and Values (The Search Engine of Attention)
For each token vector $\mathbf{x}$, we create three vectors:
- **Query ($Q$)**: What this token is looking for.
- **Key ($K$)**: What this token contains (its label).
- **Value ($V$)**: The actual information to pass along if there is a match.

Attention computes the dot product between Query and Key:
$$\text{Scores} = \frac{Q K^T}{\sqrt{d_k}}$$
If the query and key point in the same direction, the score is high.

### Step 4: The Causal Mask (Lower Triangular)
To make sure position 1 cannot see position 2:
$$\begin{pmatrix}
s_{11} & -\infty & -\infty \\
s_{21} & s_{22} & -\infty \\
s_{31} & s_{32} & s_{33}
\end{pmatrix}$$
When we apply $\text{softmax}$, $e^{-\infty} = 0$, so future tokens receive **0.00%** attention.

### Step 5: Logits and Loss
The model outputs a vector of 256 numbers (logits) for the next character:
$$\text{Probabilities} = \text{Softmax}(\text{Logits})$$
If the real next character was `'i'` (token 105), we want the probability for 105 to be close to $1.0$.
We measure error using **Cross-Entropy Loss**:
$$\mathcal{L} = -\ln(P_{\text{target}})$$
- If the model gives target 100% probability: $\mathcal{L} = -\ln(1.0) = 0$ (perfect!).
- At random initialization with 256 tokens: $\mathcal{L} \approx -\ln(1/256) \approx 5.54$.

### Step 6: Gradient Descent & AdamW
$$\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \nabla_{\theta} \mathcal{L}$$
AdamW computes the direction to adjust every parameter weight to make the loss smaller on the next step.

---

## 4. How We Verified It

We wrote 4 rigorous unit tests in `tests/models/`:
1. `test_transformer_forward_shape`: Verified tensor dimensions $(B, T) \to (B, T, 256)$.
2. `test_causal_mask`: Proved that altering future tokens has zero effect on past logits.
3. `test_overfit`: Proved that backpropagation mathematically decreases loss on a sample batch.
4. `test_checkpoint`: Proved save/load state dict roundtrip determinism.

---

## 5. NEXT: What Comes Next?

In **Phase 2: Tokenizer**, we will replace raw byte/character mapping with a modern **Byte-Pair Encoding (BPE)** subword tokenizer, explaining how words are broken into optimal chunks to maximize vocabulary efficiency!