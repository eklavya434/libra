# Phase 45: Mixture of Experts (MoE) Architecture (Sparse Routing & Top-K Gating)

Project Libra — First-Principles Educational Laboratory & ChatGPT-like Assistant

---

## 1. WHAT: What Was Built

Phase 45 implements a **Sparse Mixture of Experts (MoE)** architecture for educational language models in Project Libra from mathematical first principles in PyTorch. It decouples total model capacity (parameters) from active computation cost per token (FLOPs) through dynamic conditional routing.

Key Deliverables:
1. **MoE Components (`packages/models/components/moe.py`)**:
   - `ExpertLayer`: Specialized SwiGLU feed-forward network.
   - `MoERouter`: Top-$k$ gating router with optional training-time exploration noise and Switch/Shazeer load-balancing auxiliary loss computation.
   - `SparseMoEBlock`: Dispatches tokens to selected top-$k$ experts ($k \ll E$) and accumulates weighted activations.
2. **MoE Transformer LM (`packages/models/moe_transformer.py`)**:
   - `MoETransformerConfig`: Extends modern transformer configuration with `num_experts`, `num_experts_per_tok`, and `aux_loss_coef`.
   - `MoETransformerBlock`: Interleaves causal RoPE attention with sparse MoE blocks.
   - `MoETransformerLM`: Autoregressive language model returning task cross-entropy loss, load-balancing auxiliary loss, and per-layer routing traces.
   - `count_parameters`: Distinguishes total parameter capacity from active parameters per token (e.g. 50% or 75% active compute savings).
3. **MoE Evaluator & Routing Telemetry (`packages/evaluation/moe_eval.py`)**:
   - `MoEEvaluator`: Traces token-by-token expert assignments, computes load distribution entropy ($H(f)$), calculates the Coefficient of Variation (CV) across experts, and flags starved experts ($f_i < 0.05$).
4. **FastAPI Endpoints (`apps/backend/api/v1/endpoints/moe.py`)**:
   - `POST /api/v1/moe/forward`: Evaluates input prompts and returns token-level expert assignments and gating weights.
   - `POST /api/v1/moe/train`: Runs lightweight educational MoE training demonstrating auxiliary loss stabilizing expert utilization.
   - `GET /api/v1/moe/utilization`: Evaluates expert balance and parameter efficiency across benchmark sequences.
   - `GET /api/v1/moe/presets`: Pre-configured educational templates (Mixtral 2-of-4, Switch 1-of-4, Starvation Ablation).
5. **Interactive Next.js MoE Lab UI (`apps/frontend/src/components/MoEView.tsx`)**:
   - Parameter decoupling gauge cards (Total vs Active parameters, compute savings percentage).
   - Interactive token stream tape with color-coded expert pills.
   - Token inspector drawer displaying per-layer gating weight bars.
   - Expert utilization dashboard and load-balancing auxiliary loss monitor.

---

## 2. WHY: Why It Exists & What Problem It Solves

### The Parameter Scaling Dilemma
In standard dense transformers, every token activates every single parameter in every feed-forward network. As models scale from 1B to 70B parameters, inference latency and memory bandwidth requirements scale linearly, making deployment on commodity CPUs impossible.

### Conditional Computation (Sparse Routing)
Mixture of Experts decouples model size from compute:
- **Total Capacity**: $E$ independent expert sub-networks store domain-specialized knowledge.
- **Compute per Token**: Each token is dynamically routed to only $k$ experts (e.g. $k=2$ out of $E=4$, or $k=1$ in Switch Transformer).
- **Result**: The model gains the representational capacity of a large model while consuming the FLOPs of a fractionally sized dense model.

### Preventing Expert Collapse
Without a balancing mechanism, gating routers suffer from "winner-take-all" feedback loops: the router slightly favors Expert 0, Expert 0 receives more gradients and improves faster, causing the router to route even more tokens to Expert 0. Meanwhile, other experts starve and learn nothing. The Switch/Shazeer load-balancing auxiliary loss explicitly penalizes non-uniform routing, guaranteeing all experts are utilized.

---

## 3. HOW: Mathematical Foundations & Code Implementation

### A. Top-$K$ Gating Formulation
Given token activation $x \in \mathbb{R}^{d_{\text{model}}}$:
1. **Router Logits**:
   $$H(x) = x W_g + \epsilon \cdot \text{Softplus}(x W_{\text{noise}})$$
   where $\epsilon \sim \mathcal{N}(0, 1)$ adds exploration noise during training.
2. **Selection**:
   Select indices $\mathcal{T} = \text{TopK}(H(x), k)$.
3. **Renormalization**:
   $$G(x)_i = \begin{cases} \frac{\exp(H(x)_i)}{\sum_{j \in \mathcal{T}} \exp(H(x)_j)} & \text{if } i \in \mathcal{T} \\ 0 & \text{otherwise} \end{cases}$$
   Renormalizing ensures that $\sum_{i \in \mathcal{T}} G(x)_i = 1.0$, preventing numerical scale shifts.

### B. Switch / Shazeer Load-Balancing Auxiliary Loss
$$\mathcal{L}_{\text{aux}} = \alpha \cdot E \sum_{i=1}^E f_i \cdot P_i$$
where:
- $f_i = \frac{1}{T} \sum_{t=1}^T \mathbb{I}(i \in \mathcal{T}_t)$ is the actual fraction of tokens dispatched to expert $i$.
- $P_i = \frac{1}{T} \sum_{t=1}^T \text{Softmax}(H(x_t))_i$ is the average probability mass assigned to expert $i$.
- $\alpha$ is the auxiliary loss coefficient (typically $0.01 - 0.05$).

When load is uniformly distributed ($f_i = 1/E, P_i = 1/E$):
$$\mathcal{L}_{\text{aux}} = \alpha \cdot E \sum_{i=1}^E \frac{1}{E^2} = \alpha \cdot E \cdot E \cdot \frac{1}{E^2} = \alpha$$
Any deviation increases $\mathcal{L}_{\text{aux}}$, providing smooth gradients to guide the router towards balance.

---

## 4. TEST: How It Was Verified

1. **Unit Test Suite (`pytest`)**:
   - `tests/models/test_moe_components.py`: Verified `ExpertLayer` forward shape, `MoERouter` top-$k$ weight summation to 1.0, valid expert index range, and imbalance detection.
   - `tests/models/test_moe_transformer.py`: Verified `MoETransformerLM` end-to-end forward/backward passes, task loss + auxiliary loss combination, gradient flow to router weights, and parameter sparsity counting.
   - `tests/api/test_moe_endpoint.py`: Verified `/forward`, `/train`, `/utilization`, and `/presets` API endpoints.
2. **Full Repository Regression**:
   - **517 / 517 unit tests passing** across all 45 phases in 82 seconds on CPU.
3. **Static Analysis & Linting**:
   - `ruff check .` and `ruff format --check .`: 100% clean.
4. **Production Frontend Build**:
   - `npm run build`: Compiled 4/4 static pages cleanly with zero TypeScript errors.

---

## 5. NEXT: What the Next Phase Requires

The curriculum transitions to:
- **Phase 46: Speculative Verification & Medusa Multi-Head Drafting**:
  - Multiple simultaneous speculative prediction heads attached to the final hidden state.
  - Tree-structured attention verification over candidate draft branches.
  - Multi-token speedup verification on CPU without requiring an auxiliary draft model.
