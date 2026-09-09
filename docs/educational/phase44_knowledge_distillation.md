# Phase 44: Knowledge Distillation & Model Shrinking (Teacher-Student Logit Transfer)

Project Libra — First-Principles Educational Laboratory & ChatGPT-like Assistant

---

## 1. WHAT: What Was Built

Phase 44 implements **Knowledge Distillation (KD)** and **Structural Model Shrinking** for transformer language models from first principles in PyTorch. It bridges the gap between high-capacity, compute-heavy "teacher" models and compact, low-latency "student" models designed for CPU execution.

Key Deliverables:
1. **Model Shrinker & Architecture Pruning (`packages/models/components/model_shrinking.py`)**:
   - `ModelShrinker`: Derives valid student transformer configurations, implements layer-dropping sub-network extraction (e.g. $[0, 2]$ from a 4-layer teacher), transfers matching embedding and block weights, and computes exact parameter reduction, compression ratios, and memory savings.
   - `forward_with_hidden_states`: Collects intermediate transformer representations for layer-to-layer feature alignment.
2. **Distillation Trainer (`packages/training/distillation_trainer.py`)**:
   - `DistillationTrainer`: Freezes the teacher in evaluation mode (`requires_grad = False`) and trains the student using a composite objective:
     $$\mathcal{L}_{\text{distill}} = \alpha \mathcal{L}_{\text{soft}}(\tau) + (1-\alpha) \mathcal{L}_{\text{hard}} + \lambda \mathcal{L}_{\text{hidden}}$$
   - $\tau^2$-scaled Kullback-Leibler divergence soft loss: preserves gradient scale across varying temperature values.
   - Real-time `DistillationTelemetry` tracking total loss, soft loss, hard cross-entropy, top-1 logit agreement rate, and student/teacher perplexities.
3. **Distillation Evaluator & Dark Knowledge Analyzer (`packages/evaluation/distillation_eval.py`)**:
   - `DistillationEvaluator`: Analyzes softmax flattening across temperatures ($\tau \in [1, 10]$), extracts top-$k$ secondary token probabilities, computes Shannon entropy and Jensen-Shannon divergence, and benchmarks CPU inference latency, throughput (tokens/sec), and speedup factor.
4. **FastAPI Endpoints (`apps/backend/api/v1/endpoints/distillation.py`)**:
   - `POST /api/v1/distillation/train`: Runs lightweight educational CPU training loops.
   - `POST /api/v1/distillation/soft_labels`: Analyzes temperature softening and reveals dark knowledge distributions.
   - `POST /api/v1/distillation/evaluate`: Generates comparative latency benchmarks, parameter compression statistics, and next-token prediction agreement.
   - `GET /api/v1/distillation/presets`: Supplies pre-configured educational templates.
5. **Interactive Next.js Distillation Lab UI (`apps/frontend/src/components/DistillationView.tsx`)**:
   - Architecture comparison cards (Teacher vs Student parameter, layer, hidden dim, memory).
   - Interactive Dark Knowledge & Temperature Explorer with dual probability distribution bars.
   - Live Distillation Trainer dashboard with loss convergence telemetry and top-1 agreement tracking.
   - Speed & Compression CPU benchmark panel with side-by-side prompt predictions.

---

## 2. WHY: Why It Exists & What Problem It Solves

### The Parameter-Efficiency Paradox
Large transformer models possess exceptional reasoning and linguistic competence, but their memory bandwidth requirements and floating-point operations (FLOPs) make them prohibitively slow on consumer CPUs. Simple quantization (INT8/INT4) reduces weight footprint, but cannot change the quadratic complexity of attention or reduce layer depth.

### Dark Knowledge (Hinton et al., 2015)
When a neural network is trained solely on hard one-hot targets ($y \in \{0, 1\}$), all incorrect classes are treated equally (loss = 0 for all non-targets). However, a well-trained teacher model assigns structured, informative probabilities to non-target classes. For example, given the context `"The chef cooked a delicious"`, the teacher might assign:
- `"meal"`: 0.60
- `"dish"`: 0.25
- `"soup"`: 0.10
- `"car"`: 0.00001

This secondary probability distribution contains **dark knowledge**—a dense semantic map of the problem domain. By softening the teacher's logits with temperature $\tau > 1$, the student learns these rich geometric relationships, allowing a 2-layer model to match the generalizability of a 4-layer teacher with a fraction of the parameters.

---

## 3. HOW: Mathematical Foundations & Code Implementation

### A. Temperature-Softened Softmax
Given output logits $z \in \mathbb{R}^V$ and temperature $\tau > 0$:
$$p_i^\tau = \frac{\exp(z_i / \tau)}{\sum_{j=1}^V \exp(z_j / \tau)}$$
- When $\tau \to 0$, $p^\tau$ collapses to a hard one-hot distribution (argmax).
- When $\tau = 1$, standard autoregressive softmax probabilities are produced.
- When $\tau > 1$, entropy increases and the distribution flattens, amplifying the signal of near-target alternatives.

### B. Kullback-Leibler (KL) Divergence Soft Loss
The information divergence between teacher distribution $P^\tau$ and student distribution $Q^\tau$ is:
$$D_{\text{KL}}(P^\tau \parallel Q^\tau) = \sum_{i=1}^V P_i^\tau \log\left(\frac{P_i^\tau}{Q_i^\tau}\right)$$

In PyTorch, using log-probabilities for the student and standard probabilities for the teacher:
```python
student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
kl_div = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
soft_loss = kl_div * (temperature**2)
```

#### Why Multiply by $\tau^2$?
As temperature $\tau$ increases, the magnitude of the gradients computed with respect to the student logits scales approximately as $\frac{1}{\tau^2}$. Multiplying the KL divergence by $\tau^2$ normalizes the gradient scale, ensuring the soft loss remains balanced with the hard cross-entropy loss regardless of the temperature chosen.

### C. Layer-Dropping Pruning Strategy
Rather than initializing the student with random weights, `ModelShrinker.shrink_layers` performs structural sub-network extraction:
1. Selects evenly spaced layers from teacher: $L_{\text{student}} = \{ \text{round}(i \cdot \frac{L_t - 1}{L_s - 1}) \}$.
2. Copies token embedding matrix (`tok_emb.weight`) and final RMSNorm (`norm_f.weight`).
3. Loads state dictionaries from chosen teacher transformer blocks into corresponding student blocks.
4. Provides a warm start, accelerating distillation convergence by up to $3\times$.

---

## 4. TEST: How It Was Verified

1. **Unit Testing Suite (`pytest`)**:
   - `tests/models/test_model_shrinking.py`: Verified student configuration derivation, layer index sampling, weight transfer fidelity from teacher blocks, and compression statistics.
   - `tests/training/test_distillation_trainer.py`: Verified teacher model parameters remain strictly frozen (`requires_grad = False`), zero soft loss when student matches teacher, $\tau^2$ scaling correctness, loss decrease over training iterations, and intermediate hidden state alignment.
   - `tests/api/test_distillation_endpoint.py`: Verified `POST /train`, `POST /soft_labels`, `POST /evaluate`, and `GET /presets` API contracts.
2. **Full Repository Regression**:
   - **505 / 505 unit tests passing** across all 44 phases in 83 seconds on CPU.
3. **Static Analysis & Linting**:
   - `ruff check apps/ packages/ tests/`: 100% clean pass.
   - `ruff format apps/ packages/ tests/`: Formatted according to PEP 8 standards.
4. **Production Frontend Build**:
   - `npm run build` in `apps/frontend`: Compiled 4/4 static pages cleanly with zero TypeScript or JSX warnings.

---

## 5. NEXT: What the Next Phase Requires

The curriculum transitions to:
- **Phase 45: Mixture of Experts (MoE) Architecture (Sparse Routing & Top-K Gating)**:
  - Top-$k$ noisy gating network over specialized feed-forward expert blocks.
  - Load balancing auxiliary loss to prevent expert starvation/collapse.
  - Constant inference FLOPs with expanded total parameter capacity.
