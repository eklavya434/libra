# Phase 46: Speculative Verification & Medusa Multi-Head Drafting (Multi-Token Speculative Decoding without an Auxiliary Draft Model)

Project Libra — First-Principles Educational Laboratory & ChatGPT-like Assistant

---

## 1. WHAT: What Was Built

Phase 46 implements **Medusa Multi-Head Speculative Drafting and Parallel Prefix Verification** for educational language models in Project Libra from mathematical first principles in PyTorch. It accelerates autoregressive decoding by generating multiple future tokens (	+1, t+2, ..., t+M+1) in a single forward pass without the memory overhead, architecture mismatch, or sequential drafting bottleneck of an auxiliary draft model.

Key Deliverables:
1. **Medusa Components (packages/models/components/medusa.py)**:
   - MedusaHead: A single speculative decoding head comprising a residual Linear-SiLU-Linear projection block followed by an unembedding projection layer to vocabulary space.
   - MedusaModel: Wraps any standard causal transformer (e.g. LlamaForCausalLM or TransformerLM), mounts M Medusa heads to the final hidden state, and provides:
     - orward_with_medusa: Returns base logits and a list of M speculative head logits.
     - compute_medusa_loss: Trains heads with ground-truth token targets discounted by an exponential decay factor \lambda^k = 0.8^k.
     - medusa_generate: Generates M+1 candidates per step and verifies them in parallel in a single forward pass, dynamically accepting valid prefixes.
2. **Medusa Evaluator & Benchmarking (packages/evaluation/medusa_eval.py)**:
   - MedusaEvaluator: Quantitatively benchmarks autoregressive generation vs Medusa speculative decoding, calculating:
     - Mean accepted tokens per step (E[alpha]).
     - Per-head top-1 accuracy on validation corpora.
     - Wall-clock latency and throughput speedup (S = Throughput_Medusa / Throughput_Base).
     - Rejection rate statistics and acceptance distribution histograms.
3. **FastAPI Endpoints (pps/backend/api/v1/endpoints/medusa.py)**:
   - POST /api/v1/medusa/generate: Runs speculative generation returning generated text, accepted tokens, rejected drafts, speedup, and acceptance rate.
   - POST /api/v1/medusa/benchmark: Runs side-by-side benchmark of autoregressive vs Medusa decoding across prompt datasets.
   - POST /api/v1/medusa/train: Runs educational head training with configurable epochs, learning rate, and decay factor.
   - GET /api/v1/medusa/presets: Provides educational templates for greedy decoding, conservative drafting, aggressive drafting, and code completion.
4. **Interactive Next.js Medusa Lab UI (pps/frontend/src/components/MedusaView.tsx)**:
   - Interactive speculative playground with prompt presets, head count slider (M), and real-time generation.
   - Dynamic **Verification Tape** with color-coded pills for accepted vs rejected candidate tokens.
   - Speedup gauge meter, throughput comparator, and per-head acceptance accuracy bar charts.
   - Interactive training panel for fine-tuning Medusa heads on CPU in seconds.

---

## 2. WHY: Why It Exists & What Problem It Solves

### The Memory-Bound Decoding Bottleneck
In standard autoregressive language generation, tokens are generated strictly one by one:
x_{t+1} ~ P(. | x_{<= t})
Each token requires loading all model weights from RAM/VRAM to the processor, resulting in low arithmetic intensity (memory-bandwidth bound). On CPU-only environments, this sequential memory traversal creates severe latency bottlenecks.

### Limitations of Traditional Speculative Decoding (Phase 21)
Traditional speculative decoding employs a small auxiliary **Draft Model** (e.g. a 68M parameter model drafting for a 7B parameter target model):
1. **Memory Duplication**: Both models must reside concurrently in RAM.
2. **Vocabulary & Architecture Alignment**: Both models must share identical tokenizers and coordinate KV-caches.
3. **Draft Latency**: The draft model still generates candidate tokens sequentially (M steps for M tokens).

### The Medusa Breakthrough
Medusa replaces the auxiliary draft model entirely by augmenting the original model's final hidden representation h_t with M lightweight residual prediction heads:
- **Zero Additional Base Invocations**: In the base model's forward pass, Head 0 predicts 	+2, Head 1 predicts 	+3, and Head k predicts 	+k+2 **in parallel**.
- **Trivial Parameter Overhead**: Each head is merely an MLP block with a projection layer (<3% parameter overhead).
- **Guaranteed Output Distribution**: Greedy prefix verification guarantees that the accepted token sequence matches the base model's exact greedy trajectory.

---

## 3. HOW: Mathematical Foundations & Code Implementation

### A. Residual Medusa Head Architecture
Given the final hidden state vector h_t in R^{d_model} from the base transformer at step 	:
z_t = SiLU(h_t W_in + b_in)
h'_t = h_t + z_t W_out + b_out
logits^(k)_t = h'_t W_proj + b_proj
where W_proj projects into vocabulary space. The residual connection preserves the semantic features learned by the base model.

### B. Parallel Prefix Verification Algorithm
At each step 	:
1. **Draft Formulation**:
   - Base model standard head produces s_0 = argmax(logits_base), which is **always accepted**.
   - Medusa heads produce draft candidates s_k = argmax(logits^(k-1)_t) for k in {1, ..., M}.
   - Candidate sequence: [s_0, s_1, s_2, ..., s_M].
2. **Parallel Verification Forward Pass**:
   - The concatenated sequence [x_{<= t}, s_0, s_1, ..., s_{M-1}] is passed to the base model in **a single forward pass**.
   - Let \hat{s}_k = argmax(P_base(. | x_{<= t}, s_0, ..., s_{k-1})).
3. **Greedy Acceptance**:
   - Check condition: for k = 1, ..., M, accept s_k if and only if s_k == \hat{s}_k and all preceding drafts were accepted.
   - If head k mismatches, reject s_k and all subsequent candidates.
   - Append all accepted tokens to the generation sequence. The base prediction from the rejection position becomes the next anchor token.

### C. Head Training with Exponential Loss Discounting
To train Medusa heads while keeping base model weights frozen:
L_Medusa = sum_{k=1}^M \lambda^k * CrossEntropy(logits^(k)_t, y_{t+k+1})
where \lambda in (0, 1] (default \lambda = 0.8) discounts predictions further into the future, reflecting increased prediction entropy and preventing distant noise from destabilizing early heads.

---

## 4. TEST: How It Was Verified

1. **Unit Test Suite (pytest)**:
   - 	ests/models/test_medusa_components.py:
     - Verified MedusaHead forward shape and residual connection integrity.
     - Verified MedusaModel forward pass produces correct base logits and M head logits.
     - Verified compute_medusa_loss backpropagation updates only head parameters while base remains frozen.
   - 	ests/models/test_medusa_generation.py:
     - Verified medusa_generate produces valid output with acceptance rate >= 1.0.
     - Verified verification step tracks accepted tokens, rejected drafts, and acceptance history.
     - Verified fallback behavior when all speculative drafts are rejected (s_0 always accepted).
   - 	ests/api/test_medusa_endpoint.py:
     - Verified /generate, /benchmark, /train, and /presets HTTP endpoints with complete request/response validation.
2. **Full Repository Regression**:
   - **526 / 526 unit tests passing** across all 46 phases in ~84 seconds on CPU.
3. **Static Analysis & Linting**:
   - uff check . and uff format --check .: 100% clean.
4. **Production Frontend Build**:
   - 
pm run build: Compiled 4/4 static pages cleanly with zero TypeScript errors.

---

## 5. NEXT: What the Next Phase Requires

The curriculum transitions to:
- **Phase 47: Direct Alignment & Online DPO / Kahneman-Tversky Optimization (KTO)**:
  - Binary prospect-theoretic preference optimization without requiring pairwise preference pairs.
  - Reference model implicit reward margin and online preference updates.
  - Educational alignment benchmarking comparing PPO, offline DPO, and online KTO.
