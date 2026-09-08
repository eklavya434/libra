# Phase 29: Deliberative Reasoning Engine & Test-Time Compute (Chain-of-Thought, Self-Consistency & Best-of-N Verification)

## 1. WHAT
Phase 29 introduces **System 2 Deliberative Reasoning and Test-Time Compute Scaling** to Project Libra:
- **Thought Trace Parser & State Machine** ([`packages/models/reasoning/trace_parser.py`](file:///c:/Users/eklav/Desktop/Libra/packages/models/reasoning/trace_parser.py)): Parses raw token streams and static text, isolating internal `<think>...</think>` reflection scratchpads from clean user-facing answers, and structuring step-by-step logic.
- **Self-Consistency Majority Voting** ([`packages/models/reasoning/self_consistency.py`](file:///c:/Users/eklav/Desktop/Libra/packages/models/reasoning/self_consistency.py)): Generates $k \in [3, 5]$ stochastic reasoning trajectories at non-zero temperature and aggregates consensus answers using canonical normalization and plurality voting.
- **Best-of-$N$ Test-Time Compute Search** ([`packages/models/reasoning/search_verifier.py`](file:///c:/Users/eklav/Desktop/Libra/packages/models/reasoning/search_verifier.py)): Searches over $N$ candidate reasoning traces, scoring them via multi-criteria reward evaluation (factuality, step depth, reflection presence, and coherence) to select the optimal solution.
- **REST Endpoints** ([`apps/backend/api/v1/endpoints/reasoning.py`](file:///c:/Users/eklav/Desktop/Libra/apps/backend/api/v1/endpoints/reasoning.py)):
  - `POST /api/v1/reasoning/generate`
  - `POST /api/v1/reasoning/self-consistency`
  - `POST /api/v1/reasoning/best-of-n`
  - `POST /api/v1/reasoning/parse`
- **Interactive UI Accordion** ([`apps/frontend/src/components/ReasoningTraceAccordion.tsx`](file:///c:/Users/eklav/Desktop/Libra/apps/frontend/src/components/ReasoningTraceAccordion.tsx)): Collapsible, animated "Thinking Process" card displaying live thinking durations, step breakdowns, and clean answer separation.

---

## 2. WHY
Standard autoregressive LLM decoding operates as **System 1** (fast, reactive, intuition-based token generation). On multi-step mathematical, algorithmic, or syllogistic reasoning tasks, greedily predicting the next token often leads to early branching errors from which the model cannot backtrack.

Test-time compute scaling (formalized by OpenAI o1/o3, DeepSeek-R1, and Wang et al.) proves that spending additional FLOPs during **inference time** dramatically improves reasoning accuracy:
1. **Chain-of-Thought (CoT)**: Unlocks intermediate hidden state transitions, allowing the model to decompose complex problems into verifiable sub-steps.
2. **Self-Consistency (Wang et al., 2022)**: Replaces greedy decoding with diverse stochastic paths. While individual reasoning trajectories may err, the correct final answer is usually the modal consensus across multiple paths.
3. **Best-of-$N$ Verification**: Directly uses a reward model or automated referee to filter out hallucinated or poorly justified completions.

---

## 3. HOW (The Mathematics & Mechanics)

### Self-Consistency Formulation
Given prompt $x$, we sample $k$ independent reasoning trajectories $(t_1, a_1), (t_2, a_2), \dots, (t_k, a_k) \sim P_{\text{LM}}(\cdot \mid x)$ where $t_i$ represents the internal reasoning trace and $a_i$ represents the extracted answer:
$$\text{Consensus Answer} = \arg\max_{a \in \mathcal{A}} \sum_{i=1}^k \mathbb{I}(\text{norm}(a_i) = a)$$
$$\text{Confidence Score} = \frac{\max_{a} \sum_{i=1}^k \mathbb{I}(\text{norm}(a_i) = a)}{k}$$

### Best-of-$N$ Search
Given candidate generator $G(x) \to \{(t_1, a_1), \dots, (t_N, a_N)\}$ and reward/referee scoring function $R(x, t, a) \in [0.0, 10.0]$:
$$(t^*, a^*) = \arg\max_{i \in \{1, \dots, N\}} R(x, t_i, a_i)$$
where $R$ rewards:
$$R(x, t, a) = R_{\text{base}}(x, a) + \lambda_{\text{steps}} \cdot \min(1.5, |\text{steps}| \times 0.3) + \lambda_{\text{thought}} \cdot \mathbb{I}(\text{has\_thought})$$

---

## 4. TEST & Verification
- **Unit Tests (`tests/models/test_reasoning.py`)**:
  - Validated parsing of standard `<think>...</think>` tags and unclosed tags.
  - Tested streaming state machine transitions (`initial` $\to$ `thinking` $\to$ `answering`).
  - Tested answer normalization across numbers, percentages, and text.
  - Tested `SelfConsistencyEngine` and `BestOfNVerifier` execution.
- **API Tests (`tests/api/test_reasoning_endpoint.py`)**:
  - `POST /api/v1/reasoning/generate`: verified trace separation.
  - `POST /api/v1/reasoning/self-consistency`: verified vote distribution and consensus.
  - `POST /api/v1/reasoning/best-of-n`: verified candidate ranking.
  - `POST /api/v1/reasoning/parse`: verified raw text parsing.
- **Frontend Build**: Verified zero-error compilation with Next.js App Router and TypeScript.
- **Interactive Demo**: Verified via [`scripts/run_phase29_reasoning_demo.py`](file:///c:/Users/eklav/Desktop/Libra/scripts/run_phase29_reasoning_demo.py).

---

## 5. NEXT (Phase 30)
**Phase 30: Long-Context Architecture & Rotary Position Scaling (YaRN & Dynamic NTK-Aware RoPE)**.
Extending model context windows beyond training length (up to 8K/16K tokens) without catastrophic perplexity explosion on local CPU.
