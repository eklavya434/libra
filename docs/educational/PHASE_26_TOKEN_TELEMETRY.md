# Phase 26: Streaming Token Telemetry & Token-Level Metrics

Welcome to **Phase 26** of Project Libra. In this phase, we look inside the autoregressive transformer decoding process at runtime. Instead of treating text generation as an opaque "black-box" stream of characters, Phase 26 captures exact mathematical decision metrics for every single generated token from first principles.

---

## 1. WHAT: What Was Built

We implemented a fine-grained token-level telemetry engine and real-time Server-Sent Events (SSE) streaming infrastructure:

1. **First-Principles Mathematical Telemetry Core (`packages/models/telemetry.py`)**:
   - **Conditional Token Probability**: $p(w_t \mid w_{<t}) = \text{softmax}(z_t)[w_t]$.
   - **Shannon Surprisal (Self-Information)**: $I(w_t) = -\log_2 p(w_t \mid w_{<t})$ in bits.
   - **Distribution Entropy**: $H(P_t) = -\sum_{v \in V} p_t(v) \log_2 p_t(v)$ in bits.
   - **Top-$k$ Alternative Candidates**: Slices top-5 competing tokens, their probability fractions, and natural log-probabilities.
   - **Per-Token Step Latency**: Wall-clock duration ($\Delta t$ ms) spent calculating each forward pass.
   - **Sequence Perplexity**: $\text{PPL} = 2^{\bar{I}} = 2^{\frac{1}{N}\sum_{t=1}^N I(w_t)}$.
   - **Autoregressive Telemetry Generators**: `stream_generate_with_telemetry` and `astream_generate_with_telemetry`.
   - **Teacher-Forcing Evaluator**: `analyze_sequence_telemetry` evaluates surprisal without sampling.

2. **Backend API Endpoints (`apps/backend/api/v1/endpoints/telemetry.py`)**:
   - `POST /api/v1/telemetry/generate`: Non-streaming generation returning full token-by-token metadata and sequence metrics.
   - `POST /api/v1/telemetry/stream`: Real-time SSE streaming emitting `event: token` chunks and `event: done` sequence summaries.
   - `POST /api/v1/telemetry/analyze`: Teacher-forcing evaluation of arbitrary user text.

3. **Frontend Visual Inspection (`apps/frontend/src/components/TokenSurprisalHeatmap.tsx` & `ChatArea.tsx`)**:
   - **Color-Coded Heatmap**:
     - 🟢 **Low Surprisal (< 1.0 bit, $p > 50\%$)**: High confidence, deterministic word choice.
     - 🟡 **Moderate Surprisal (1.0 to 3.0 bits, $12.5\% \le p \le 50\%$)**: Branching choice.
     - 🟣 **High Surprisal (> 3.0 bits, $p < 12.5\%$)**: Unexpected or creative token choice.
   - **Interactive Popover Inspector**: Clicking any token reveals step latency, probability %, logprob, entropy, and top-5 alternative candidate probability bars.
   - **On-Demand Inspection**: "Inspect Tokens" button in chat interface dynamically analyzes assistant responses.

4. **Verification & Demonstration**:
   - 10 unit and API tests in `tests/models/test_telemetry.py` and `tests/api/test_telemetry_endpoint.py`.
   - Interactive CLI demo in `scripts/run_phase26_telemetry_demo.py` featuring ANSI colorized terminal streams.

---

## 2. WHY: Why It Exists and What Problem It Solves

### The Black-Box Illusion
Standard LLM chat applications stream plain strings (`delta: {"content": "..."}`). This hides the statistical nature of autoregressive generation:
- The user cannot tell if the model was 99.9% confident in a statement or blindly guessing at 2% probability among fifty near-equal candidates.
- Hallucinations often manifest as sudden spikes in token surprisal or high-entropy distributions where the model is uncertain.
- Sampling parameters (temperature, top-p, top-k) distort or truncate the distribution in subtle ways that are invisible without inspecting step-level candidate probabilities.

### First-Principles Explainability
Phase 26 empowers learners and researchers to observe:
1. **Model Confidence**: Exactly how certain the network was when emitting each token.
2. **Entropy as Uncertainty**: When $H(P_t)$ is low, the model's path is constrained (e.g. syntax, boilerplate). When $H(P_t)$ is high, the model encounters a critical semantic fork.
3. **Alternative Candidates**: What other words the model considered and why it picked the chosen token.
4. **Information Content**: Surprisal in bits quantitatively answers: *how much new information does this token impart?*

---

## 3. HOW: Mathematical Foundations & Code Implementation

### A. Shannon Surprisal (Self-Information)
In information theory (Claude Shannon, 1948), the self-information or surprisal $I(x)$ of an event $x$ with probability $p(x)$ is defined as:
$$I(x) = -\log_2 p(x) \quad \text{(bits)}$$

- If $p(x) = 1.0$ (certain event): $I(x) = -\log_2(1.0) = 0.0\text{ bits}$. No new information is revealed.
- If $p(x) = 0.5$ (fair coin flip): $I(x) = -\log_2(0.5) = 1.0\text{ bit}$.
- If $p(x) = 0.125$: $I(x) = -\log_2(0.125) = 3.0\text{ bits}$.
- If $p(x) \to 0$: $I(x) \to \infty$. Highly unexpected events convey maximal information.

In code (`packages/models/telemetry.py`):
```python
def compute_surprisal_bits(prob: float, eps: float = 1e-12) -> float:
    safe_prob = max(float(prob), eps)
    return -math.log2(safe_prob)
```

### B. Distribution Entropy
Shannon entropy measures the average unpredictability or uncertainty of the entire vocabulary distribution $P_t$:
$$H(P_t) = -\sum_{v \in V} p_t(v) \log_2 p_t(v) \quad \text{(bits)}$$

- **Minimum Entropy**: $H = 0.0$ bits when one token has probability $1.0$ (pure determinism).
- **Maximum Entropy**: $H = \log_2(V)$ bits when all $V$ vocabulary tokens are equally likely (uniform uncertainty).

In PyTorch:
```python
def compute_entropy_bits(probs: torch.Tensor, eps: float = 1e-12) -> float:
    mask = probs > eps
    if not torch.any(mask):
        return 0.0
    p = probs[mask]
    return max(0.0, float(-torch.sum(p * torch.log2(p)).item()))
```

### C. Sequence Perplexity Equivalence
Perplexity ($\text{PPL}$) is traditionally expressed as the exponential of the cross-entropy loss in natural logarithms:
$$\text{PPL} = \exp\left( -\frac{1}{N} \sum_{t=1}^N \ln p(w_t) \right)$$

Using the change-of-base identity $\ln(x) = \log_2(x) \cdot \ln(2)$:
$$\text{PPL} = \exp\left( \ln(2) \cdot \frac{1}{N} \sum_{t=1}^N -\log_2 p(w_t) \right) = 2^{\bar{I}}$$
where $\bar{I} = \frac{1}{N} \sum_{t=1}^N I(w_t)$ is the **mean surprisal in bits**. Our implementation verifies this identity to machine precision.

---

## 4. TEST: Verification Results

### Automated Test Suite
Run the 10 dedicated telemetry tests:
```powershell
.\.venv\Scripts\pytest -v tests/models/test_telemetry.py tests/api/test_telemetry_endpoint.py
```
Output:
```
tests\models\test_telemetry.py::test_surprisal_calculation PASSED
tests\models\test_telemetry.py::test_entropy_calculation PASSED
tests\models\test_telemetry.py::test_step_telemetry_top_k PASSED
tests\models\test_telemetry.py::test_sequence_aggregation_perplexity PASSED
tests\models\test_telemetry.py::test_stream_generate_with_telemetry PASSED
tests\models\test_telemetry.py::test_analyze_sequence_telemetry PASSED
tests\api\test_telemetry_endpoint.py::test_telemetry_generate_endpoint PASSED
tests\api\test_telemetry_endpoint.py::test_telemetry_stream_endpoint PASSED
tests\api\test_telemetry_endpoint.py::test_telemetry_analyze_endpoint PASSED
tests\api\test_telemetry_endpoint.py::test_telemetry_analyze_too_short PASSED

10 passed in 5.86s
```

Full repository test suite: **305 passed, 0 failed** in 37.10s.

### Interactive CLI Demo
```powershell
.\.venv\Scripts\python.exe scripts/run_phase26_telemetry_demo.py
```
Console output:
```
===========================================================================
  ♎ PROJECT LIBRA — PHASE 26: STREAMING TOKEN TELEMETRY & METRICS
===========================================================================
Consumer CPU Target: Intel Core i5-12450H | Free Offline Inference ($0 / ₹0)

1. First-Principles Mathematical Definitions:
   • Conditional Probability:   p(w_t | w_<t) = softmax(z_t)[w_t]
   • Shannon Surprisal:         I(w_t) = -log2(p(w_t | w_<t)) [bits]
   • Distribution Entropy:      H(P_t) = -sum_v p_t(v) log2(p_t(v)) [bits]
   • Sequence Perplexity:       PPL = 2^(mean_surprisal_bits)

2. Initializing Educational Modern Transformer Architecture:
   Model parameters: 123,200 trainable weights
   Vocab size: 128 | Embedding dim: 64 | Layers: 2

3. Live Streaming Autoregressive Decoding with Surprisal Heatmap:
   Prompt: "Libra AI"
   Surprisal Color Key: ■ <1.0 bit (Confident)  ■ 1.0-3.0 bits (Moderate)  ■ >3.0 bits (Surprising)

   Generated stream: SQ9pkad|Jj[

4. Token-by-Token Telemetry Breakdown Table:
   Step  | Token    | Prob (%)   | Surprisal    | Entropy    | Latency   | Top Candidate Alternatives
   --------------------------------------------------------------------------
   #1    | S        | 1.06%      | 6.554 bits   | 6.971 bits | 3.9 ms    | "J" (1.3%), "s" (1.3%), "D" (1.2%)
   #2    | Q        | 0.83%      | 6.913 bits   | 6.975 bits | 2.4 ms    | "b" (1.4%), "#" (1.2%), "Y" (1.1%)
   ...
   #12   | [        | 0.63%      | 7.318 bits   | 6.974 bits | 1.9 ms    | "9" (1.2%), " " (1.2%), "Z" (1.2%)

5. Aggregate Sequence-Level Metrics:
   • Total Generated Tokens:    12
   • Cumulative Step Latency:   25.20 ms
   • Generation Throughput:     476.2 tokens/second
   • Mean Sequence Surprisal:   6.8753 bits/token
   • Sequence Perplexity:       117.3982 (2^6.8753)
   • Mean Distribution Entropy: 6.9715 bits
   • Most Surprising Token:     "[" (7.318 bits, p = 0.63%)
   • Most Confident Token:      "9" (6.419 bits, p = 1.17%)

6. Teacher-Forcing Sequence Surprisal Evaluation (Without Sampling):
   Evaluated text: "Deep learning laboratory" (23 transitions)
   Evaluation Perplexity: 116.0620 | Mean Surprisal: 6.8588 bits/tok

✓ Phase 26 Streaming Token Telemetry verification completed successfully.
```

---

## 5. NEXT: Phase 27 Preview

With Phase 26 complete, Project Libra possesses transparent, real-time observability over token distributions, Shannon surprisal, distribution entropy, and sequence perplexity.

Next Milestone: **Phase 27: Constrained Decoding & Grammar Masking (CFG & Regex)**
- Direct token-masking logits processor driven by formal grammars (Context-Free Grammars / EBNF) and regular expressions.
- Guarantees 100% syntactically valid JSON, SQL, Python, or custom schema outputs from first principles without post-hoc parsing failures.
