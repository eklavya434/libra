# Educational Guide: Phase 6 — Evaluation & Benchmarking

Welcome to **Phase 6** of **Libra**!

In this phase, we constructed a **quantitative evaluation framework** to assess language models scientifically, without relying on subjective impressions or fabricating benchmark numbers.

---

## 1. WHAT Was Built?

1. **Loss & Perplexity Evaluator** (`packages/evaluation/loss_eval.py`):
   - Computes empirical Cross-Entropy Loss ($\mathcal{L}$), Perplexity ($\text{PPL}$), and Bits-per-Token ($\text{BPT}$) over held-out evaluation sequences.
   - Implements sliding-window context evaluation with strided steps to evaluate corpora longer than the model's context window without boundary degradation.
2. **Multiple-Choice Log-Likelihood Engine** (`packages/evaluation/multiple_choice.py`):
   - Standard scoring architecture behind modern academic benchmarks (MMLU, ARC, HellaSwag).
   - Evaluates conditional log probabilities across candidate completions without generation hallucinations.
3. **Domain Task Probes** (`packages/evaluation/probes.py`):
   - Standardized evaluation probes spanning 5 core disciplines:
     - **Mathematics**: Arithmetic, subtraction, multiplication, magnitude comparison.
     - **Reasoning**: Temporal sequences, alphabetic succession, analogical reasoning.
     - **Coding**: Python keywords, container types, iteration syntax, branching delimiters.
     - **Instruction Following**: Structured JSON closure, binary YES/NO constraints, bracket pairing.
     - **Safety & Alignment**: Harmful request refusal pattern recognition.
4. **Unified Evaluation Harness** (`packages/evaluation/harness.py`):
   - Orchestrates automated benchmarking, formats GitHub-flavored Markdown tables, and serializes verifiable evaluation reports to JSON.

---

## 2. WHY Do We Evaluate This Way?

### A. Why Not Just "Chat" with the Model to See If It's Good?
As a beginner, it is tempting to judge a language model simply by asking it a question and reading its generated output. While this is intuitive, it is scientifically flawed for two major reasons:
1. **Sampling Stochasticity**: With temperature sampling ($T > 0$), a model might give a great answer once and gibberish five seconds later.
2. **Base Models vs. Instruct Models**: A pre-trained base model is an autoregressive next-token predictor, not an instruction-following chatbot. If you ask a base model `"What is 2 + 2?"`, it might complete the sentence with `"What is 3 + 3? What is 4 + 4?"` because it treats your prompt as an exam worksheet rather than a conversational question!

### B. The Log-Likelihood Method (How MMLU Works)
To measure what a base model actually *knows* without confusing it with conversational instructions, researchers use **Log-Likelihood Ranking**:
- Prompt: `"The capital of France is"`
- Candidate A: `" Berlin"`
- Candidate B: `" Paris"`
- Candidate C: `" Madrid"`

We feed each combined sequence into the model and calculate the model's exact probability for each choice:
$$\log P(\text{Choice} \mid \text{Prompt}) = \sum_{t=1}^{|\text{Choice}|} \log P(\text{token}_t \mid \text{Prompt}, \text{token}_{<t})$$

The choice with the highest probability is selected. If the model assigns higher probability to `" Paris"` than `" Berlin"`, it has demonstrably learned that geographical fact!

---

## 3. HOW Does the Math Operate?

### A. The Three Core Compression Metrics
$$\begin{aligned}
\text{Cross-Entropy Loss: } \mathcal{L} &= -\frac{1}{N} \sum_{i=1}^{N} \log P(x_i \mid x_{<i}) \\
\text{Perplexity: } \text{PPL} &= e^{\mathcal{L}} \\
\text{Bits per Token: } \text{BPT} &= \frac{\mathcal{L}}{\ln(2)} = \log_2(\text{PPL})
\end{aligned}$$

| Metric | Random Guessing (Vocab = 512) | Our Trained Model | Meaning |
| :--- | :--- | :--- | :--- |
| **Loss** | $\ln(512) \approx 6.24$ | $2.43$ | Cross-entropy error per token |
| **Perplexity** | $512.0$ | $11.36$ | Model is choosing between only ~11 likely tokens |
| **Bits per Token** | $9.0$ bits | $3.5$ bits | Information needed to store text under this model |

### B. The Critical Educational Lesson: Data Determines Capabilities
When we ran our empirical comparison between the untrained baseline and our model trained on `educational_science_corpus.txt`:
- **Validation Perplexity**: Dropped by **-97.8%** (from 519.44 down to 11.36). The model became extremely good at predicting words like *gravity*, *cells*, and *energy*.
- **Math & Coding Probes**: The model scored near chance level (0% to 25%).
- **Why?**: The model was never trained on arithmetic or code! It is impossible for a neural network to deduce multiplication tables or Python syntax if those patterns are never present in its pre-training corpus.
- **Takeaway**: Models do not have general magical intelligence; they internalize statistical patterns from their training data. To get reasoning and math, we must pre-train on reasoning and math datasets, followed by **Supervised Fine-Tuning (SFT)**!

---

## 4. How We Verified It

1. **Automated Unit & Integration Tests** (`tests/evaluation/`):
   - `test_loss_eval.py`: Verified mathematical limits of perplexity and bits-per-token, sliding window handling, and zero-token safety.
   - `test_multiple_choice.py`: Tested log-likelihood ranking and length normalization against deterministic mock models.
   - `test_harness.py`: Verified multi-domain probe filtering, report generation, and JSON serialization.
2. **Benchmark Execution** (`scripts/run_phase6_eval_demo.py`):
   - Completed 17 multi-domain probes + validation perplexity computation in **0.26 seconds** on CPU.

---

## 5. NEXT: The Comprehension Gate Checkpoint

According to **Directive 7** of `AGENTS.md` and **Prompt 2**:
> **The Comprehension Gate**: Do not begin Phase 7 (Model Registry) or any phase beyond it until the user can unprompted explain:
> 1. What a token is
> 2. What attention computes
> 3. What a training step does
> 4. What a checkpoint contains

We have now reached the **Comprehension Gate** at the boundary of Phase 6 $\to$ Phase 7. We must pause here to review and test these 4 fundamental pillars together!
