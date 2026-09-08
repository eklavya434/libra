# Phase 33: Domain Adaptation & Instruction Fine-Tuning Corpus Pipeline (ChatML, Multi-Turn Packing, and Loss Masking)

## 1. Overview & Pedagogical Purpose
Supervised Fine-Tuning (SFT) transforms a raw causal next-token prediction model into an instruction-following assistant.
However, naive SFT implementation suffers from two major inefficiencies and subtle failure modes:
1. **Backpropagating through Prompts**: If the model computes loss and gradients on user queries and system prompts, it attempts to "memorize" or "predict" the user's input rather than learning to generate high-quality assistant answers.
2. **Padding Token Waste**: Dialogues vary widely in length (e.g. 50 to 500 tokens). Padding each dialogue to `max_length` (e.g. 1024 or 2048) causes 70–90% of GPU/CPU operations to compute attention over `<pad>` tokens that carry no gradient.
3. **Delimiter Ambiguity**: Without standardized delimiters like ChatML (`<|im_start|>role\ncontent<|im_end|>`), the model cannot distinguish between system instructions, user prompts, and its own output, leading to prompt injection vulnerabilities.

Phase 33 implements an end-to-end, zero-cost, CPU-friendly instruction fine-tuning pipeline solving all three challenges.

---

## 2. Core Architecture & Mathematics

### A. Completion-Only Cross-Entropy Loss Masking
In standard language modeling, given a sequence of tokens $x = (x_1, x_2, \dots, x_T)$, the cross-entropy loss is:
$$\mathcal{L}(\theta) = -\frac{1}{T} \sum_{t=1}^T \log P_\theta(x_t \mid x_{<t})$$

In Supervised Instruction Fine-Tuning, we partition the sequence into prompt tokens $\mathcal{P}$ (system prompt + user questions) and completion tokens $\mathcal{C}$ (assistant answers).
We assign label:
$$y_t = \begin{cases} -100 & \text{if } x_t \in \mathcal{P} \\ x_t & \text{if } x_t \in \mathcal{C} \end{cases}$$

PyTorch's `torch.nn.CrossEntropyLoss(ignore_index=-100)` ignores any token with label `-100`, computing loss strictly over assistant output:
$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{|\mathcal{C}|} \sum_{t \in \mathcal{C}} \log P_\theta(x_t \mid x_{<t})$$

### B. ChatML Serialization Template
Every dialogue turn is wrapped in OpenAI/LLaMA-standard ChatML tags:
```text
<|im_start|>system
You are Libra, an educational AI assistant.<|im_end|>
<|im_start|>user
What is attention?<|im_end|>
<|im_start|>assistant
Attention computes weighted sum of values based on query-key similarity.<|im_end|>
```

### C. Multi-Turn Sequence Packing
Instead of padding each sample with hundreds of `<pad>` tokens, sequence packing concatenates multiple independent dialogues $(D_1, D_2, \dots, D_k)$ into a single contiguous context window separated by `<|endoftext|>` delimiters until reaching `max_length`. This eliminates wasted compute and boosts training throughput by 2x–5x on CPU.

---

## 3. Educational Summary (LIBRA Protocol)

- **WHAT**:
  - `packages/training/chat_formatter.py`: `ChatMessage`, `ChatMLFormatter`, and `tokenize_with_loss_masking` (masking prompt tokens with `-100`).
  - `packages/training/sft_dataset.py`: `InstructionDataset` (PyTorch Dataset) and `pack_sequences` (contiguous dialogue packing).
  - `packages/training/domain_corpora.py`: Curated educational dialogues across systems, machine learning, and mathematics.
  - `apps/backend/api/v1/endpoints/corpus.py`: REST API endpoints (`/format_chat`, `/pack`, `/sample_dataset`).
  - `apps/frontend/src/components/CorpusViewer.tsx`: Interactive Next.js component to inspect ChatML formatting, token masks, and packing efficiency.
  - `scripts/run_phase33_corpus_demo.py`: Executable CLI demo.

- **WHY**:
  - Masking prompt tokens prevents the model from wasting gradient capacity memorizing user queries.
  - ChatML provides robust demarcation between prompt and completion.
  - Sequence packing eliminates up to 80% padding token overhead on CPU training budgets.

- **HOW**:
  - Formatted messages are parsed turn by turn; tokens belonging to `system` or `user` are mapped to `labels = -100`, while `assistant` tokens retain their token IDs.
  - Sequence packer tracks cumulative token counts and flushes contiguous chunks with EOS tokens.

- **TEST**:
  - 10 new automated unit and API tests passing in `tests/training/test_chat_formatter.py`, `tests/training/test_sft_dataset.py`, and `tests/api/test_corpus_endpoint.py`.
  - Full test suite passes: **380 tests passed** in ~55s.
  - Next.js production build compiles with zero TypeScript errors.

- **NEXT**:
  - **Phase 34: Public Deployment & Containerized Lab Infrastructure (Docker & Cloud Readiness)**.
