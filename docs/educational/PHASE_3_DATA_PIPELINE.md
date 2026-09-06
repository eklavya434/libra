# Educational Guide: Phase 3 — Data Pipeline

Welcome to **Phase 3** of **Libra**!

A common misconception among beginners is that Large Language Models are mostly about neural network architecture. In reality:
> **"Data quality is the ceiling of model capability. The model can never be smarter than the statistical quality of the data it learns from."**

---

## 1. WHAT Was Built?

1. **Storage Quota Guardrail** (`packages/data/quota.py`): Automatic disk calculation strictly enforcing our **15 GB limit** across caches and data folders before any ingestion.
2. **Text Cleaning Engine** (`packages/data/cleaning.py`): Stripping unprintable control characters, standardizing Unicode (NFC), collapsing excessive whitespace, and filtering degenerate noise.
3. **Cryptographic Deduplication** (`packages/data/deduplication.py`): SHA-256 paragraph-level exact-match deduplication purging repetitive text.
4. **Binary Sharding Pipeline** (`packages/data/pipeline.py`): End-to-end coordinator that cleans, deduplicates, splits (Train/Val), tokenizes, and saves compact `uint16` binary shard files (`train.bin`, `val.bin`).
5. **Educational Science Corpus & Metadata Catalog** (`data/raw/` & `data/metadata/`): Curated CC0 open reference corpus with complete licensing records.

---

## 2. WHY Do We Need Each Stage?

### A. The Deduplication Dilemma
Internet text scraped from Common Crawl, Reddit, or Wikipedia contains **30% to 50% duplicate content** (boilerplate navigation bars, terms of service, repetitive spam, copied articles).
* **What happens if you train on duplicates?**
  1. The model overfits and memorizes repeated sentences word-for-word instead of generalizing concepts.
  2. The model exhibits "repetition loops" during generation (repeating the same sentence over and over).
* **The Solution**: SHA-256 hash sets ensure every paragraph or document is seen only once during pre-training.

### B. Train vs. Validation Split
- **Training Set (90%)**: The data the model learns from and adjusts its weights against.
- **Validation Set (10%)**: Data the model **never** sees during training.
If training loss drops to 0.2 but validation loss stays high at 3.5, the model is **overfitting** (memorizing). If both drop together, the model is genuinely **learning**!

### C. Why Binary Sharding (`uint16`)?
Why don't training loops just read `.txt` files directly?
1. **Reading raw text is slow**: Parsing strings and running tokenizers during training causes the CPU to bottleneck the GPU.
2. **Memory explosion**: Raw text and 64-bit Python integers take huge memory.
3. **The `uint16` Solution**: Since our vocabulary size is under 65,536, each token is stored in exactly **2 bytes** (`uint16`). Reading 100,000 tokens from a `.bin` file takes less than **1 millisecond**!

---

## 3. HOW Does the Pipeline Operate?

```
Raw Text Ingest
      │
      ▼
Storage Quota Validator (Ensures < 15 GB total usage)
      │
      ▼
Unicode Normalization (NFC) + Control Character Purge
      │
      ▼
SHA-256 Cryptographic Deduplication (Drops repeated blocks)
      │
      ▼
Train/Val Split (90% Train, 10% Validation)
      │
      ▼
Subword Tokenization (via Educational BPE or HF Tokenizer)
      │
      ▼
Binary Array Serialization (np.uint16 -> train.bin, val.bin)
```

---

## 4. Understanding Dataset Licensing

In AI, mixing licenses can create legal liabilities. We strictly catalog every dataset:

| Term | What It Means | Examples |
| :--- | :--- | :--- |
| **Open Data (Public Domain)** | Anyone can use, modify, train on, or monetize without restrictions. | CC0, Public Domain scientific texts. |
| **Permissive Open Source** | Free to use and modify, provided copyright and attribution notices are kept. | Apache 2.0, MIT. |
| **Open Weights / Community** | The model weights are freely downloadable, but subject to specific commercial caps (e.g. 700M monthly active users). | Llama 3 Community License. |
| **Non-Commercial / Research** | Usable only for academic research; commercial deployment is strictly prohibited. | CC BY-NC-4.0. |

---

## 5. How We Verified It

We wrote unit tests in `tests/unit/test_data_pipeline.py`:
1. `test_storage_quota_validation`: Confirmed operations above 15 GB are blocked with `QuotaExceededError`.
2. `test_text_cleaning`: Confirmed control characters and degenerate lines are purged.
3. `test_deduplication`: Confirmed duplicate blocks are removed and exact duplication rates are reported.
4. `test_pipeline_end_to_end`: Confirmed lossless roundtrip reading from `train.bin` and `val.bin`.

---

## 6. NEXT: What Comes Next?

In **Phase 4: Improved Transformer**, we will upgrade our foundational model architecture with modern components used in state-of-the-art models like Llama 3 and Mistral:
- **Rotary Positional Embeddings (RoPE)** (replacing absolute positional embeddings).
- **RMSNorm** (Root Mean Square Layer Normalization for faster, simpler scaling).
- **SwiGLU** (Swish Gated Linear Unit activation in the MLP).