# Educational Guide: Phase 2 — Tokenizer

Welcome to **Phase 2** of **Libra**!

In this phase, we demystify the bridge between human language and neural networks: **Tokenization**.

---

## 1. WHAT Was Built?

1. **`BaseTokenizer`** (`packages/core/tokenizer/base.py`): The abstract blueprint defining `encode()`, `decode()`, `special_tokens`, `save()`, and `load()`.
2. **`EducationalBPETokenizer`** (`packages/core/tokenizer/educational_bpe.py`): A transparent, pure-Python implementation of the **Byte-Pair Encoding (BPE)** algorithm built from mathematical first principles.
3. **`HFTokenizer`** (`packages/core/tokenizer/hf_tokenizer.py`): A high-performance production wrapper utilizing Hugging Face's Rust-backed `tokenizers` library.
4. **Demonstration Lab** (`scripts/run_phase2_tokenizer_demo.py`): Interactive script comparing character vs. subword compression ratios and visual merge hierarchies.

---

## 2. WHY Do We Need Subword Tokenization?

When humans look at text, we see words and sentences. Neural networks only multiply numerical matrices. There are three classic ways to convert text to numbers:

| Approach | How It Works | The Fatal Flaw |
| :--- | :--- | :--- |
| **Character-Level** | Every letter is a token (`'c', 'a', 't'`). Vocab size ~256. | Sequence lengths explode! A 500-word essay requires ~3,000 tokens. The model wastes its context window spelling words rather than understanding meaning. |
| **Word-Level** | Every distinct word is a token (`"cat"`, `"dog"`). Vocab size 500,000+. | **The Out-Of-Vocabulary (OOV) Problem**: If the model encounters a typo, compound word, or new slang (`"un-friend"`, `"ChatGPT"`), it has never seen it and outputs `<UNK>` (Unknown). |
| **Subword (BPE)** *(Modern Standard)* | Frequent words become single tokens, while rare words are broken into recognizable chunks (`"un" + "happi" + "ness"`). | **Zero OOV errors, compact sequences, and manageable vocabulary sizes (32k to 128k).** |

---

## 3. HOW Does Byte-Pair Encoding (BPE) Work?

Originally developed in 1994 as a data compression algorithm (by Philip Gage), BPE was adapted for Natural Language Processing by Sennrich et al. in 2015.

### The Algorithm:
1. **Initialize Base Vocabulary**: Start with 256 individual byte values (covering all ASCII characters and UTF-8 bytes) plus special tokens (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`).
2. **Count Pair Frequencies**: Scan the training text and count every adjacent pair of tokens.
   For example, in `"the other mother"`:
   - `('t', 'h')` appears 3 times.
   - `('h', 'e')` appears 2 times.
3. **Merge the Most Frequent Pair**:
   - Merge `('t', 'h')` $\to$ new token `'th'` (assign new ID, e.g. 260).
   - The text becomes: `"'th'e o'th'er mo'th'er"`.
4. **Repeat**: In the next iteration, `'th' + 'e'` might merge into `'the'`.
5. **Result**: Frequent words like `"the"`, `"is"`, `"attention"` become single tokens, while rare words like `"discombobulated"` break down into subwords `["dis", "com", "bob", "ulated"]`.

### Why Byte-Level BPE is Lossless
Because the base vocabulary begins with all 256 possible bytes (0 to 255):
- Any language (English, Hindi, Mandarin, Arabic) can be tokenized.
- Emojis (🎉, 🚀, ♎) and code symbols are preserved.
- The model will **never** throw an Out-Of-Vocabulary error!

---

## 4. Why Do We Need Special Tokens?

Special tokens are non-textual control markers that provide structural instructions to the transformer:

- **`<BOS>` (Beginning of Sequence, ID: 2)**: Tells the model: *"A new prompt or document begins here. Reset your context."*
- **`<EOS>` (End of Sequence, ID: 3)**: Tells the model: *"Stop generating! The answer is complete."* Without `<EOS>`, the model would keep rambling forever until hitting its maximum token limit.
- **`<PAD>` (Padding, ID: 0)**: When training batches of different sentence lengths (e.g. 10 tokens and 25 tokens), we pad the shorter sentences with `<PAD>` so they form a rectangular tensor $(B, T)$ for matrix multiplication.
- **`<UNK>` (Unknown, ID: 1)**: Fallback token when an input symbol cannot be recognized.

---

## 5. How We Verified It

1. `test_educational_bpe_train_merges`: Verified that training BPE iteratively compresses text length.
2. `test_educational_bpe_roundtrip_lossless`: Proved `decode(encode(text)) == text` across standard English, punctuation, and out-of-training words.
3. `test_educational_bpe_special_tokens`: Proved `<BOS>` and `<EOS>` are properly injected and stripped during generation.
4. `test_hf_tokenizer_save_and_load`: Proved state serialization and Rust-backed performance.

---

## 6. NEXT: What Comes Next?

In **Phase 3: Data Pipeline**, we will build an automated pipeline to clean, deduplicate, shard, and stream datasets for pre-training, strictly respecting our **15 GB storage quota**.