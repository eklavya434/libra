"""
Libra Phase 2 - Educational Tokenizer Demonstration
Visualizes:
  1. Character vs. Word vs. Subword Tokenization
  2. BPE Pair Frequencies & Merge Mechanics
  3. Compression Ratio Comparisons
  4. Hugging Face Production Byte-Level BPE
"""

import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.core.tokenizer.educational_bpe import EducationalBPETokenizer
from packages.core.tokenizer.hf_tokenizer import HFTokenizer


def main() -> None:
    print("=" * 75)
    print("LIBRA PHASE 2: TOKENIZER LABORATORY")
    print("Demystifying Characters, Words, Subwords, and Byte-Pair Encoding (BPE)")
    print("=" * 75)

    sample_corpus = (
        "Libra is an educational laboratory and AI assistant built from first principles. "
        "Large language models do not read raw words; they read integer tokens. "
        "Attention mechanisms process token embeddings to predict the next token. "
        "Through byte-pair encoding, frequent subword chunks merge into single tokens. "
        "This compression allows transformers to process much longer documents efficiently."
    )

    print(
        f"\n1. Training Corpus: {len(sample_corpus)} characters ({len(sample_corpus.split())} words)."
    )

    # Part A: Educational BPE Tokenizer
    print("\n" + "-" * 75)
    print("PART A: First-Principles Educational BPE Tokenizer (Pure Python)")
    print("-" * 75)
    edu_tok = EducationalBPETokenizer()
    print(f"Base Vocabulary Size: {edu_tok.vocab_size} (4 Special Tokens + 256 Raw Bytes)")

    num_merges = 30
    print(f"Training BPE for {num_merges} merges...")
    edu_tok.train(sample_corpus, num_merges=num_merges, min_frequency=2)

    print(
        f"Post-Training Vocabulary Size: {edu_tok.vocab_size} (+{len(edu_tok.merges)} new subwords learned)"
    )
    print("\nSample Learned Subword Merges:")
    for i, (pair, new_id) in enumerate(list(edu_tok.merges.items())[:8], 1):
        p1_bytes = edu_tok.vocab[pair[0]]
        p2_bytes = edu_tok.vocab[pair[1]]
        merged_bytes = edu_tok.vocab[new_id]
        print(
            f"  Merge #{i:02d}: {p1_bytes.decode('latin1')!r} + {p2_bytes.decode('latin1')!r} "
            f"--> {merged_bytes.decode('latin1')!r} (Token ID: {new_id})"
        )

    # Compression Comparison
    test_sentence = "Attention mechanisms process token embeddings efficiently."
    raw_chars = len(test_sentence)
    encoded_tokens = edu_tok.encode(test_sentence)
    compression = (1.0 - (len(encoded_tokens) / raw_chars)) * 100

    print(f'\nTest Sentence: "{test_sentence}"')
    print(f"  • Raw Characters: {raw_chars}")
    print(f"  • Subword Tokens: {len(encoded_tokens)}")
    print(f"  • Compression:    {compression:.1f}% fewer steps for the Transformer!")
    print(f"  • Token IDs:      {encoded_tokens[:12]}...")

    # Verify Lossless Roundtrip
    decoded_text = edu_tok.decode(encoded_tokens)
    assert decoded_text == test_sentence
    print("  • Roundtrip Check: PASS (Decoded exactly matches original string)")

    # Part B: Production Hugging Face Tokenizer
    print("\n" + "-" * 75)
    print("PART B: Production Hugging Face Byte-Level BPE (Rust-Backed)")
    print("-" * 75)
    hf_tok = HFTokenizer()
    hf_tok.train_from_iterator([sample_corpus], vocab_size=350)
    print(f"HF Tokenizer Trained! Vocab Size: {hf_tok.vocab_size}")

    hf_encoded = hf_tok.encode(test_sentence, add_special_tokens=True)
    print("HF Encoded with Special Tokens (<BOS> ... <EOS>):")
    print(f"  • Token IDs: {hf_encoded}")
    print(f"  • First Token: {hf_encoded[0]} (<BOS>)")
    print(f"  • Last Token:  {hf_encoded[-1]} (<EOS>)")
    print(f'  • Decoded:     "{hf_tok.decode(hf_encoded, skip_special_tokens=True)}"')

    # Save to disk for inspection
    save_dir = "data/tokenized"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "libra_educational_bpe.json")
    edu_tok.save(save_path)
    print(f"\nSaved Tokenizer State to: {save_path} ({os.path.getsize(save_path) / 1024:.1f} KB)")
    print("=" * 75)


if __name__ == "__main__":
    main()
