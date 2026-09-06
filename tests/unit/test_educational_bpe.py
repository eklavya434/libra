"""
Unit tests for EducationalBPETokenizer.
"""

import os

from packages.core.tokenizer.educational_bpe import EducationalBPETokenizer


def test_educational_bpe_initial_vocab():
    tok = EducationalBPETokenizer()
    # 4 special tokens + 256 byte tokens = 260
    assert tok.vocab_size == 260
    assert tok.pad_token_id == 0
    assert tok.unk_token_id == 1
    assert tok.bos_token_id == 2
    assert tok.eos_token_id == 3


def test_educational_bpe_train_merges():
    tok = EducationalBPETokenizer()
    training_text = "the cat in the hat saw the other cat and the hat"
    tok.train(training_text, num_merges=10, min_frequency=2)

    # Merges should have occurred
    assert len(tok.merges) > 0
    assert tok.vocab_size > 260

    # Encoding trained text should compress sequence length compared to raw bytes
    raw_bytes_len = len(training_text.encode("utf-8"))
    encoded = tok.encode(training_text)
    assert len(encoded) < raw_bytes_len


def test_educational_bpe_roundtrip_lossless():
    tok = EducationalBPETokenizer()
    training_text = "Transformers are neural network architectures based on self-attention."
    tok.train(training_text, num_merges=15, min_frequency=2)

    test_samples = [
        "Transformers are neural network architectures.",
        "Hello world!",
        "12345 special symbols !@#$%",
        "Even unseen words like extraterrestrial are preserved by byte fallback.",
    ]

    for sample in test_samples:
        encoded = tok.encode(sample)
        decoded = tok.decode(encoded)
        assert decoded == sample, f"Roundtrip failed for: {sample}"


def test_educational_bpe_special_tokens():
    tok = EducationalBPETokenizer()
    text = "Hello Libra"
    encoded = tok.encode(text, add_special_tokens=True)
    assert encoded[0] == tok.bos_token_id
    assert encoded[-1] == tok.eos_token_id

    # Decoding with skip_special_tokens=True should recover exact original text
    decoded = tok.decode(encoded, skip_special_tokens=True)
    assert decoded == text


def test_educational_bpe_save_and_load(tmp_path):
    tok = EducationalBPETokenizer()
    tok.train("apple banana orange apple banana orange", num_merges=5, min_frequency=2)

    test_text = "apple orange banana"
    orig_encoded = tok.encode(test_text)

    save_path = os.path.join(tmp_path, "bpe_tokenizer.json")
    tok.save(save_path)

    loaded_tok = EducationalBPETokenizer.load(save_path)
    loaded_encoded = loaded_tok.encode(test_text)

    assert orig_encoded == loaded_encoded
    assert loaded_tok.decode(loaded_encoded) == test_text
