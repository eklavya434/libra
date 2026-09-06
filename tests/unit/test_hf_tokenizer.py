"""
Unit tests for HFTokenizer wrapper.
"""

import os

from packages.core.tokenizer.hf_tokenizer import HFTokenizer


def test_hf_tokenizer_train_and_encode():
    tok = HFTokenizer()
    sample_texts = [
        "Libra is an educational LLM laboratory.",
        "Tokenizers convert words into numbers.",
        "Deep learning uses neural networks.",
    ]
    tok.train_from_iterator(sample_texts, vocab_size=300)

    assert tok.vocab_size >= 256

    test_str = "Libra is an educational laboratory."
    encoded = tok.encode(test_str)
    decoded = tok.decode(encoded)

    assert isinstance(encoded, list)
    assert len(encoded) > 0
    assert decoded == test_str


def test_hf_tokenizer_special_tokens():
    tok = HFTokenizer()
    tok.train_from_iterator(["Sample sentence for special tokens."], vocab_size=300)

    encoded = tok.encode("Hello", add_special_tokens=True)
    assert encoded[0] == tok.bos_token_id
    assert encoded[-1] == tok.eos_token_id


def test_hf_tokenizer_save_and_load(tmp_path):
    tok = HFTokenizer()
    tok.train_from_iterator(["Testing saving and loading tokenizer state."], vocab_size=300)

    save_path = os.path.join(tmp_path, "hf_tokenizer.json")
    tok.save(save_path)

    loaded_tok = HFTokenizer.load(save_path)
    test_str = "Testing saving."
    assert tok.encode(test_str) == loaded_tok.encode(test_str)
    assert loaded_tok.decode(loaded_tok.encode(test_str)) == test_str
