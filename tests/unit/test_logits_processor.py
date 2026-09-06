"""
Tests for ConstrainedLogitsProcessor (packages/core/grammar/logits_processor.py)
"""

import pytest
import torch
from packages.core.grammar.logits_processor import ConstrainedLogitsProcessor


@pytest.fixture
def mock_vocab():
    # Miniature test vocabulary
    vocab = [
        "<eos>",      # 0
        " ",          # 1
        "{",          # 2
        "}",          # 3
        '"',          # 4
        "key",        # 5
        ":",          # 6
        "100",        # 7
        "bad_token",  # 8
    ]
    decode_fn = lambda ids: "".join(vocab[i] for i in ids)
    return vocab, decode_fn


def test_logits_processor_masking_initial_step(mock_vocab):
    vocab, decode_fn = mock_vocab
    processor = ConstrainedLogitsProcessor(
        decode_fn=decode_fn,
        vocab_size=len(vocab),
        eos_token_id=0,
    )

    # Initial step: prefix is empty
    scores = torch.zeros(len(vocab))
    input_ids = torch.tensor([], dtype=torch.long)

    masked = processor(input_ids, scores)

    # At start, "{" or whitespace or '"' should be allowed, but ":" or "}" or "bad_token" should be -inf
    assert masked[vocab.index("{")] == 0.0
    assert masked[vocab.index(" ")] == 0.0
    assert masked[vocab.index(":")] == -float("inf")
    assert masked[vocab.index("bad_token")] == -float("inf")


def test_logits_processor_after_key(mock_vocab):
    vocab, decode_fn = mock_vocab
    processor = ConstrainedLogitsProcessor(
        decode_fn=decode_fn,
        vocab_size=len(vocab),
        eos_token_id=0,
    )

    # Prefix is '{"key"'
    prefix_tokens = [vocab.index("{"), vocab.index('"'), vocab.index("key"), vocab.index('"')]
    input_ids = torch.tensor(prefix_tokens, dtype=torch.long)
    scores = torch.zeros(len(vocab))

    masked = processor(input_ids, scores)

    # After key, expecting ':' or whitespace
    assert masked[vocab.index(":")] == 0.0
    assert masked[vocab.index(" ")] == 0.0
    assert masked[vocab.index("100")] == -float("inf")
    assert masked[vocab.index("{")] == -float("inf")


def test_logits_processor_completion(mock_vocab):
    vocab, decode_fn = mock_vocab
    processor = ConstrainedLogitsProcessor(
        decode_fn=decode_fn,
        vocab_size=len(vocab),
        eos_token_id=0,
    )

    # Prefix is complete: '{}'
    prefix_tokens = [vocab.index("{"), vocab.index("}")]
    input_ids = torch.tensor(prefix_tokens, dtype=torch.long)
    scores = torch.zeros(len(vocab))

    masked = processor(input_ids, scores)

    # When complete, <eos> must be allowed!
    assert masked[vocab.index("<eos>")] == 0.0
    assert masked[vocab.index("bad_token")] == -float("inf")
