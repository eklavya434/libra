"""
Unit tests for TinyTransformerLM output tensor shapes and loss computation.
"""

import torch
from packages.models.config import TinyTransformerConfig
from packages.models.transformer import TinyTransformerLM


def test_transformer_forward_shape():
    config = TinyTransformerConfig(
        vocab_size=64,
        max_context_length=32,
        d_model=32,
        n_heads=2,
        n_layers=1,
    )
    model = TinyTransformerLM(config)

    batch_size = 3
    seq_len = 16
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(x)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is None


def test_transformer_forward_with_loss():
    config = TinyTransformerConfig(
        vocab_size=64,
        max_context_length=32,
        d_model=32,
        n_heads=2,
        n_layers=1,
    )
    model = TinyTransformerLM(config)

    batch_size = 2
    seq_len = 10
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(x, targets)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is not None
    assert loss.item() > 0.0