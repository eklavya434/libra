"""
Unit test proving that Causal Masking prevents future information leakage.
"""

import torch
from packages.models.config import TinyTransformerConfig
from packages.models.transformer import TinyTransformerLM


def test_causal_attention_mask_strictly_enforces_past_only():
    """Mathematically proves that altering future tokens does not change logits for past tokens."""
    config = TinyTransformerConfig(
        vocab_size=64,
        max_context_length=16,
        d_model=32,
        n_heads=2,
        n_layers=2,
        dropout=0.0,
    )
    model = TinyTransformerLM(config)
    model.eval()

    # Create sequence A: [5, 12, 18, 25]
    seq_a = torch.tensor([[5, 12, 18, 25]], dtype=torch.long)

    # Create sequence B: [5, 12, 99, 99] (identical first two tokens, completely different future tokens)
    # Note: 99 is out of vocab for 64, so let's use 33, 44 within vocab
    seq_b = torch.tensor([[5, 12, 33, 44]], dtype=torch.long)

    with torch.no_grad():
        logits_a, _ = model(seq_a)
        logits_b, _ = model(seq_b)

    # The prediction at position 0 (given token 5) MUST be identical in both sequences
    assert torch.allclose(logits_a[:, 0, :], logits_b[:, 0, :], atol=1e-5), (
        "Causality violation! Token at position 0 was affected by future tokens."
    )

    # The prediction at position 1 (given tokens [5, 12]) MUST be identical in both sequences
    assert torch.allclose(logits_a[:, 1, :], logits_b[:, 1, :], atol=1e-5), (
        "Causality violation! Token at position 1 was affected by future tokens."
    )