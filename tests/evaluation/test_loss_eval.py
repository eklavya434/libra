import math

import pytest
import torch

from packages.evaluation.loss_eval import (
    LossMetrics,
    compute_bits_per_token,
    compute_perplexity,
    evaluate_tokens_loss,
)
from packages.models.modern_transformer import ModernTransformerConfig, ModernTransformerLM


def test_perplexity_and_bits_formulas():
    # If loss is 0.0, perplexity is 1.0 (zero uncertainty)
    assert compute_perplexity(0.0) == pytest.approx(1.0)
    assert compute_bits_per_token(0.0) == pytest.approx(0.0)

    # If loss is ln(4) approx 1.3863, perplexity is 4.0
    loss = math.log(4.0)
    assert compute_perplexity(loss) == pytest.approx(4.0)
    assert compute_bits_per_token(loss) == pytest.approx(2.0)  # log2(4) = 2 bits


def test_evaluate_tokens_loss_bounds():
    config = ModernTransformerConfig(
        vocab_size=64,
        max_context_length=32,
        n_layers=1,
        d_model=32,
        n_heads=2,
        hidden_dim=64,
    )
    model = ModernTransformerLM(config)
    model.eval()

    # Generate pseudo-random token sequence of 50 tokens
    torch.manual_seed(42)
    tokens = torch.randint(0, 64, (50,))

    metrics = evaluate_tokens_loss(model, tokens, context_length=16, stride=8)

    assert isinstance(metrics, LossMetrics)
    assert metrics.total_tokens > 0
    # Cross-entropy loss must be positive
    assert metrics.mean_loss > 0.0
    # Perplexity must be >= 1.0
    assert metrics.perplexity >= 1.0
    # Bits per token must be >= 0.0
    assert metrics.bits_per_token >= 0.0


def test_evaluate_short_sequence():
    config = ModernTransformerConfig(
        vocab_size=32,
        max_context_length=16,
        n_layers=1,
        d_model=16,
        n_heads=1,
        hidden_dim=32,
    )
    model = ModernTransformerLM(config)
    tokens = torch.tensor([5])  # Single token, cannot predict next token

    metrics = evaluate_tokens_loss(model, tokens)
    assert metrics.total_tokens == 0
    assert metrics.perplexity == 1.0
