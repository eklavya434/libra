"""
Tests for Medusa Speculative Decoding Components (Phase 46)
"""

import torch

from packages.models.components.medusa import MedusaHead, MedusaModel
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM


def test_medusa_head_forward():
    d_model = 32
    vocab_size = 64
    head = MedusaHead(d_model=d_model, vocab_size=vocab_size, hidden_dim=48)

    h = torch.randn(2, 5, d_model)
    logits = head(h)

    assert logits.shape == (2, 5, vocab_size)
    assert not torch.isnan(logits).any()


def test_medusa_model_forward_and_loss():
    torch.manual_seed(42)
    cfg = ModernTransformerConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        max_context_length=32,
        hidden_dim=64,
    )
    base = ModernTransformerLM(cfg)
    medusa = MedusaModel(base_model=base, num_heads=3)

    idx = torch.randint(0, 64, (2, 8))
    base_logits, medusa_logits, normed_h = medusa.forward_with_medusa(idx)

    assert base_logits.shape == (2, 8, 64)
    assert len(medusa_logits) == 3
    for k, h_logits in enumerate(medusa_logits):
        assert h_logits.shape == (2, 8, 64)
    assert normed_h.shape == (2, 8, 32)

    # Compute Medusa discounted loss
    targets = torch.randint(0, 64, (2, 8))
    loss, head_losses = medusa.compute_medusa_loss(medusa_logits, targets=targets, decay=0.8)

    assert loss.ndim == 0
    assert loss.item() > 0.0
    assert len(head_losses) == 3

    # Check backpropagation on Medusa heads
    loss.backward()
    assert medusa.medusa_heads[0].proj.weight.grad is not None
