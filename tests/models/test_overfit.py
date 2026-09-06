"""
Unit test verifying that the model can successfully memorize a tiny batch (overfit test).
Proves that loss, gradient computation, and backpropagation are mathematically functioning.
"""

import torch
import torch.optim as optim
from packages.models.config import TinyTransformerConfig
from packages.models.transformer import TinyTransformerLM


def test_model_overfits_tiny_batch():
    config = TinyTransformerConfig(
        vocab_size=32,
        max_context_length=8,
        d_model=32,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
    )
    model = TinyTransformerLM(config)
    optimizer = optim.AdamW(model.parameters(), lr=0.01)

    # Fixed tiny batch
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
    y = torch.tensor([[2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)

    model.train()
    _, initial_loss = model(x, y)
    assert initial_loss is not None
    init_val = initial_loss.item()

    # Train for 50 steps
    for _ in range(50):
        optimizer.zero_grad()
        _, loss = model(x, y)
        assert loss is not None
        loss.backward()
        optimizer.step()

    final_val = loss.item()

    # The loss MUST decrease significantly
    assert final_val < init_val * 0.5, f"Loss did not decrease sufficiently: {init_val} -> {final_val}"