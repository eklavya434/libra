"""
Tests for First-Principles Transformer Reward Model (Phase 22)
"""

import torch

from packages.models.modern_config import ModernTransformerConfig
from packages.models.reward_model import TransformerRewardModel


def get_small_reward_model() -> TransformerRewardModel:
    torch.manual_seed(42)
    cfg = ModernTransformerConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        max_context_length=64,
        hidden_dim=64,
    )
    return TransformerRewardModel(cfg)


def test_reward_model_forward():
    model = get_small_reward_model()
    input_ids = torch.randint(0, 64, (3, 16))
    scores = model(input_ids)

    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (3,)


def test_reward_model_end_indices():
    model = get_small_reward_model()
    input_ids = torch.randint(0, 64, (2, 20))
    end_indices = torch.tensor([10, 15])
    scores = model(input_ids, end_indices=end_indices)

    assert scores.shape == (2,)


def test_reward_model_loss_and_telemetry():
    model = get_small_reward_model()
    chosen_ids = torch.randint(0, 64, (4, 12))
    rejected_ids = torch.randint(0, 64, (4, 12))

    loss, telemetry = model.compute_loss(chosen_ids, rejected_ids)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0.0

    assert 0.0 <= telemetry.accuracy <= 1.0
    assert isinstance(telemetry.reward_margin, float)


def test_reward_model_optimization():
    """Verify that gradient descent increases reward margin on a preference pair."""
    model = get_small_reward_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    chosen_ids = torch.tensor([[1, 2, 3, 4, 5]])
    rejected_ids = torch.tensor([[1, 2, 6, 7, 8]])

    initial_margin = (model(chosen_ids) - model(rejected_ids)).item()

    for _ in range(15):
        optimizer.zero_grad()
        loss, _ = model.compute_loss(chosen_ids, rejected_ids)
        loss.backward()
        optimizer.step()

    final_margin = (model(chosen_ids) - model(rejected_ids)).item()
    # The margin between chosen and rejected should increase
    assert final_margin > initial_margin
