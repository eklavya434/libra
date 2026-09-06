"""
Unit test proving mathematical equivalence of Gradient Accumulation.
"""

import copy

import torch
from torch import nn


def test_gradient_accumulation_matches_full_batch():
    """Mathematically proves that accumulating over 2 micro-steps equals a single batch of size 2."""
    # Simple linear model
    torch.manual_seed(42)
    model_single = nn.Linear(4, 2, bias=False)
    model_accum = copy.deepcopy(model_single)

    # Two micro-batches (batch_size = 1 each)
    x1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    y1 = torch.tensor([[1.0, 0.0]])

    x2 = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
    y2 = torch.tensor([[0.0, 1.0]])

    criterion = nn.MSELoss()

    # Approach 1: Combined full batch (batch_size = 2)
    x_full = torch.cat([x1, x2], dim=0)
    y_full = torch.cat([y1, y2], dim=0)

    model_single.zero_grad()
    loss_single = criterion(model_single(x_full), y_full)
    loss_single.backward()

    # Approach 2: Gradient Accumulation (2 micro-steps of size 1)
    model_accum.zero_grad()
    loss_accum1 = criterion(model_accum(x1), y1) / 2.0
    loss_accum1.backward()

    loss_accum2 = criterion(model_accum(x2), y2) / 2.0
    loss_accum2.backward()

    # Gradients accumulated across micro-steps MUST equal single batch gradients!
    assert torch.allclose(model_single.weight.grad, model_accum.weight.grad, atol=1e-5)
