"""
Unit tests for RMSNorm.
"""

import torch

from packages.models.components.rmsnorm import RMSNorm


def test_rmsnorm_forward_shape_and_scale():
    dim = 64
    norm = RMSNorm(dim=dim)

    # Input tensor with arbitrary scale
    x = torch.randn(2, 10, dim) * 5.0 + 2.0
    out = norm(x)

    assert out.shape == (2, 10, dim)

    # Root mean square along the last dimension should be close to 1.0 (since gamma is all ones)
    rms_vals = torch.sqrt(out.pow(2).mean(dim=-1))
    assert torch.allclose(rms_vals, torch.ones_like(rms_vals), atol=1e-3)


def test_rmsnorm_gradient_flow():
    dim = 32
    norm = RMSNorm(dim=dim)
    x = torch.randn(2, 5, dim, requires_grad=True)

    out = norm(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert norm.weight.grad is not None
    assert not torch.isnan(x.grad).any()
