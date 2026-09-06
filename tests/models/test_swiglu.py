"""
Unit tests for SwiGLU Gated Feed-Forward layer.
"""

import torch

from packages.models.components.swiglu import SwiGLU


def test_swiglu_shape_and_gating():
    d_model = 64
    hidden_dim = 128
    swiglu = SwiGLU(d_model=d_model, hidden_dim=hidden_dim)

    batch_size = 3
    seq_len = 8
    x = torch.randn(batch_size, seq_len, d_model)

    out = swiglu(x)
    assert out.shape == (batch_size, seq_len, d_model)

    # Test gradient propagation through both gate and value paths
    loss = out.sum()
    loss.backward()
    assert swiglu.w_gate.weight.grad is not None
    assert swiglu.w_up.weight.grad is not None
    assert swiglu.w_down.weight.grad is not None
