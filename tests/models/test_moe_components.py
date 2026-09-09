"""
Tests for Sparse Mixture of Experts (MoE) Components (Phase 45)
"""

import torch

from packages.models.components.moe import ExpertLayer, MoERouter, SparseMoEBlock


def test_expert_layer_forward():
    d_model = 32
    hidden_dim = 64
    layer = ExpertLayer(d_model=d_model, hidden_dim=hidden_dim)

    x = torch.randn(2, 5, d_model)
    out = layer(x)

    assert out.shape == (2, 5, d_model)
    assert not torch.isnan(out).any()


def test_moe_router_weights_and_selection():
    d_model = 32
    num_experts = 4
    top_k = 2
    router = MoERouter(d_model=d_model, num_experts=num_experts, top_k=top_k, noisy_gating=False)

    x = torch.randn(2, 8, d_model)
    weights, indices, aux_loss, metadata = router(x)

    assert weights.shape == (2, 8, top_k)
    assert indices.shape == (2, 8, top_k)

    # Weights must sum to 1.0 per token
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    # Indices must be in valid expert range
    assert (indices >= 0).all() and (indices < num_experts).all()

    # Aux loss must be non-negative
    assert aux_loss.item() >= 0.0
    assert "expert_fractions" in metadata
    assert len(metadata["expert_fractions"]) == num_experts


def test_moe_router_aux_loss_detects_imbalance():
    torch.manual_seed(42)
    router = MoERouter(d_model=16, num_experts=4, top_k=1, aux_loss_coef=1.0, noisy_gating=False)

    # Uniform-like input
    x_balanced = torch.randn(10, 20, 16)
    _, _, aux_loss_bal, meta_bal = router(x_balanced)

    # Skewed weights heavily toward expert 0
    with torch.no_grad():
        router.w_gate.weight.data[0] = 50.0
        router.w_gate.weight.data[1:] = -50.0

    x_skewed = torch.randn(10, 20, 16)
    _, _, aux_loss_skew, meta_skew = router(x_skewed)

    # Imbalanced routing produces higher CV imbalance
    assert meta_skew["load_imbalance_cv"] >= meta_bal["load_imbalance_cv"]


def test_sparse_moe_block_forward():
    d_model = 32
    num_experts = 4
    top_k = 2
    block = SparseMoEBlock(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
        hidden_dim=64,
        noisy_gating=False,
    )

    x = torch.randn(2, 6, d_model)
    out, aux_loss, routing = block(x)

    assert out.shape == (2, 6, d_model)
    assert aux_loss.ndim == 0  # scalar
    assert "topk_indices" in routing
    assert "topk_weights" in routing
    assert routing["topk_indices"].shape == (2, 6, top_k)
