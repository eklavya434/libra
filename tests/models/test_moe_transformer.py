"""
Tests for MoETransformerLM and MoEEvaluator (Phase 45)
"""

import torch

from packages.evaluation.moe_eval import MoEEvaluator
from packages.models.moe_transformer import MoETransformerConfig, MoETransformerLM


def create_toy_moe_model() -> MoETransformerLM:
    torch.manual_seed(42)
    cfg = MoETransformerConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        max_context_length=32,
        hidden_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        aux_loss_coef=0.02,
        noisy_gating=False,
    )
    return MoETransformerLM(cfg)


def test_moe_transformer_forward_and_loss():
    model = create_toy_moe_model()
    model.train()

    idx = torch.randint(0, 64, (2, 6))
    targets = torch.randint(0, 64, (2, 6))

    logits, total_loss, task_loss, aux_loss, routing = model(idx, targets=targets)

    assert logits.shape == (2, 6, 64)
    assert total_loss is not None
    assert task_loss is not None
    assert aux_loss is not None
    assert torch.isclose(total_loss, task_loss + aux_loss)
    assert len(routing) == 2  # 2 MoE layers

    # Backpropagation check
    total_loss.backward()
    router_w = model.blocks[0].ffn.router.w_gate.weight
    assert router_w.grad is not None
    assert not torch.isnan(router_w.grad).any()


def test_moe_transformer_parameter_counting_sparsity():
    model = create_toy_moe_model()
    stats = model.count_parameters()

    assert stats["total_parameters"] > stats["active_parameters_per_token"]
    assert stats["sparsity_ratio"] > 1.0
    assert stats["compute_savings_pct"] > 0.0
    assert stats["num_experts"] == 4
    assert stats["num_experts_per_tok"] == 2


def test_moe_evaluator_trace():
    model = create_toy_moe_model()
    idx = torch.randint(0, 64, (1, 5))
    vocab = [f"tok_{i}" for i in range(64)]

    analysis = MoEEvaluator.analyze_token_routing(model, idx, vocab_tokens=vocab)

    assert analysis["prompt_tokens_count"] == 5
    assert analysis["num_experts"] == 4
    assert len(analysis["tokens"]) == 5
    assert len(analysis["layer_stats"]) == 2

    tok0 = analysis["tokens"][0]
    assert "token_str" in tok0
    assert len(tok0["layers"]) == 2
    assert len(tok0["layers"][0]["selected_experts"]) == 2


def test_moe_evaluator_utilization():
    model = create_toy_moe_model()
    seqs = [torch.randint(0, 64, (1, 8)) for _ in range(3)]

    util = MoEEvaluator.compute_expert_utilization(model, seqs)

    assert util["total_tokens_evaluated"] == 24
    assert len(util["expert_counts"]) == 4
    assert len(util["expert_fractions"]) == 4
    assert abs(sum(util["expert_fractions"]) - 1.0) < 1e-3
    assert "coefficient_of_variation" in util
    assert "parameter_efficiency" in util
