"""
Tests for Model Shrinking & Architecture Pruning (Phase 44)
"""

import torch

from packages.models.components.model_shrinking import ModelShrinker
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM


def test_student_config_creation():
    teacher_cfg = ModernTransformerConfig(
        vocab_size=256,
        d_model=64,
        n_layers=4,
        n_heads=4,
        max_context_length=128,
        hidden_dim=128,
    )
    student_cfg = ModelShrinker.create_student_config(teacher_cfg, n_layers=2)
    assert student_cfg.n_layers == 2
    assert student_cfg.d_model == 64
    assert student_cfg.n_heads == 4
    assert student_cfg.vocab_size == 256


def test_select_layer_indices():
    # 4 layers down to 2
    indices_4_to_2 = ModelShrinker.select_layer_indices(4, 2)
    assert len(indices_4_to_2) == 2
    assert indices_4_to_2 == [0, 3] or indices_4_to_2 == [0, 2]

    # 6 layers down to 3
    indices_6_to_3 = ModelShrinker.select_layer_indices(6, 3)
    assert len(indices_6_to_3) == 3
    assert indices_6_to_3[0] == 0
    assert indices_6_to_3[-1] == 5


def test_shrink_layers_execution_and_weights():
    torch.manual_seed(42)
    teacher_cfg = ModernTransformerConfig(
        vocab_size=128,
        d_model=32,
        n_layers=4,
        n_heads=2,
        max_context_length=64,
        hidden_dim=64,
    )
    teacher = ModernTransformerLM(teacher_cfg)

    student = ModelShrinker.shrink_layers(teacher, target_layer_indices=[0, 2])
    assert student.config.n_layers == 2
    assert len(student.blocks) == 2

    # Verify student block 0 matches teacher block 0
    s_b0_weight = student.blocks[0].attn.q_proj.weight
    t_b0_weight = teacher.blocks[0].attn.q_proj.weight
    assert torch.equal(s_b0_weight, t_b0_weight)

    # Verify student block 1 matches teacher block 2
    s_b1_weight = student.blocks[1].attn.q_proj.weight
    t_b2_weight = teacher.blocks[2].attn.q_proj.weight
    assert torch.equal(s_b1_weight, t_b2_weight)

    # Test forward pass
    dummy_x = torch.randint(0, 128, (2, 10))
    logits, loss = student(dummy_x)
    assert logits.shape == (2, 10, 128)
    assert loss is None


def test_forward_with_hidden_states():
    cfg = ModernTransformerConfig(
        vocab_size=64,
        d_model=32,
        n_layers=3,
        n_heads=2,
        max_context_length=32,
        hidden_dim=64,
    )
    model = ModernTransformerLM(cfg)
    idx = torch.randint(0, 64, (1, 8))

    logits, hiddens, final_h = ModelShrinker.forward_with_hidden_states(model, idx)
    assert logits.shape == (1, 8, 64)
    assert len(hiddens) == 3
    for h in hiddens:
        assert h.shape == (1, 8, 32)
    assert final_h.shape == (1, 8, 32)


def test_compute_compression_stats():
    teacher_cfg = ModernTransformerConfig(
        vocab_size=128,
        d_model=32,
        n_layers=4,
        n_heads=2,
        max_context_length=64,
        hidden_dim=64,
    )
    teacher = ModernTransformerLM(teacher_cfg)
    student = ModelShrinker.shrink_layers(teacher, target_layer_indices=[0, 2])

    stats = ModelShrinker.compute_compression_stats(teacher, student)
    assert stats["teacher_params"] > stats["student_params"]
    assert stats["compression_ratio"] > 1.0
    assert stats["param_reduction_pct"] > 25.0
    assert stats["teacher_layers"] == 4
    assert stats["student_layers"] == 2
