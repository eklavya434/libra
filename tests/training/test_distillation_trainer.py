"""
Tests for Distillation Trainer & Evaluation (Phase 44)
"""

import torch

from packages.evaluation.distillation_eval import DistillationEvaluator
from packages.models.components.model_shrinking import ModelShrinker
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.training.distillation_trainer import DistillationConfig, DistillationTrainer


def create_models() -> tuple[ModernTransformerLM, ModernTransformerLM]:
    torch.manual_seed(42)
    teacher_cfg = ModernTransformerConfig(
        vocab_size=64,
        d_model=32,
        n_layers=4,
        n_heads=2,
        max_context_length=32,
        hidden_dim=64,
    )
    teacher = ModernTransformerLM(teacher_cfg)
    student = ModelShrinker.shrink_layers(teacher, target_layer_indices=[0, 2])
    return teacher, student


def test_teacher_frozen_in_trainer():
    teacher, student = create_models()
    trainer = DistillationTrainer(student_model=student, teacher_model=teacher)

    assert not teacher.training
    for param in teacher.parameters():
        assert not param.requires_grad

    assert student.training
    for param in student.parameters():
        assert param.requires_grad


def test_soft_loss_zero_when_distributions_identical():
    logits = torch.randn(2, 5, 20)
    temperature = 2.0
    soft_loss, kl_div = DistillationTrainer.compute_soft_loss(logits, logits, temperature)

    assert kl_div.item() < 1e-5
    assert soft_loss.item() < 1e-5


def test_soft_loss_tau_squared_scaling():
    torch.manual_seed(123)
    s_logits = torch.randn(2, 4, 10)
    t_logits = torch.randn(2, 4, 10)

    soft_loss_2, kl_2 = DistillationTrainer.compute_soft_loss(s_logits, t_logits, temperature=2.0)
    assert torch.isclose(soft_loss_2, kl_2 * 4.0, rtol=1e-4)

    soft_loss_3, kl_3 = DistillationTrainer.compute_soft_loss(s_logits, t_logits, temperature=3.0)
    assert torch.isclose(soft_loss_3, kl_3 * 9.0, rtol=1e-4)


def test_distillation_train_step_decreases_loss():
    teacher, student = create_models()
    config = DistillationConfig(temperature=2.0, alpha=0.5, lr=0.01)
    trainer = DistillationTrainer(student_model=student, teacher_model=teacher, config=config)

    input_ids = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))

    # Save initial teacher parameters to verify they never change
    teacher_param_before = teacher.blocks[0].attn.q_proj.weight.clone()

    telemetry_history = trainer.train_sequence(input_ids, targets, steps=10)

    assert len(telemetry_history) == 10
    assert telemetry_history[-1].total_loss < telemetry_history[0].total_loss
    assert telemetry_history[-1].step == 10

    # Ensure teacher was not modified
    assert torch.equal(teacher.blocks[0].attn.q_proj.weight, teacher_param_before)


def test_intermediate_hidden_loss():
    teacher, student = create_models()
    config = DistillationConfig(
        temperature=2.0,
        alpha=0.5,
        lr=0.01,
        hidden_loss_weight=0.5,
    )
    trainer = DistillationTrainer(student_model=student, teacher_model=teacher, config=config)

    input_ids = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))

    telemetry = trainer.train_step(input_ids, targets)
    assert telemetry.hidden_loss >= 0.0
    assert telemetry.total_loss > 0.0


def test_distillation_evaluator_dark_knowledge_and_speedup():
    teacher, student = create_models()
    input_ids = torch.randint(0, 64, (1, 6))

    # Dark knowledge analysis
    dark_result = DistillationEvaluator.analyze_dark_knowledge(
        teacher, student, input_ids, temperatures=[1.0, 5.0], top_k=3
    )
    assert len(dark_result["temperature_analysis"]) == 2
    t1 = dark_result["temperature_analysis"][0]
    t5 = dark_result["temperature_analysis"][1]
    # Higher temperature has higher or equal entropy (more uniform/dark knowledge)
    assert t5["teacher_entropy"] >= t1["teacher_entropy"]

    # Speed and agreement benchmark
    seqs = [torch.randint(0, 64, (1, 8)) for _ in range(3)]
    bench = DistillationEvaluator.benchmark_speed_and_agreement(teacher, student, seqs, runs=2)
    assert bench["total_sequences"] == 3
    assert bench["speedup_ratio"] > 0.0
    assert 0.0 <= bench["top1_agreement_pct"] <= 100.0
