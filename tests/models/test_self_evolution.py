"""
Tests for Autonomous Self-Evolution Engine (Phase 50 Grand Capstone)
"""

import pytest

from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.models.reasoning.prm import ProcessRewardModel
from packages.training.self_evolution import (
    EvolutionStage,
    EvolutionStepMetric,
    SelfEvolutionConfig,
    SelfEvolutionEngine,
    SelfEvolutionReport,
)


@pytest.fixture
def small_models():
    cfg = ModernTransformerConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=2,
        max_context_length=128,
        hidden_dim=128,
    )
    policy = ModernTransformerLM(cfg)
    ref = ModernTransformerLM(cfg)
    ref.load_state_dict(policy.state_dict())
    prm = ProcessRewardModel()
    return policy, ref, prm


def test_evolution_config():
    cfg = SelfEvolutionConfig(
        num_cycles=3,
        candidates_per_prompt=2,
        dpo_beta=0.08,
        margin_threshold=0.05,
    )
    assert cfg.num_cycles == 3
    assert cfg.candidates_per_prompt == 2
    assert cfg.dpo_beta == 0.08


def test_evolution_single_cycle(small_models):
    policy, ref, prm = small_models
    cfg = SelfEvolutionConfig(
        num_cycles=1,
        candidates_per_prompt=2,
        max_new_tokens=12,
    )
    engine = SelfEvolutionEngine(
        policy_model=policy,
        reference_model=ref,
        prm=prm,
        config=cfg,
    )

    metric = engine.run_evolution_cycle("Explain linear regression", cycle_id=1)
    assert isinstance(metric, EvolutionStepMetric)
    assert metric.cycle_id == 1
    assert metric.stage == EvolutionStage.COMPLETED
    assert metric.candidates_generated == 2
    assert isinstance(metric.dpo_loss, float)
    assert metric.pre_evolution_reasoning_acc > 0.0
    assert metric.post_evolution_reasoning_acc > 0.0
    assert "Cycle 1 Complete" in metric.summary


def test_evolution_multi_cycle(small_models):
    policy, ref, prm = small_models
    cfg = SelfEvolutionConfig(
        num_cycles=2,
        candidates_per_prompt=2,
        max_new_tokens=10,
    )
    engine = SelfEvolutionEngine(
        policy_model=policy,
        reference_model=ref,
        prm=prm,
        config=cfg,
    )

    prompts = ["Prompt A", "Prompt B"]
    report = engine.run_autonomous_evolution(prompts, max_cycles=2)
    assert isinstance(report, SelfEvolutionReport)
    assert report.total_cycles == 2
    assert len(report.cycle_metrics) == 2
    assert report.total_tokens_synthesized > 0
    assert "Autonomous self-evolution succeeded" in report.curriculum_evolution_status
