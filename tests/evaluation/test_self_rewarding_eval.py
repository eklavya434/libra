"""
Tests for Self-Rewarding Evaluator (Phase 49)
"""

import pytest

from packages.evaluation.self_rewarding_eval import (
    PositionBiasAnalysis,
    SelfRewardingEvaluator,
)
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.training.self_rewarding import LLMJudge


@pytest.fixture
def test_models():
    cfg = ModernTransformerConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=2,
        max_context_length=128,
        hidden_dim=128,
    )
    m0 = ModernTransformerLM(cfg)
    m1 = ModernTransformerLM(cfg)
    m1.load_state_dict(m0.state_dict())
    return m0, m1


def test_position_bias_analysis():
    judge = LLMJudge()
    evaluator = SelfRewardingEvaluator(judge=judge)
    bias = evaluator.analyze_position_bias(
        "Summarize photosynthesis",
        "Plants convert sunlight into chemical energy.",
        "Short text.",
    )
    assert isinstance(bias, PositionBiasAnalysis)
    assert bias.forward_preference in ["A", "B", "TIE"]
    assert bias.reverse_preference in ["A", "B", "TIE"]
    assert 0.0 <= bias.score_a_forward <= 1.0


def test_benchmark_iterations(test_models):
    m0, m1 = test_models
    evaluator = SelfRewardingEvaluator()
    res = evaluator.benchmark_iterations(
        [(0, m0), (1, m1)],
        test_prompts=["Test prompt 1", "Test prompt 2"],
    )
    assert res.prompts_evaluated == 2
    assert len(res.iterations) == 2
    assert res.iterations[0].win_rate >= 0.0
    assert -1.0 <= res.iterations[0].oracle_correlation <= 1.0
    assert "Evaluated 2 iterations" in res.summary
