"""
Tests for Self-Rewarding Language Models & LLM-as-a-Judge (Phase 49)
"""

import pytest

from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.training.self_rewarding import (
    JudgeDimension,
    JudgeRubric,
    LLMJudge,
    SelfRewardingConfig,
    SelfRewardingTrainer,
)


@pytest.fixture
def small_model():
    cfg = ModernTransformerConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=2,
        max_context_length=128,
        hidden_dim=128,
    )
    return ModernTransformerLM(cfg)


def test_judge_prompt_and_rubric():
    rubric = JudgeRubric(
        name="Custom Rubric",
        dimensions=[
            JudgeDimension(name="Precision", description="Factual correctness", weight=2.0),
            JudgeDimension(name="Conciseness", description="No unnecessary tokens", weight=1.0),
        ],
        min_score=1,
        max_score=5,
    )
    judge = LLMJudge(rubric=rubric)
    prompt = judge.build_evaluation_prompt("What is 2+2?", "4")
    assert "### Instruction:" in prompt
    assert "### Response:" in prompt
    assert "Precision (weight 2.0)" in prompt
    assert "[SCORE: <rating>]" in prompt


def test_judge_score_parsing():
    judge = LLMJudge()
    score, critique, dims = judge.parse_judge_output("The solution is accurate.\n[SCORE: 4.5]")
    assert score == 4.5
    assert "The solution is accurate" in critique
    assert "Correctness" in dims

    # Test alternate formatting
    score2, _, _ = judge.parse_judge_output("Overall rating: Score: 3/5 points")
    assert score2 == 3.0

    # Test clamping
    score3, _, _ = judge.parse_judge_output("[SCORE: 10]")
    assert score3 == 5.0


def test_judge_position_debiasing():
    judge = LLMJudge()
    s_a, s_b, critique = judge.evaluate_pairwise(
        "Write a poem",
        "Roses are red, violets are blue.",
        "A brief line.",
        debias_position=True,
    )
    assert 0.0 <= s_a <= 1.0
    assert 0.0 <= s_b <= 1.0
    assert "Position-debiased evaluation" in critique


def test_self_rewarding_trainer_step(small_model):
    ref_model = ModernTransformerLM(small_model.config)
    ref_model.load_state_dict(small_model.state_dict())

    cfg = SelfRewardingConfig(
        beta=0.1,
        lr=1e-4,
        num_candidates=2,
        max_new_tokens=12,
        margin_threshold=0.05,
    )
    trainer = SelfRewardingTrainer(
        policy_model=small_model,
        reference_model=ref_model,
        config=cfg,
    )

    result = trainer.train_iteration_step("Explain gradient descent", iteration_id=1)
    assert result.iteration_id == 1
    assert len(result.candidates) == 2
    assert result.chosen_id in [0, 1]
    assert result.rejected_id in [0, 1]
    assert isinstance(result.dpo_loss, float)
    assert isinstance(result.score_margin, float)
