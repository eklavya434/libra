"""Unit tests for OnlineDPOTrainer."""

from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.training.online_dpo_trainer import OnlineDPOConfig, OnlineDPOTrainer


def _create_toy_models():
    cfg = ModernTransformerConfig(
        vocab_size=256,
        d_model=32,
        n_layers=2,
        n_heads=2,
        max_context_length=64,
        hidden_dim=64,
    )
    policy = ModernTransformerLM(cfg)
    ref = ModernTransformerLM(cfg)
    ref.load_state_dict(policy.state_dict())
    return policy, ref


def _mock_scorer(prompt: str, comp: str) -> float:
    return float(len(comp))


def test_online_dpo_generation_and_step():
    policy, ref = _create_toy_models()
    config = OnlineDPOConfig(beta=0.1, lr=1e-3, max_new_tokens=8, num_candidates=2)
    trainer = OnlineDPOTrainer(policy, ref, config=config, reward_fn=_mock_scorer)

    cand = trainer.generate_candidate("Test prompt", temperature=0.8)
    assert isinstance(cand, str)

    telemetry_list = trainer.step(["What is AI?"], scorer_fn=_mock_scorer)
    assert len(telemetry_list) == 1
    t = telemetry_list[0]

    assert t.prompt == "What is AI?"
    assert isinstance(t.chosen_text, str)
    assert isinstance(t.rejected_text, str)
    assert t.chosen_oracle_score >= t.rejected_oracle_score
    assert isinstance(t.loss, float)
