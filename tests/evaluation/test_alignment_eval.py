"""Unit tests for AlignmentEvaluator."""

from packages.evaluation.alignment_eval import AlignmentEvaluator
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.training.kto_dataset import KTOSample


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


def test_alignment_eval_kto():
    policy, ref = _create_toy_models()
    samples = [
        KTOSample(prompt="Query A", completion=" Good response", is_desirable=True),
        KTOSample(prompt="Query B", completion=" Bad response", is_desirable=False),
    ]

    metrics = AlignmentEvaluator.evaluate_kto(policy, ref, samples, beta=0.1)

    assert metrics["desirable_count"] == 1
    assert metrics["undesirable_count"] == 1
    assert "reward_margin" in metrics
    assert "binary_ranking_accuracy" in metrics


def test_alignment_eval_compare_paradigms():
    policy, ref = _create_toy_models()
    prompts = ["Hello", "World"]

    def _scorer(prompt: str, comp: str) -> float:
        return float(len(comp))

    policies = {"Test Policy": policy}
    result = AlignmentEvaluator.compare_paradigms(policies, ref, prompts, _scorer, max_tokens=8)

    assert len(result.paradigms) == 2
    assert result.paradigms[0].paradigm == "Base / SFT Reference"
    assert result.paradigms[1].paradigm == "Test Policy"
    assert result.winner in ["Base / SFT Reference", "Test Policy"]
