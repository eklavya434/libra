"""
Tests for Medusa Speculative Generation and Evaluation (Phase 46)
"""

import torch

from packages.evaluation.medusa_eval import MedusaEvaluator
from packages.models.components.medusa import MedusaModel
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM


def create_toy_medusa() -> MedusaModel:
    torch.manual_seed(42)
    cfg = ModernTransformerConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        max_context_length=64,
        hidden_dim=64,
    )
    base = ModernTransformerLM(cfg)
    return MedusaModel(base_model=base, num_heads=2)


def test_medusa_generate_loop():
    medusa = create_toy_medusa()
    prompt = torch.randint(0, 64, (1, 6))

    res = medusa.medusa_generate(prompt, max_new_tokens=10, temperature=0.0)

    assert res["prompt_length"] == 6
    assert res["generated_tokens_count"] >= 10
    assert res["avg_accepted_per_step"] >= 1.0
    assert len(res["steps"]) > 0

    first_step = res["steps"][0]
    assert "drafted_tokens" in first_step
    assert "accepted_tokens" in first_step
    assert first_step["acceptance_mask"][0] is True  # Base token is always accepted!


def test_medusa_evaluator_benchmark():
    medusa = create_toy_medusa()
    prompts = [torch.randint(0, 64, (1, 5)) for _ in range(2)]

    bench = MedusaEvaluator.benchmark_speculative_speedup(
        medusa_model=medusa,
        test_prompts=prompts,
        max_new_tokens=8,
    )

    assert bench["test_prompts_count"] == 2
    assert bench["speedup_ratio"] > 0.0
    assert bench["avg_accepted_tokens_per_step"] >= 1.0
    assert "autoregressive_tokens_per_sec" in bench
    assert "medusa_tokens_per_sec" in bench


def test_medusa_evaluator_head_accuracies():
    medusa = create_toy_medusa()
    seqs = [torch.randint(0, 64, (1, 10)) for _ in range(3)]

    acc = MedusaEvaluator.analyze_head_accuracies(medusa_model=medusa, sequences=seqs)

    assert acc["num_heads"] == 2
    assert len(acc["per_head_accuracy_pct"]) == 2
    for p in acc["per_head_accuracy_pct"]:
        assert 0.0 <= p <= 100.0
