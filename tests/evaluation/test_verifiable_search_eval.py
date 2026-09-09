"""Unit tests for VerifiableSearchEvaluator."""

from packages.evaluation.verifiable_search_eval import VerifiableSearchEvaluator
from packages.models.reasoning.verifiable_search import (
    VerifiableSearchConfig,
    VerifiableSearchEngine,
)


def test_verifiable_search_evaluator_benchmark():
    cfg = VerifiableSearchConfig(beam_width=2, max_depth=4, step_prune_threshold=0.55)
    engine = VerifiableSearchEngine(config=cfg)

    prompts_with_expected = [
        ("Calculate (15 * 4) + (24 / 3) - 17", "51"),
        ("Sarah has 12 apples, buys 8 more, eats 3, divides rest among 2 friends", "8.5"),
    ]

    metrics = VerifiableSearchEvaluator.evaluate_search(engine, prompts_with_expected)

    assert metrics["total_problems"] == 2
    assert metrics["solved_correctly"] >= 1
    assert metrics["accuracy"] >= 0.5
    assert metrics["total_pruned_branches"] >= 1


def test_verifiable_search_compare_methods():
    cfg = VerifiableSearchConfig(beam_width=2, max_depth=4, step_prune_threshold=0.55)
    engine = VerifiableSearchEngine(config=cfg)

    prompts_with_expected = [
        ("Calculate (15 * 4) + (24 / 3) - 17", "51"),
    ]

    result = VerifiableSearchEvaluator.compare_search_methods(
        engine, prompts_with_expected, n_best_of_n=3
    )

    assert len(result.methods) == 3
    assert result.methods[0].method == "Greedy Autoregressive"
    assert "Best-of-3" in result.methods[1].method
    assert "Verifiable Search" in result.methods[2].method
    assert result.winner == "Step-Level Verifiable Search (PRM)"
