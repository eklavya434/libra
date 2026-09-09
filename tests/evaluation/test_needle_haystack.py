"""
Unit Tests for Needle-In-A-Haystack (NIAH) Retrieval Benchmark
"""

from packages.evaluation.needle_haystack import NeedleInHaystackEvaluator, NeedleResult


def test_haystack_construction_depth():
    evaluator = NeedleInHaystackEvaluator(
        needle="The secret pin is 9988.",
        target_key="9988",
    )

    # Needle inserted at top (depth 0.0)
    haystack_top = evaluator.construct_haystack(target_words=100, depth_fraction=0.0)
    assert haystack_top.startswith("The secret pin is 9988.")

    # Needle inserted at bottom (depth 1.0)
    haystack_bottom = evaluator.construct_haystack(target_words=100, depth_fraction=1.0)
    assert haystack_bottom.endswith("The secret pin is 9988.")

    # Needle inserted at middle (depth 0.5)
    haystack_mid = evaluator.construct_haystack(target_words=100, depth_fraction=0.5)
    words = haystack_mid.split()
    assert "9988." in words
    pos = words.index("9988.")
    assert 40 <= pos <= 60


def test_needle_evaluation_scoring():
    evaluator = NeedleInHaystackEvaluator(target_key="9988")

    correct, score = evaluator.evaluate_response("The passcode retrieved is 9988.")
    assert correct is True
    assert score == 1.0

    wrong, score_w = evaluator.evaluate_response("I could not find the code.")
    assert wrong is False
    assert score_w == 0.0


def test_needle_run_single_mock():
    evaluator = NeedleInHaystackEvaluator(
        needle="The secret passcode is 9988.",
        target_key="9988",
    )

    def mock_generator(prompt: str) -> str:
        assert "9988" in prompt
        return "The code is 9988."

    res = evaluator.run_single(mock_generator, context_words=50, depth_fraction=0.5)
    assert isinstance(res, NeedleResult)
    assert res.is_correct is True
    assert res.score == 1.0
    assert res.depth_percent == 50.0
    assert res.context_length == 50
    assert res.latency_ms >= 0.0


def test_needle_run_grid():
    evaluator = NeedleInHaystackEvaluator(target_key="alpha")

    def mock_generator(prompt: str) -> str:
        return "alpha"

    results = evaluator.run_grid(
        generator_fn=mock_generator,
        context_lengths=[20, 40],
        depth_fractions=[0.0, 1.0],
    )
    assert len(results) == 4
    assert all(r.is_correct for r in results)
