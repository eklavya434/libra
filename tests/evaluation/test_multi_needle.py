"""
Unit tests for MultiNeedleEvaluator and multi-key associative recall.
"""

from packages.evaluation.multi_needle import MultiNeedleEvaluator, MultiNeedleItem


def test_multi_needle_haystack_construction():
    evaluator = MultiNeedleEvaluator()
    needles = [
        MultiNeedleItem(key="Key-A", fact="The first passcode is Key-A.", depth_fraction=0.25),
        MultiNeedleItem(key="Key-B", fact="The second passcode is Key-B.", depth_fraction=0.75),
    ]

    haystack = evaluator.construct_haystack(target_words=200, needles=needles)
    assert "Key-A" in haystack
    assert "Key-B" in haystack

    # Check relative ordering
    idx_a = haystack.index("Key-A")
    idx_b = haystack.index("Key-B")
    assert idx_a < idx_b


def test_multi_needle_evaluation_scoring():
    evaluator = MultiNeedleEvaluator()
    expected = ["Code-101", "Callsign-Eagle", "Key-Omega"]

    # All found
    res1 = "The keys are Code-101, Callsign-Eagle, and Key-Omega."
    all_corr, score, found, missing = evaluator.evaluate_response(res1, expected)
    assert all_corr is True
    assert score == 1.0
    assert len(found) == 3
    assert len(missing) == 0

    # Partial found (2 of 3)
    res2 = "Found Code-101 and Key-Omega."
    all_corr2, score2, found2, missing2 = evaluator.evaluate_response(res2, expected)
    assert all_corr2 is False
    assert score2 == round(2 / 3, 3)
    assert "Callsign-Eagle" in missing2

    # None found
    res3 = "I do not have access to any credentials."
    all_corr3, score3, found3, missing3 = evaluator.evaluate_response(res3, expected)
    assert all_corr3 is False
    assert score3 == 0.0
    assert len(found3) == 0


def test_multi_needle_run_single():
    evaluator = MultiNeedleEvaluator()

    def mock_gen(prompt: str) -> str:
        # Echoes keys if they exist in prompt
        keys = []
        if "Alpha-77" in prompt:
            keys.append("Alpha-77")
        if "Falcon" in prompt:
            keys.append("Falcon")
        if "Omega-Zero" in prompt:
            keys.append("Omega-Zero")
        return f"Retrieved keys: {', '.join(keys)}"

    result = evaluator.run_single(generator_fn=mock_gen, context_words=300)
    assert result.all_correct is True
    assert result.partial_score == 1.0
    assert result.num_needles == 3
    assert result.latency_ms >= 0
