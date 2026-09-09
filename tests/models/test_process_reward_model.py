"""
Unit tests for ProcessRewardModel (PRM) and step-level reasoning verification.
"""

from packages.models.reasoning.prm import ProcessRewardModel


def test_prm_arithmetic_verification():
    prm = ProcessRewardModel()

    # Valid equations
    valid, errors = prm.verify_arithmetic("Step 1: Notice that 4 * 7 = 28 and 28 - 4 = 24.")
    assert valid is True
    assert len(errors) == 0

    # Faulty equation
    valid_bad, errors_bad = prm.verify_arithmetic("Step 2: 7 * 7 = 50, which is near 50.")
    assert valid_bad is False
    assert any("Arithmetic mismatch" in e for e in errors_bad)


def test_prm_step_scoring():
    prm = ProcessRewardModel()

    # Sound reasoning step
    score_valid = prm.score_step("Step 1: Simplify by computing 7 / 7 = 1.")
    assert score_valid.is_valid is True
    assert score_valid.score >= 0.70

    # Contradictory step
    score_contradict = prm.score_step("This means 0 = 1 which is impossible.")
    assert score_contradict.is_valid is False
    assert score_contradict.score < 0.55

    # Repetitive circular step
    history = ["Step 1: Compute 7 / 7 = 1."]
    score_repeat = prm.score_step("Step 1: Compute 7 / 7 = 1.", context_history=history)
    assert score_repeat.is_valid is False
    assert any("Circular reasoning" in err for err in score_repeat.detected_errors)


def test_prm_trace_evaluation():
    prm = ProcessRewardModel()

    valid_steps = [
        "Step 1: Compute 7 / 7 = 1.",
        "Step 2: Compute 7 - 1 = 6.",
        "Step 3: Finally 4 * 6 = 24.",
    ]
    res_valid = prm.score_trace(valid_steps)
    assert res_valid.is_valid_trace is True
    assert res_valid.first_error_index is None
    assert res_valid.mean_step_score >= 0.70

    flawed_steps = [
        "Step 1: Compute 7 / 7 = 1.",
        "Step 2: Compute 7 - 1 = 9.",  # Arithmetic error: 7 - 1 = 6
        "Step 3: Multiply by 4.",
    ]
    res_flawed = prm.score_trace(flawed_steps)
    assert res_flawed.is_valid_trace is False
    assert res_flawed.first_error_index == 1
