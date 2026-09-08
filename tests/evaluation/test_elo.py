"""
Unit tests for Bradley-Terry Elo, Automated Judge, and Tournament Engine (Phase 28).
"""

import math
import tempfile

import pytest

from packages.evaluation.automated_judge import AutomatedJudge
from packages.evaluation.elo import (
    EloLeaderboard,
    calculate_confidence_interval,
    calculate_expected_score,
    compute_elo_update,
)
from packages.evaluation.tournament import (
    DEFAULT_BENCHMARK_PROMPTS,
    TournamentRunner,
)


class TestBradleyTerryElo:
    def test_expected_score_symmetry(self):
        # When ratings are equal, expected score must be exactly 0.5
        assert math.isclose(calculate_expected_score(1200.0, 1200.0), 0.5, abs_tol=1e-5)

        # Symmetry: E_A + E_B == 1.0
        e_a = calculate_expected_score(1500.0, 1300.0)
        e_b = calculate_expected_score(1300.0, 1500.0)
        assert math.isclose(e_a + e_b, 1.0, abs_tol=1e-5)
        assert e_a > 0.5
        assert e_b < 0.5

        # 400 point difference -> 10:1 ratio (approx 0.909 vs 0.091)
        e_400 = calculate_expected_score(1600.0, 1200.0)
        assert math.isclose(e_400, 1.0 / (1.0 + 10.0 ** (-1)), abs_tol=1e-3)

    def test_rating_conservation(self):
        # In any match outcome, delta_a + delta_b must equal 0
        for outcome in [1.0, 0.5, 0.0]:
            new_a, new_b, delta_a, delta_b = compute_elo_update(
                1200.0, 1200.0, outcome, k_factor=32.0
            )
            assert math.isclose(delta_a + delta_b, 0.0, abs_tol=1e-2)
            assert math.isclose((new_a - 1200.0) + (new_b - 1200.0), 0.0, abs_tol=1e-2)

    def test_confidence_interval_tightens(self):
        ci_1 = calculate_confidence_interval(1200.0, 1)
        ci_10 = calculate_confidence_interval(1200.0, 10)
        ci_100 = calculate_confidence_interval(1200.0, 100)

        width_1 = ci_1[1] - ci_1[0]
        width_10 = ci_10[1] - ci_10[0]
        width_100 = ci_100[1] - ci_100[0]

        # Margin of error shrinks inversely with sqrt(N)
        assert width_1 > width_10 > width_100
        assert ci_100[0] < 1200.0 < ci_100[1]

    def test_leaderboard_lifecycle(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        lb = EloLeaderboard(storage_path=tmp_path, default_rating=1200.0)

        # Model A defeats Model B in coding
        m1 = lb.record_match("model_alpha", "model_beta", "model_a", category="coding")
        assert m1.winner == "model_a"
        assert m1.rating_a_after > 1200.0
        assert m1.rating_b_after < 1200.0

        # Check records
        alpha = lb.get_model("model_alpha")
        beta = lb.get_model("model_beta")
        assert alpha is not None and beta is not None
        assert alpha.wins == 1 and alpha.losses == 0
        assert beta.wins == 0 and beta.losses == 1
        assert alpha.win_rate == 100.0
        assert beta.win_rate == 0.0
        assert "coding" in alpha.category_ratings

        # Check rankings
        ranked = lb.get_leaderboard()
        assert ranked[0].model_id == "model_alpha"
        assert ranked[1].model_id == "model_beta"

        # Check persistence reload
        lb2 = EloLeaderboard(storage_path=tmp_path, default_rating=1200.0)
        assert len(lb2.models) == 2
        assert lb2.get_model("model_alpha").rating == alpha.rating


class TestAutomatedJudge:
    def test_dual_pass_mitigates_position_bias(self):
        judge = AutomatedJudge()
        prompt = "Write a Python function to check prime numbers."

        comp_good = (
            '```python\ndef is_prime(n: int) -> bool:\n    """Check if n is prime."""\n'
            "    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n"
            "        if n % i == 0:\n            return False\n    return True\n```"
        )
        comp_poor = "I think prime numbers are numbers that cannot be divided by anything."

        verdict = judge.evaluate_pair(
            prompt, comp_good, comp_poor, model_a_id="good", model_b_id="poor"
        )
        assert verdict.position_swapped is True
        assert verdict.winner == "model_a"
        assert verdict.score_a > verdict.score_b
        assert "good" in verdict.criteria_breakdown

    def test_tie_on_identical_quality(self):
        judge = AutomatedJudge()
        prompt = "What is the capital of France?"
        comp_1 = "The capital of France is Paris."
        comp_2 = "Paris is the capital city of France."

        verdict = judge.evaluate_pair(prompt, comp_1, comp_2, model_a_id="m1", model_b_id="m2")
        assert verdict.winner == "tie"
        assert abs(verdict.score_a - verdict.score_b) < 0.35


class TestTournamentRunner:
    def test_benchmark_prompts_coverage(self):
        categories = {p.category for p in DEFAULT_BENCHMARK_PROMPTS}
        assert "coding" in categories
        assert "reasoning" in categories
        assert "factual" in categories
        assert "instruction" in categories
        assert len(DEFAULT_BENCHMARK_PROMPTS) >= 12

    @pytest.mark.asyncio
    async def test_tournament_round_robin_execution(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        lb = EloLeaderboard(storage_path=tmp_path)
        judge = AutomatedJudge()
        tournament = TournamentRunner(leaderboard=lb, judge=judge)

        models = [
            {"model": "libra-mock-v1", "provider": "mock-provider"},
            {"model": "libra-mock-v2", "provider": "mock-provider"},
        ]

        report = await tournament.run_tournament(
            models=models,
            categories=["factual"],
            prompts_per_category=1,
        )

        assert report.total_matches == 1
        assert len(report.matches) == 1
        assert len(report.leaderboard_rankings) >= 2
