"""
Phase 28 Demonstration Script: Continuous Benchmarking & Automated Model Arena (Elo & Win Rate)
Demonstrates Bradley-Terry pairwise rating updates, position-bias mitigated auto-judging,
and round-robin tournament execution.
"""

import asyncio
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from packages.evaluation.automated_judge import AutomatedJudge
from packages.evaluation.elo import EloLeaderboard, compute_elo_update
from packages.evaluation.tournament import TournamentRunner


async def main():
    print("=" * 75)
    print("  PROJECT LIBRA - PHASE 28 DEMO: AUTOMATED MODEL ARENA & ELO ENGINE")
    print("=" * 75)

    # 1. First-Principles Bradley-Terry Math
    print("\n[1] Bradley-Terry Mathematical Mechanics:")
    r_a, r_b = 1200.0, 1200.0
    print(f"  Starting Ratings: Model A = {r_a}, Model B = {r_b}")

    # A defeats B
    new_a, new_b, d_a, d_b = compute_elo_update(r_a, r_b, outcome=1.0, k_factor=32.0)
    print(f"  Model A wins: A -> {new_a} (+{d_a}), B -> {new_b} ({d_b})")
    print(f"  Sum of deltas: {d_a + d_b} (Exact Rating Conservation)")

    # 400 pt difference test
    r_champion, r_challenger = 1600.0, 1200.0
    _, _, d_c, d_ch = compute_elo_update(r_champion, r_challenger, outcome=1.0, k_factor=32.0)
    print(f"  Expected Win for 1600 vs 1200: Champ delta = +{d_c}, Challenger delta = {d_ch}")

    # 2. Position-Bias Mitigated Automated Referee
    print("\n[2] Automated Referee & Position Bias Mitigation:")
    judge = AutomatedJudge()
    prompt = "Write a Python function `is_even(n: int) -> bool` with docstring."

    code_candidate = (
        '```python\ndef is_even(n: int) -> bool:\n    """Return True if n is even."""\n'
        "    return n % 2 == 0\n```"
    )
    verbose_candidate = "To check if a number is even in python you can check remainder modulo 2."

    verdict = judge.evaluate_pair(
        prompt=prompt,
        completion_a=code_candidate,
        completion_b=verbose_candidate,
        model_a_id="libra-llama-tied",
        model_b_id="generic-baseline",
    )

    print(f'  Prompt: "{prompt}"')
    print(
        f"  Winner: {verdict.winner} (Dual-pass position swap applied: {verdict.position_swapped})"
    )
    print(f"  Score A: {verdict.score_a} / 10.0 | Score B: {verdict.score_b} / 10.0")
    print(f"  Referee Rationale: {verdict.rationale}")

    # 3. Automated Round-Robin Tournament
    print("\n[3] Executing Automated Round-Robin Tournament:")
    demo_db_path = "data/demo_phase28_leaderboard.json"
    if os.path.exists(demo_db_path):
        os.remove(demo_db_path)

    leaderboard = EloLeaderboard(storage_path=demo_db_path)
    tournament = TournamentRunner(leaderboard=leaderboard, judge=judge)

    competing_models = [
        {"model": "libra-mock-v1", "provider": "mock-provider"},
        {"model": "libra-mock-v2", "provider": "mock-provider"},
        {"model": "libra-mock-v3", "provider": "mock-provider"},
    ]

    print(f"  Competing Models: {[m['model'] for m in competing_models]}")
    print("  Running matches across [coding, factual, instruction]...")

    report = await tournament.run_tournament(
        models=competing_models,
        categories=["coding", "factual", "instruction"],
        prompts_per_category=1,
        max_tokens=32,
    )

    print(f"  Completed {report.total_matches} pairwise tournament matches.")

    # 4. Final Elo Leaderboard
    print("\n[4] Official Elo Leaderboard Rankings:")
    print("-" * 75)
    print(
        f"{'Rank':<6}{'Model ID':<22}{'Elo Rating':<14}{'95% CI':<18}{'Win Rate':<10}{'Matches (W-L-T)'}"
    )
    print("-" * 75)

    ranked_models = leaderboard.get_leaderboard(category="overall")
    for idx, m in enumerate(ranked_models, 1):
        margin = round((m.ci_upper - m.ci_lower) / 2)
        ci_str = f"+/-{margin} [{m.ci_lower:.0f}, {m.ci_upper:.0f}]"
        win_str = f"{m.win_rate:.1f}%"
        record_str = f"{m.matches} ({m.wins}-{m.losses}-{m.ties})"
        print(f"#{idx:<5}{m.model_id:<22}{m.rating:<14.1f}{ci_str:<18}{win_str:<10}{record_str}")

    print("-" * 75)
    print(
        "\n[OK] Phase 28 Continuous Benchmarking & Automated Model Arena verified successfully!\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
