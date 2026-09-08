"""
Libra Evaluation - Bradley-Terry Elo Rating Engine (Phase 28)
First-principles implementation of pairwise Elo rating estimation, win-rate tracking,
confidence intervals, and leaderboard management for LLM model comparison.
"""

from __future__ import annotations

import json
import math
import os
import time

from pydantic import BaseModel, Field


class ModelEloRecord(BaseModel):
    """Tracks historical Elo ratings and match statistics for a model."""

    model_id: str = Field(..., description="Unique model identifier")
    rating: float = Field(default=1200.0, description="Current Bradley-Terry Elo rating")
    matches: int = Field(default=0, description="Total number of evaluated matches")
    wins: int = Field(default=0, description="Total number of match wins")
    losses: int = Field(default=0, description="Total number of match losses")
    ties: int = Field(default=0, description="Total number of match ties")
    win_rate: float = Field(default=0.0, description="Win rate percentage (0.0 to 100.0)")
    ci_lower: float = Field(default=1200.0, description="95% confidence interval lower bound")
    ci_upper: float = Field(default=1200.0, description="95% confidence interval upper bound")
    category_ratings: dict[str, float] = Field(
        default_factory=dict,
        description="Category-specific Elo ratings (e.g. coding, reasoning, factual)",
    )


class MatchRecord(BaseModel):
    """Represents a completed head-to-head match between two models."""

    match_id: str = Field(..., description="Unique match identifier")
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp of match")
    model_a: str = Field(..., description="First competing model ID")
    model_b: str = Field(..., description="Second competing model ID")
    winner: str = Field(..., description="'model_a', 'model_b', or 'tie'")
    category: str = Field(default="overall", description="Task evaluation domain")
    prompt: str | None = Field(None, description="Benchmark prompt used for evaluation")
    rating_a_before: float = Field(..., description="Model A Elo before match")
    rating_b_before: float = Field(..., description="Model B Elo before match")
    rating_a_after: float = Field(..., description="Model A Elo after match")
    rating_b_after: float = Field(..., description="Model B Elo after match")
    delta_a: float = Field(..., description="Rating change for Model A")
    delta_b: float = Field(..., description="Rating change for Model B")


def calculate_expected_score(rating_a: float, rating_b: float) -> float:
    """Calculates expected score / win probability of Model A against Model B

    using the standard Bradley-Terry logistic formula with base 400:
    E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    """
    exponent = (rating_b - rating_a) / 400.0
    return 1.0 / (1.0 + 10.0**exponent)


def calculate_confidence_interval(
    rating: float, match_count: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Calculates standard error and confidence interval bounds for Elo rating.

    Standard error is estimated via Bradley-Terry asymptotic variance ~ 400 / sqrt(N).
    """
    if match_count <= 0:
        return (rating, rating)

    # Critical value: 1.96 for 95% confidence
    z = 1.96 if math.isclose(confidence, 0.95, rel_tol=0.01) else 1.645
    std_error = 400.0 / math.sqrt(max(1, match_count))
    margin = z * std_error

    return (round(rating - margin, 1), round(rating + margin, 1))


def compute_elo_update(
    rating_a: float,
    rating_b: float,
    outcome: float,
    k_factor: float = 32.0,
) -> tuple[float, float, float, float]:
    """Computes updated ratings and deltas for Model A and Model B.

    outcome: 1.0 (A wins), 0.5 (tie), 0.0 (B wins).
    Returns (new_rating_a, new_rating_b, delta_a, delta_b).
    """
    expected_a = calculate_expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a

    outcome_a = outcome
    outcome_b = 1.0 - outcome

    delta_a = k_factor * (outcome_a - expected_a)
    delta_b = k_factor * (outcome_b - expected_b)

    new_rating_a = rating_a + delta_a
    new_rating_b = rating_b + delta_b

    return (
        round(new_rating_a, 2),
        round(new_rating_b, 2),
        round(delta_a, 2),
        round(delta_b, 2),
    )


class EloLeaderboard:
    """Manages model records, match history, and Elo ranking computations."""

    def __init__(
        self,
        storage_path: str | None = "data/arena_leaderboard.json",
        default_rating: float = 1200.0,
    ) -> None:
        self.storage_path = storage_path
        self.default_rating = default_rating
        self.models: dict[str, ModelEloRecord] = {}
        self.matches: list[MatchRecord] = []
        self._load()

    def _ensure_model(self, model_id: str) -> ModelEloRecord:
        if model_id not in self.models:
            self.models[model_id] = ModelEloRecord(
                model_id=model_id,
                rating=self.default_rating,
                matches=0,
                wins=0,
                losses=0,
                ties=0,
                win_rate=0.0,
                ci_lower=self.default_rating,
                ci_upper=self.default_rating,
                category_ratings={},
            )
        return self.models[model_id]

    def record_match(
        self,
        model_a: str,
        model_b: str,
        winner: str,
        category: str = "overall",
        prompt: str | None = None,
        custom_k: float | None = None,
    ) -> MatchRecord:
        """Records a match between two models, updating ratings and persisting state.

        winner: 'model_a', 'model_b', or 'tie'.
        """
        rec_a = self._ensure_model(model_a)
        rec_b = self._ensure_model(model_b)

        # Map winner to numeric score for Model A
        if winner == "model_a":
            outcome_val = 1.0
            rec_a.wins += 1
            rec_b.losses += 1
        elif winner == "model_b":
            outcome_val = 0.0
            rec_a.losses += 1
            rec_b.wins += 1
        else:
            outcome_val = 0.5
            rec_a.ties += 1
            rec_b.ties += 1

        rec_a.matches += 1
        rec_b.matches += 1

        # Dynamic K-factor: higher for new models (< 10 matches) to converge quickly
        k = custom_k or (48.0 if min(rec_a.matches, rec_b.matches) < 10 else 32.0)

        # Overall rating update
        r_a_before = rec_a.rating
        r_b_before = rec_b.rating
        new_a, new_b, delta_a, delta_b = compute_elo_update(
            r_a_before, r_b_before, outcome_val, k_factor=k
        )

        rec_a.rating = new_a
        rec_b.rating = new_b

        # Update win rates
        rec_a.win_rate = round((rec_a.wins + 0.5 * rec_a.ties) / rec_a.matches * 100.0, 1)
        rec_b.win_rate = round((rec_b.wins + 0.5 * rec_b.ties) / rec_b.matches * 100.0, 1)

        # Update confidence intervals
        rec_a.ci_lower, rec_a.ci_upper = calculate_confidence_interval(rec_a.rating, rec_a.matches)
        rec_b.ci_lower, rec_b.ci_upper = calculate_confidence_interval(rec_b.rating, rec_b.matches)

        # Category-specific rating update if not overall
        if category and category.lower() != "overall":
            cat_key = category.lower()
            cat_a_before = rec_a.category_ratings.get(cat_key, self.default_rating)
            cat_b_before = rec_b.category_ratings.get(cat_key, self.default_rating)
            cat_new_a, cat_new_b, _, _ = compute_elo_update(
                cat_a_before, cat_b_before, outcome_val, k_factor=k
            )
            rec_a.category_ratings[cat_key] = cat_new_a
            rec_b.category_ratings[cat_key] = cat_new_b

        # Create match record
        match_id = f"match_{int(time.time() * 1000)}_{len(self.matches)}"
        match = MatchRecord(
            match_id=match_id,
            timestamp=time.time(),
            model_a=model_a,
            model_b=model_b,
            winner=winner,
            category=category,
            prompt=prompt,
            rating_a_before=r_a_before,
            rating_b_before=r_b_before,
            rating_a_after=new_a,
            rating_b_after=new_b,
            delta_a=delta_a,
            delta_b=delta_b,
        )
        self.matches.append(match)
        self._save()
        return match

    def get_leaderboard(self, category: str = "overall") -> list[ModelEloRecord]:
        """Returns sorted list of models ranked by rating in descending order."""
        cat_key = category.lower()

        if cat_key == "overall":
            ranked = sorted(self.models.values(), key=lambda m: m.rating, reverse=True)
        else:
            # Rank by category rating if present, fallback to overall rating
            ranked = sorted(
                self.models.values(),
                key=lambda m: m.category_ratings.get(cat_key, m.rating),
                reverse=True,
            )

        return ranked

    def get_model(self, model_id: str) -> ModelEloRecord | None:
        return self.models.get(model_id)

    def reset(self) -> None:
        """Clears all records and match history."""
        self.models.clear()
        self.matches.clear()
        self._save()

    def _save(self) -> None:
        if not self.storage_path:
            return
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            data = {
                "models": {k: v.model_dump() for k, v in self.models.items()},
                "matches": [
                    m.model_dump() for m in self.matches[-200:]
                ],  # Retain latest 200 matches
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    def _load(self) -> None:
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "models" in data:
                for k, v in data["models"].items():
                    self.models[k] = ModelEloRecord(**v)
            if "matches" in data:
                for m in data["matches"]:
                    self.matches.append(MatchRecord(**m))
        except (OSError, json.JSONDecodeError):
            pass


_global_leaderboard: EloLeaderboard | None = None


def get_leaderboard_store() -> EloLeaderboard:
    """Returns the singleton EloLeaderboard instance."""
    global _global_leaderboard
    if _global_leaderboard is None:
        _global_leaderboard = EloLeaderboard()
    return _global_leaderboard
