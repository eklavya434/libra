"""Libra Evaluation & Benchmarking Package."""

from packages.evaluation.arena import ModelComparisonArena
from packages.evaluation.automated_judge import AutomatedJudge, JudgeVerdict
from packages.evaluation.distillation_eval import DistillationEvaluator
from packages.evaluation.elo import (
    EloLeaderboard,
    MatchRecord,
    ModelEloRecord,
    calculate_confidence_interval,
    calculate_expected_score,
    compute_elo_update,
    get_leaderboard_store,
)
from packages.evaluation.harness import CategoryScore, EvaluationHarness, EvaluationReport
from packages.evaluation.loss_eval import (
    LossMetrics,
    compute_bits_per_token,
    compute_perplexity,
    evaluate_dataset_loss,
    evaluate_tokens_loss,
)
from packages.evaluation.metrics import ArenaModelMetric
from packages.evaluation.multiple_choice import (
    ChoiceScore,
    MultipleChoiceEvaluator,
    MultipleChoiceResult,
)
from packages.evaluation.needle_haystack import NeedleInHaystackEvaluator, NeedleResult
from packages.evaluation.probes import ProbeExample, filter_probes, get_standard_probes
from packages.evaluation.tournament import (
    DEFAULT_BENCHMARK_PROMPTS,
    BenchmarkPrompt,
    TournamentReport,
    TournamentRunner,
)

__all__ = [
    "DEFAULT_BENCHMARK_PROMPTS",
    "ArenaModelMetric",
    "AutomatedJudge",
    "BenchmarkPrompt",
    "CategoryScore",
    "ChoiceScore",
    "DistillationEvaluator",
    "EloLeaderboard",
    "EvaluationHarness",
    "EvaluationReport",
    "JudgeVerdict",
    "LossMetrics",
    "MatchRecord",
    "ModelComparisonArena",
    "ModelEloRecord",
    "MultipleChoiceEvaluator",
    "MultipleChoiceResult",
    "NeedleInHaystackEvaluator",
    "NeedleResult",
    "ProbeExample",
    "TournamentReport",
    "TournamentRunner",
    "calculate_confidence_interval",
    "calculate_expected_score",
    "compute_bits_per_token",
    "compute_elo_update",
    "compute_perplexity",
    "evaluate_dataset_loss",
    "evaluate_tokens_loss",
    "filter_probes",
    "get_leaderboard_store",
    "get_standard_probes",
]
