"""Libra Evaluation & Benchmarking Package."""

from packages.evaluation.harness import CategoryScore, EvaluationHarness, EvaluationReport
from packages.evaluation.loss_eval import (
    LossMetrics,
    compute_bits_per_token,
    compute_perplexity,
    evaluate_dataset_loss,
    evaluate_tokens_loss,
)
from packages.evaluation.multiple_choice import (
    ChoiceScore,
    MultipleChoiceEvaluator,
    MultipleChoiceResult,
)
from packages.evaluation.probes import ProbeExample, filter_probes, get_standard_probes

__all__ = [
    "compute_perplexity",
    "compute_bits_per_token",
    "LossMetrics",
    "evaluate_tokens_loss",
    "evaluate_dataset_loss",
    "ChoiceScore",
    "MultipleChoiceResult",
    "MultipleChoiceEvaluator",
    "ProbeExample",
    "get_standard_probes",
    "filter_probes",
    "CategoryScore",
    "EvaluationReport",
    "EvaluationHarness",
]
