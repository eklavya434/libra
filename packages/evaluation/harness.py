"""Unified Evaluation Harness for Project Libra.

Executes comprehensive model benchmarks:
1. Perplexity & Loss evaluation
2. Domain task probes (Reasoning, Math, Coding, Instruction, Safety)
3. Empirical comparison against random chance baselines
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn

from packages.evaluation.loss_eval import LossMetrics, evaluate_tokens_loss
from packages.evaluation.multiple_choice import MultipleChoiceEvaluator, MultipleChoiceResult
from packages.evaluation.probes import ProbeExample, filter_probes, get_standard_probes


@dataclass
class CategoryScore:
    category: str
    total: int
    correct: int
    accuracy: float
    chance_accuracy: float


@dataclass
class EvaluationReport:
    """Consolidated evaluation report for a benchmarked model."""

    model_name: str
    device: str
    loss_metrics: Optional[LossMetrics] = None
    category_scores: dict[str, CategoryScore] = field(default_factory=dict)
    overall_total_probes: int = 0
    overall_correct_probes: int = 0
    overall_accuracy: float = 0.0
    random_chance_accuracy: float = 0.25
    sample_results: list[MultipleChoiceResult] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Format the evaluation report into a clean, GitHub-flavored markdown table."""
        lines = [
            f"# Benchmark Evaluation Report: `{self.model_name}`",
            f"**Device**: `{self.device}`",
            "",
        ]

        if self.loss_metrics is not None:
            lines.extend(
                [
                    "## 1. Language Modeling & Compression Metrics",
                    "| Metric | Value | Interpretation |",
                    "| :--- | :--- | :--- |",
                    f"| **Tokens Evaluated** | {self.loss_metrics.total_tokens:,} | Total token count |",
                    f"| **Mean Loss (NLL)** | {self.loss_metrics.mean_loss:.4f} | Cross-entropy loss |",
                    f"| **Perplexity (PPL)** | {self.loss_metrics.perplexity:.2f} | Branching factor |",
                    f"| **Bits per Token** | {self.loss_metrics.bits_per_token:.4f} | Compression efficiency |",
                    "",
                ]
            )

        lines.extend(
            [
                "## 2. Domain Task Probe Accuracies",
                "| Domain / Category | Evaluated | Correct | Accuracy | Chance Baseline | Status |",
                "| :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )

        for cat, sc in self.category_scores.items():
            diff = sc.accuracy - sc.chance_accuracy
            status = (
                "🏆 Above Chance"
                if diff > 0.05
                else ("⚠️ At Chance" if abs(diff) <= 0.05 else "❌ Below Chance")
            )
            lines.append(
                f"| **{cat.capitalize()}** | {sc.total} | {sc.correct} | "
                f"**{sc.accuracy * 100:.1f}%** | {sc.chance_accuracy * 100:.1f}% | {status} |"
            )

        lines.extend(
            [
                "| :--- | :--- | :--- | :--- | :--- | :--- |",
                f"| **OVERALL TOTAL** | {self.overall_total_probes} | {self.overall_correct_probes} | "
                f"**{self.overall_accuracy * 100:.1f}%** | {self.random_chance_accuracy * 100:.1f}% | "
                f"{'🏆 Better than Random' if self.overall_accuracy > self.random_chance_accuracy else '⚠️ Baseline Level'} |",
                "",
            ]
        )

        if self.sample_results:
            lines.extend(
                [
                    "## 3. Qualitative Sample Probes",
                    "| Prompt | Predicted | Expected | Correct? |",
                    "| :--- | :--- | :--- | :--- |",
                ]
            )
            for r in self.sample_results[:6]:
                pred = r.choices[r.predicted_index].strip().replace("\n", " ")
                exp = r.choices[r.correct_index].strip().replace("\n", " ")
                mark = "✅" if r.is_correct else "❌"
                prompt_clean = r.prompt.strip().replace("\n", " ")
                lines.append(f"| `{prompt_clean}` | `{pred}` | `{exp}` | {mark} |")
            lines.append("")

        return "\n".join(lines)

    def save_json(self, output_path: str | Path) -> None:
        """Serialize report to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_name": self.model_name,
            "device": self.device,
            "loss_metrics": self.loss_metrics.to_dict() if self.loss_metrics else None,
            "overall_accuracy": round(self.overall_accuracy, 4),
            "overall_total_probes": self.overall_total_probes,
            "overall_correct_probes": self.overall_correct_probes,
            "category_scores": {
                k: {
                    "category": v.category,
                    "total": v.total,
                    "correct": v.correct,
                    "accuracy": round(v.accuracy, 4),
                    "chance_accuracy": round(v.chance_accuracy, 4),
                }
                for k, v in self.category_scores.items()
            },
            "sample_results": [r.to_dict() for r in self.sample_results[:10]],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


class EvaluationHarness:
    """Unified test runner orchestrating loss and probe benchmarking."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        model_name: str = "Libra-Model",
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device or next(model.parameters()).device
        self.mc_evaluator = MultipleChoiceEvaluator(model, tokenizer, device=self.device)

    def run_benchmark(
        self,
        probes: Optional[Sequence[ProbeExample]] = None,
        val_tokens: Optional[torch.Tensor] = None,
        categories: Optional[Sequence[str]] = None,
        normalize_length: bool = True,
    ) -> EvaluationReport:
        """Run the full benchmark suite on the model."""
        loss_metrics = None
        if val_tokens is not None:
            loss_metrics = evaluate_tokens_loss(self.model, val_tokens, device=self.device)

        if probes is None:
            probes = get_standard_probes()

        if categories:
            probes = filter_probes(probes, categories)

        category_counts: dict[str, dict[str, int]] = {}
        sample_results: list[MultipleChoiceResult] = []

        total_correct = 0
        total_probes = len(probes)
        total_chance_sum = 0.0

        for probe in probes:
            result = self.mc_evaluator.evaluate_question(
                prompt=probe.prompt,
                choices=probe.choices,
                correct_index=probe.correct_index,
                normalize_length=normalize_length,
            )
            sample_results.append(result)

            cat = probe.category
            if cat not in category_counts:
                category_counts[cat] = {"total": 0, "correct": 0, "chance_choices": 0}
            category_counts[cat]["total"] += 1
            category_counts[cat]["chance_choices"] += len(probe.choices)

            if result.is_correct:
                total_correct += 1
                category_counts[cat]["correct"] += 1

            total_chance_sum += 1.0 / len(probe.choices)

        category_scores: dict[str, CategoryScore] = {}
        for cat, counts in category_counts.items():
            tot = counts["total"]
            corr = counts["correct"]
            acc = corr / tot if tot > 0 else 0.0
            avg_chance = tot / counts["chance_choices"] if counts["chance_choices"] > 0 else 0.25
            category_scores[cat] = CategoryScore(
                category=cat,
                total=tot,
                correct=corr,
                accuracy=acc,
                chance_accuracy=avg_chance,
            )

        overall_acc = total_correct / total_probes if total_probes > 0 else 0.0
        avg_chance_all = total_chance_sum / total_probes if total_probes > 0 else 0.25

        return EvaluationReport(
            model_name=self.model_name,
            device=str(self.device),
            loss_metrics=loss_metrics,
            category_scores=category_scores,
            overall_total_probes=total_probes,
            overall_correct_probes=total_correct,
            overall_accuracy=overall_acc,
            random_chance_accuracy=avg_chance_all,
            sample_results=sample_results,
        )
