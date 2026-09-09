import json

import torch
import torch.nn as nn

from packages.evaluation.harness import EvaluationHarness, EvaluationReport
from packages.evaluation.probes import ProbeExample, filter_probes, get_standard_probes


class DummyTokenizer:
    def __init__(self):
        self.vocab = {"A": 1, "B": 2, "C": 3}

    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(c, 1) for c in text.strip()]


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len = x.shape
        # Return uniform logits for vocab of 8
        return torch.zeros(batch, seq_len, 8)


def test_standard_probes_structure():
    probes = get_standard_probes()
    assert len(probes) >= 10

    categories = {p.category for p in probes}
    assert "mathematics" in categories
    assert "reasoning" in categories
    assert "coding" in categories
    assert "instruction_following" in categories
    assert "safety" in categories

    math_probes = filter_probes(probes, ["mathematics"])
    assert all(p.category == "mathematics" for p in math_probes)


def test_harness_end_to_end(tmp_path):
    model = DummyModel()
    tokenizer = DummyTokenizer()
    harness = EvaluationHarness(model, tokenizer, model_name="DummyTestModel")

    probes = [
        ProbeExample(prompt="A", choices=["A", "B"], correct_index=0, category="reasoning"),
        ProbeExample(prompt="B", choices=["A", "B"], correct_index=1, category="reasoning"),
    ]

    report = harness.run_benchmark(probes=probes)
    assert isinstance(report, EvaluationReport)
    assert report.model_name == "DummyTestModel"
    assert report.overall_total_probes == 2
    assert "reasoning" in report.category_scores

    # Markdown formatting verification
    md = report.to_markdown()
    assert "# Benchmark Evaluation Report: `DummyTestModel`" in md
    assert "Reasoning" in md

    # JSON export verification
    json_file = tmp_path / "report.json"
    report.save_json(json_file)
    assert json_file.exists()

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["model_name"] == "DummyTestModel"
    assert data["overall_total_probes"] == 2
