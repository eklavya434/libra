import pytest
import torch
import torch.nn as nn

from packages.evaluation.multiple_choice import MultipleChoiceEvaluator, MultipleChoiceResult


class DummyTokenizer:
    """Simple character/word mapping for deterministic testing."""

    def __init__(self):
        self.vocab = {"<PAD>": 0, "Hello": 1, "world": 2, "Paris": 3, "London": 4, "Rome": 5}

    def encode(self, text: str) -> list[int]:
        words = text.strip().split()
        return [self.vocab.get(w, 0) for w in words]


class DeterministicMockModel(nn.Module):
    """Model that outputs high logit for Paris (token 3) and low for others."""

    def __init__(self, vocab_size: int = 10):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size)
        # Strongly predict token 3 ("Paris")
        logits[:, :, 3] = 10.0
        logits[:, :, 4] = 2.0   # "London"
        logits[:, :, 5] = -5.0  # "Rome"
        return logits


def test_multiple_choice_evaluator_ranking():
    model = DeterministicMockModel()
    tokenizer = DummyTokenizer()
    evaluator = MultipleChoiceEvaluator(model, tokenizer)

    prompt = "Hello world"
    choices = ["Paris", "London", "Rome"]
    # Paris is index 0
    result = evaluator.evaluate_question(prompt, choices, correct_index=0)

    assert isinstance(result, MultipleChoiceResult)
    assert result.predicted_index == 0
    assert result.is_correct is True

    # Check that Paris score > London score > Rome score
    paris_score = result.scores[0].total_log_likelihood
    london_score = result.scores[1].total_log_likelihood
    rome_score = result.scores[2].total_log_likelihood

    assert paris_score > london_score > rome_score


def test_multiple_choice_result_serialization():
    model = DeterministicMockModel()
    tokenizer = DummyTokenizer()
    evaluator = MultipleChoiceEvaluator(model, tokenizer)

    result = evaluator.evaluate_question("Hello", ["Paris", "London"], correct_index=1)
    # Target was London (index 1), but model predicted Paris (index 0)
    assert result.is_correct is False

    d = result.to_dict()
    assert d["prompt"] == "Hello"
    assert d["is_correct"] is False
    assert len(d["scores"]) == 2
