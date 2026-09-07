"""
Tests for Preference Dataset & Token Collator (Phase 22)
"""

import torch

from packages.training.preference_dataset import PreferenceDataset, PreferenceSample


def test_preference_dataset_indexing():
    samples = [
        PreferenceSample(
            prompt="Hello, who are you?",
            chosen="I am an AI assistant built from first principles.",
            rejected="None of your business.",
        )
    ]
    dataset = PreferenceDataset(samples=samples, max_length=64)
    assert len(dataset) == 1

    item = dataset[0]
    assert "chosen_input_ids" in item
    assert "chosen_labels" in item
    assert "rejected_input_ids" in item
    assert "rejected_labels" in item
    assert "prompt_length" in item

    # Prompt tokens should be masked with -100 in labels
    p_len = item["prompt_length"]
    assert p_len > 0
    assert all(label == -100 for label in item["chosen_labels"][:p_len])
    assert all(label == -100 for label in item["rejected_labels"][:p_len])

    # Completion tokens should NOT be -100
    assert any(label != -100 for label in item["chosen_labels"][p_len:])
    assert any(label != -100 for label in item["rejected_labels"][p_len:])


def test_preference_dataset_collation():
    samples = [
        PreferenceSample(prompt="Q1", chosen="Long chosen answer 1", rejected="Short"),
        PreferenceSample(prompt="Question 2", chosen="A2", rejected="Rejected completion 2"),
    ]
    dataset = PreferenceDataset(samples=samples, max_length=64)
    batch = PreferenceDataset.collate_fn([dataset[0], dataset[1]], pad_token_id=0)

    assert isinstance(batch["chosen_input_ids"], torch.Tensor)
    assert isinstance(batch["chosen_labels"], torch.Tensor)
    assert isinstance(batch["rejected_input_ids"], torch.Tensor)
    assert isinstance(batch["rejected_labels"], torch.Tensor)

    # Check batch shapes (B, max_len)
    assert batch["chosen_input_ids"].shape[0] == 2
    assert batch["rejected_input_ids"].shape[0] == 2
    assert batch["chosen_labels"].shape == batch["chosen_input_ids"].shape
    assert batch["rejected_labels"].shape == batch["rejected_input_ids"].shape
