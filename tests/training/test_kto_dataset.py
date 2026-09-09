"""Unit tests for KTODataset and KTOSample collator."""

import torch

from packages.training.kto_dataset import KTODataset, KTOSample


def test_kto_dataset_item_and_masking():
    samples = [
        KTOSample(prompt="Hello", completion=" world!", is_desirable=True),
        KTOSample(prompt="Bad prompt", completion=" toxic answer", is_desirable=False),
    ]
    dataset = KTODataset(samples=samples, max_length=64)

    assert len(dataset) == 2

    item0 = dataset[0]
    assert "input_ids" in item0
    assert "labels" in item0
    assert item0["is_desirable"] is True

    prompt_len = item0["prompt_length"]
    assert prompt_len == len("Hello".encode("utf-8"))

    # Prompt positions in labels should be -100
    for i in range(prompt_len):
        assert item0["labels"][i] == -100

    # Completion positions in labels should match input_ids
    for i in range(prompt_len, len(item0["input_ids"])):
        assert item0["labels"][i] == item0["input_ids"][i]


def test_kto_collate_fn():
    samples = [
        KTOSample(prompt="A", completion=" short", is_desirable=True),
        KTOSample(
            prompt="Much longer prompt", completion=" longer completion text", is_desirable=False
        ),
    ]
    dataset = KTODataset(samples=samples, max_length=64)
    raw_batch = [dataset[i] for i in range(len(dataset))]

    batch = KTODataset.collate_fn(raw_batch, pad_token_id=0)

    assert isinstance(batch["input_ids"], torch.Tensor)
    assert isinstance(batch["labels"], torch.Tensor)
    assert isinstance(batch["is_desirable"], torch.Tensor)
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape[0] == 2

    # Check boolean values
    assert batch["is_desirable"][0].item() is True
    assert batch["is_desirable"][1].item() is False
