"""
Libra Training - Kahneman-Tversky Optimization (KTO) Dataset & Collator

Handles unpaired binary preference feedback (prompt, completion, is_desirable).
Tokenizes prompt and completion, masks prompt positions with -100 for loss computation,
and formats batches for the KTOTrainer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class KTOSample:
    """A single unpaired binary preference observation."""

    prompt: str
    completion: str
    is_desirable: bool


def default_char_tokenizer(text: str) -> list[int]:
    """Default ASCII/byte level tokenizer for educational models."""
    return list(text.encode("utf-8"))


class KTODataset(Dataset):
    """Dataset of unpaired binary feedback (prompt, completion, is_desirable)."""

    def __init__(
        self,
        samples: list[KTOSample],
        tokenizer: Callable[[str], list[int]] | None = None,
        max_length: int = 128,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer or default_char_tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        prompt_tokens = self.tokenizer(sample.prompt)
        completion_tokens = self.tokenizer(sample.completion)

        full_seq = (prompt_tokens + completion_tokens)[: self.max_length]
        prompt_len = min(len(prompt_tokens), len(full_seq))

        labels = [-100] * prompt_len + full_seq[prompt_len:]

        return {
            "input_ids": full_seq,
            "labels": labels,
            "is_desirable": sample.is_desirable,
            "prompt_length": prompt_len,
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]], pad_token_id: int = 0) -> dict[str, torch.Tensor]:
        """Pads input sequences and labels to uniform batch tensor dimensions."""
        b_size = len(batch)
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids = torch.full((b_size, max_len), pad_token_id, dtype=torch.long)
        labels = torch.full((b_size, max_len), -100, dtype=torch.long)
        is_desirable = torch.zeros(b_size, dtype=torch.bool)
        prompt_lengths = torch.zeros(b_size, dtype=torch.long)

        for i, item in enumerate(batch):
            seq_len = len(item["input_ids"])
            input_ids[i, :seq_len] = torch.tensor(item["input_ids"], dtype=torch.long)
            labels[i, :seq_len] = torch.tensor(item["labels"], dtype=torch.long)
            is_desirable[i] = bool(item["is_desirable"])
            prompt_lengths[i] = item["prompt_length"]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "is_desirable": is_desirable,
            "prompt_lengths": prompt_lengths,
        }
