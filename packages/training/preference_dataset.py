"""
Libra Training - Preference Dataset & Token Collator for Alignment (RLHF & DPO)

Manages pairwise preference data (prompt, chosen, rejected) and produces
tensors with prompt masking (label -100) for completion-only log-likelihood computation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class PreferenceSample:
    """A single pairwise preference observation."""

    prompt: str
    chosen: str
    rejected: str


def default_char_tokenizer(text: str) -> list[int]:
    """Default ASCII/byte level tokenizer for educational models."""
    return list(text.encode("utf-8"))


class PreferenceDataset(Dataset):
    """Dataset of (prompt, chosen, rejected) pairs formatted for DPO & Reward Modeling."""

    def __init__(
        self,
        samples: list[PreferenceSample],
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
        chosen_tokens = self.tokenizer(sample.chosen)
        rejected_tokens = self.tokenizer(sample.rejected)

        # Full sequence: prompt + completion
        chosen_seq = prompt_tokens + chosen_tokens
        rejected_seq = prompt_tokens + rejected_tokens

        # Truncate if exceeds max_length
        chosen_seq = chosen_seq[: self.max_length]
        rejected_seq = rejected_seq[: self.max_length]

        prompt_len = min(len(prompt_tokens), self.max_length)

        # Labels: prompt tokens masked with -100 (ignored in loss computation)
        chosen_labels = [-100] * prompt_len + chosen_seq[prompt_len:]
        rejected_labels = [-100] * prompt_len + rejected_seq[prompt_len:]

        return {
            "chosen_input_ids": chosen_seq,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_seq,
            "rejected_labels": rejected_labels,
            "prompt_length": prompt_len,
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]], pad_token_id: int = 0) -> dict[str, torch.Tensor]:
        """Pads chosen and rejected sequences in batch to uniform tensor dimensions."""
        chosen_max_len = max(len(item["chosen_input_ids"]) for item in batch)
        rejected_max_len = max(len(item["rejected_input_ids"]) for item in batch)

        b_size = len(batch)
        chosen_ids = torch.full((b_size, chosen_max_len), pad_token_id, dtype=torch.long)
        chosen_labels = torch.full((b_size, chosen_max_len), -100, dtype=torch.long)

        rejected_ids = torch.full((b_size, rejected_max_len), pad_token_id, dtype=torch.long)
        rejected_labels = torch.full((b_size, rejected_max_len), -100, dtype=torch.long)

        prompt_lens = torch.zeros(b_size, dtype=torch.long)

        for i, item in enumerate(batch):
            c_len = len(item["chosen_input_ids"])
            r_len = len(item["rejected_input_ids"])

            chosen_ids[i, :c_len] = torch.tensor(item["chosen_input_ids"], dtype=torch.long)
            chosen_labels[i, :c_len] = torch.tensor(item["chosen_labels"], dtype=torch.long)

            rejected_ids[i, :r_len] = torch.tensor(item["rejected_input_ids"], dtype=torch.long)
            rejected_labels[i, :r_len] = torch.tensor(item["rejected_labels"], dtype=torch.long)

            prompt_lens[i] = item["prompt_length"]

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_ids,
            "rejected_labels": rejected_labels,
            "prompt_lengths": prompt_lens,
        }
