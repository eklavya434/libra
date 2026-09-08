"""
Libra Training - Supervised Fine-Tuning (SFT) Dataset & Multi-Turn Sequence Packing
"""

from __future__ import annotations

from typing import Callable, Any
import torch
from torch.utils.data import Dataset

from packages.training.chat_formatter import (
    ChatMessage,
    IGNORE_INDEX,
    tokenize_with_loss_masking,
)


def default_tokenizer(text: str) -> list[int]:
    """Default ASCII byte-level tokenizer for CPU experiments."""
    return list(text.encode("utf-8"))


class InstructionDataset(Dataset):
    """PyTorch Dataset for multi-turn instruction fine-tuning with prompt loss masking."""

    def __init__(
        self,
        dialogues: list[list[ChatMessage]],
        tokenizer: Callable[[str], list[int]] | None = None,
        max_length: int = 512,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self.dialogues = dialogues
        self.tokenizer = tokenizer or default_tokenizer
        self.max_length = max_length
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        dialogue = self.dialogues[idx]
        input_ids, label_ids = tokenize_with_loss_masking(
            messages=dialogue,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            ignore_index=self.ignore_index,
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


def pack_sequences(
    dialogue_tokens: list[tuple[list[int], list[int]]],
    max_length: int = 512,
    eos_token_id: int = 0,
) -> list[dict[str, torch.Tensor]]:
    """Packs multiple short dialogues into contiguous context windows up to max_length.

    Eliminates padding token waste.
    """
    packed_batches: list[dict[str, torch.Tensor]] = []
    curr_inputs: list[int] = []
    curr_labels: list[int] = []

    for inputs, labels in dialogue_tokens:
        # Check if adding this dialogue exceeds max_length
        if len(curr_inputs) + len(inputs) + 1 > max_length and curr_inputs:
            # Finalize current packed sequence
            packed_batches.append({
                "input_ids": torch.tensor(curr_inputs, dtype=torch.long),
                "labels": torch.tensor(curr_labels, dtype=torch.long),
            })
            curr_inputs = []
            curr_labels = []

        curr_inputs.extend(inputs)
        curr_labels.extend(labels)
        # Append EOS separator
        curr_inputs.append(eos_token_id)
        curr_labels.append(IGNORE_INDEX)

    if curr_inputs:
        packed_batches.append({
            "input_ids": torch.tensor(curr_inputs, dtype=torch.long),
            "labels": torch.tensor(curr_labels, dtype=torch.long),
        })

    return packed_batches
