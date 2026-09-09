"""
Tests for SFT Dataset and Sequence Packing (Phase 33)
"""

import torch

from packages.training.chat_formatter import IGNORE_INDEX, ChatMessage
from packages.training.sft_dataset import InstructionDataset, pack_sequences


def test_instruction_dataset():
    dialogues = [
        [
            ChatMessage(role="user", content="Explain quantum computing."),
            ChatMessage(role="assistant", content="Quantum computing uses qubits..."),
        ],
        [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Say hello."),
            ChatMessage(role="assistant", content="Hello! How can I help?"),
        ],
    ]

    dataset = InstructionDataset(dialogues=dialogues, max_length=128)
    assert len(dataset) == 2

    item0 = dataset[0]
    assert "input_ids" in item0
    assert "labels" in item0
    assert isinstance(item0["input_ids"], torch.Tensor)
    assert isinstance(item0["labels"], torch.Tensor)
    assert item0["input_ids"].shape == item0["labels"].shape

    # First turn is user, so its labels must be IGNORE_INDEX
    assert item0["labels"][0].item() == IGNORE_INDEX
    # Assistant tokens must be active
    assert any(label != IGNORE_INDEX for label in item0["labels"].tolist())


def test_pack_sequences():
    # Create two short dialogues
    dialogue_tokens = [
        ([1, 2, 3], [-100, -100, 3]),
        ([4, 5], [-100, 5]),
    ]

    packed = pack_sequences(dialogue_tokens, max_length=10, eos_token_id=99)
    # Total tokens: 3 + 1 (EOS) + 2 + 1 (EOS) = 7 <= 10, so they fit in 1 sequence
    assert len(packed) == 1
    assert packed[0]["input_ids"].tolist() == [1, 2, 3, 99, 4, 5, 99]
    assert packed[0]["labels"].tolist() == [-100, -100, 3, IGNORE_INDEX, -100, 5, IGNORE_INDEX]


def test_pack_sequences_overflow():
    # When items exceed max_length, pack into multiple sequences
    dialogue_tokens = [
        ([1, 2, 3, 4], [-100, -100, 3, 4]),
        ([5, 6, 7], [-100, 6, 7]),
    ]

    packed = pack_sequences(dialogue_tokens, max_length=6, eos_token_id=0)
    # Dialogue 1: 4 + 1 EOS = 5. Dialogue 2: 3 + 1 EOS = 4. 5 + 4 > 6. So 2 batches.
    assert len(packed) == 2
    assert packed[0]["input_ids"].tolist() == [1, 2, 3, 4, 0]
    assert packed[1]["input_ids"].tolist() == [5, 6, 7, 0]
