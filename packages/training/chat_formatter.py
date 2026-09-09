"""
Libra Training - ChatML Formatter & Prompt Loss Masker (Phase 33)
Implements:
1. Message role representation (system, user, assistant)
2. ChatML template serialization (<|im_start|>role\ncontent<|im_end|>)
3. Completion-only label masking (replacing prompt token labels with -100)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

IGNORE_INDEX = -100


@dataclass
class ChatMessage:
    role: str
    content: str


class ChatMLFormatter:
    """Standard OpenAI/LLaMA ChatML delimiter template formatter."""

    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"

    def format_turn(self, message: ChatMessage) -> str:
        """Formats a single conversational turn."""
        return f"{self.IM_START}{message.role}\n{message.content}{self.IM_END}\n"

    def format_conversation(self, messages: list[ChatMessage]) -> str:
        """Concatenates conversation history into standard ChatML string."""
        return "".join(self.format_turn(m) for m in messages)

    def extract_role_spans(self, messages: list[ChatMessage]) -> list[tuple[str, int, int]]:
        """Extracts character start and end offsets for each message in the serialized string."""
        spans = []
        current_offset = 0
        for m in messages:
            turn_str = self.format_turn(m)
            start = current_offset
            end = current_offset + len(turn_str)
            spans.append((m.role, start, end))
            current_offset = end
        return spans


def tokenize_with_loss_masking(
    messages: list[ChatMessage],
    tokenizer: Callable[[str], list[int]],
    max_length: int = 512,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[list[int], list[int]]:
    """Tokenizes a multi-turn conversation and masks prompt tokens with ignore_index (-100).

    Only assistant completion tokens have active labels (for computing loss).
    System prompts and user turns are masked out.

    Returns:
        Tuple of (input_ids, label_ids)
    """
    formatter = ChatMLFormatter()
    input_ids: list[int] = []
    label_ids: list[int] = []

    for m in messages:
        turn_text = formatter.format_turn(m)
        tokens = tokenizer(turn_text)

        input_ids.extend(tokens)
        if m.role == "assistant":
            # Assistant completion tokens are active for cross-entropy loss
            label_ids.extend(tokens)
        else:
            # System and User turns are ignored during backprop
            label_ids.extend([ignore_index] * len(tokens))

    # Truncate to max_length
    input_ids = input_ids[:max_length]
    label_ids = label_ids[:max_length]

    return input_ids, label_ids
