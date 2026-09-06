"""
Libra Core Memory - Context Window Manager

Educational context window manager that tracks token budgets, preserves foundational
system prompts, and applies sliding-window truncation so multi-turn chats never
exceed the attention capacity of local CPU transformer models.
"""

from __future__ import annotations

import math
from typing import Any, Optional
from pydantic import BaseModel, Field

from packages.core.memory.models import Message


class TruncatedContext(BaseModel):
    """Result of context window formatting and truncation."""

    messages: list[dict[str, str]] = Field(..., description="Processed message dicts ready for provider")
    total_tokens: int = Field(..., description="Total tokens consumed by processed messages")
    truncated_count: int = Field(0, description="Number of older messages dropped from context")
    max_context_tokens: int = Field(..., description="Upper limit of model context window")
    reserved_completion_tokens: int = Field(..., description="Tokens reserved for model generation")
    remaining_capacity: int = Field(..., description="Remaining token headroom")


class ContextWindowManager:
    """Manages prompt history truncation and token allocation for LLM inference."""

    def __init__(
        self,
        max_context_tokens: int = 2048,
        reserved_completion_tokens: int = 512,
        chars_per_token: float = 3.8,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.reserved_completion_tokens = reserved_completion_tokens
        self.chars_per_token = chars_per_token

    def estimate_tokens(self, text: str) -> int:
        """Heuristic token estimator based on subword compression ratio (~3.8 chars/tok)."""
        if not text:
            return 0
        # Add 4 tokens for role and turn delimiter overhead
        return max(1, math.ceil(len(text) / self.chars_per_token)) + 4

    def count_message_tokens(self, msg: Message | dict[str, Any]) -> int:
        """Calculate token count for a message object or dictionary."""
        if isinstance(msg, Message) and msg.token_count > 0:
            return msg.token_count + 4
        content = msg.content if isinstance(msg, Message) else msg.get("content", "")
        return self.estimate_tokens(content)

    def prepare_context(
        self,
        messages: list[Message] | list[dict[str, Any]],
        max_context: Optional[int] = None,
        reserved_tokens: Optional[int] = None,
        override_system_prompt: Optional[str] = None,
    ) -> TruncatedContext:
        """
        Fits conversation history into token budget using sliding window.

        Key Rules:
        1. System prompt is ALWAYS preserved as the anchor instructions.
        2. The latest user message is always preserved.
        3. Intermediate older turns are dropped chronologically from the oldest first.
        """
        budget_max = max_context or self.max_context_tokens
        reserved = reserved_tokens or self.reserved_completion_tokens
        allowed_input_tokens = max(16, budget_max - reserved)

        # Normalize raw input into list of dicts
        normalized: list[dict[str, str]] = []
        for m in messages:
            if isinstance(m, Message):
                normalized.append({"role": m.role, "content": m.content})
            elif isinstance(m, dict):
                normalized.append({"role": str(m.get("role", "user")), "content": str(m.get("content", ""))})

        # Separate system messages and dialog turns
        system_msgs: list[dict[str, str]] = []
        dialog_turns: list[dict[str, str]] = []

        if override_system_prompt and override_system_prompt.strip():
            system_msgs.append({"role": "system", "content": override_system_prompt.strip()})

        for m in normalized:
            if m["role"] == "system":
                if not override_system_prompt:
                    system_msgs.append(m)
            else:
                dialog_turns.append(m)

        # Calculate system token footprint
        system_tokens = sum(self.estimate_tokens(s["content"]) for s in system_msgs)
        dialog_budget = max(8, allowed_input_tokens - system_tokens)

        # Work backwards from the newest dialog turns
        selected_turns: list[dict[str, str]] = []
        accumulated_tokens = 0
        truncated_count = 0

        for turn in reversed(dialog_turns):
            turn_tok = self.estimate_tokens(turn["content"])
            if accumulated_tokens + turn_tok <= dialog_budget:
                selected_turns.append(turn)
                accumulated_tokens += turn_tok
            else:
                # If even the latest turn exceeds budget on its own, include it truncated
                if not selected_turns:
                    # Truncate content of this critical single turn
                    max_chars = int(dialog_budget * self.chars_per_token)
                    truncated_content = turn["content"][:max_chars] + "..."
                    selected_turns.append({"role": turn["role"], "content": truncated_content})
                    accumulated_tokens += self.estimate_tokens(truncated_content)
                else:
                    truncated_count += 1

        # Re-assemble in original chronological order
        selected_turns.reverse()
        final_messages = system_msgs + selected_turns
        final_tokens = system_tokens + accumulated_tokens
        remaining = max(0, budget_max - (final_tokens + reserved))

        return TruncatedContext(
            messages=final_messages,
            total_tokens=final_tokens,
            truncated_count=truncated_count,
            max_context_tokens=budget_max,
            reserved_completion_tokens=reserved,
            remaining_capacity=remaining,
        )
