"""
Libra Providers - Prompt Formatting & Chat Templates

Transforms structured conversation messages into model-specific input prompts:
  1. ChatML (Qwen, DeepSeek, Mistral)
  2. Llama 3 / 3.2 format
  3. Simple Plain Text format
"""

from __future__ import annotations

from typing import Sequence


class PromptTemplate:
    """Formatter that compiles message dictionaries into delimited prompt strings."""

    @staticmethod
    def format_chatml(
        messages: Sequence[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Formats messages using the standard ChatML syntax (<|im_start|>role\ncontent<|im_end|>)."""
        chunks = []
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "").strip()
            chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        if add_generation_prompt:
            chunks.append("<|im_start|>assistant\n")

        return "\n".join(chunks)

    @staticmethod
    def format_llama3(
        messages: Sequence[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Formats messages using Llama 3 / 3.2 special header tokens (<|start_header_id|>...<|eot_id|>)."""
        chunks = ["<|begin_of_text|>"]
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "").strip()
            chunks.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")

        if add_generation_prompt:
            chunks.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

        return "".join(chunks)

    @staticmethod
    def format_plain(messages: Sequence[dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Simple plain text fallback format (User: ... \nAssistant: ...)."""
        chunks = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "").strip()
            chunks.append(f"{role}: {content}")

        if add_generation_prompt:
            chunks.append("Assistant: ")

        return "\n\n".join(chunks)

    @classmethod
    def format(
        cls,
        messages: Sequence[dict[str, str]],
        style: str = "chatml",
        add_generation_prompt: bool = True,
    ) -> str:
        """Dispatches message formatting based on style name."""
        style_lower = style.lower()
        if "llama" in style_lower:
            return cls.format_llama3(messages, add_generation_prompt=add_generation_prompt)
        elif "plain" in style_lower or "text" in style_lower:
            return cls.format_plain(messages, add_generation_prompt=add_generation_prompt)
        else:
            # Default to widely supported ChatML
            return cls.format_chatml(messages, add_generation_prompt=add_generation_prompt)

    @staticmethod
    def get_stop_sequences(style: str = "chatml") -> list[str]:
        """Returns the canonical stop tokens for the specified template style."""
        style_lower = style.lower()
        if "llama" in style_lower:
            return ["<|eot_id|>", "<|end_of_text|>"]
        elif "plain" in style_lower:
            return ["\nUser:", "\nSystem:"]
        else:
            return ["<|im_end|>", "<|endoftext|>"]
