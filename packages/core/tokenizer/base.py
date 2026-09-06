"""
Libra Tokenizers - Base Tokenizer Interface
Abstract definition establishing the contract for all tokenizers in Libra.
"""

from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """Abstract Base Class for Libra Tokenizers."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary."""

    @property
    @abstractmethod
    def special_tokens(self) -> dict[str, int]:
        """Mapping of special token symbols to their integer IDs."""

    @property
    def pad_token_id(self) -> int | None:
        return self.special_tokens.get("<PAD>")

    @property
    def unk_token_id(self) -> int | None:
        return self.special_tokens.get("<UNK>")

    @property
    def bos_token_id(self) -> int | None:
        return self.special_tokens.get("<BOS>")

    @property
    def eos_token_id(self) -> int | None:
        return self.special_tokens.get("<EOS>")

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Converts a string of text into a list of integer token IDs."""

    @abstractmethod
    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        """Converts a list of integer token IDs back into a human-readable string."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Saves the tokenizer vocabulary and merge rules to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseTokenizer":
        """Loads a saved tokenizer from disk."""
