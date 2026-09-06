"""
Libra Tokenizers - Educational Byte-Pair Encoding (BPE) Tokenizer
Built from mathematical first principles in pure Python.

Demonstrates:
  1. Base Byte Vocabulary (0-255) + Special Tokens (<PAD>, <UNK>, <BOS>, <EOS>)
  2. Pair Frequency Statistics
  3. Iterative Subword Merging
  4. Compression Ratio Improvement
"""

import json
from itertools import pairwise
from typing import Any

from packages.core.tokenizer.base import BaseTokenizer


class EducationalBPETokenizer(BaseTokenizer):
    """Pure-Python Byte-Pair Encoding Tokenizer for transparent educational experimentation."""

    def __init__(self) -> None:
        self._special_tokens: dict[str, int] = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }
        # Inverse mapping for special tokens
        self._special_tokens_inv = {v: k for k, v in self._special_tokens.items()}

        # Base byte vocab offset: 4 to 259 correspond to raw bytes 0 to 255
        self._byte_offset = len(self._special_tokens)
        self.vocab: dict[int, bytes] = {}
        for b in range(256):
            self.vocab[self._byte_offset + b] = bytes([b])

        # Merges dictionary: maps tuple (token_a, token_b) -> new_token_id
        self.merges: dict[tuple[int, int], int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self._special_tokens)

    @property
    def special_tokens(self) -> dict[str, int]:
        return self._special_tokens

    def _get_stats(self, tokens: list[int]) -> dict[tuple[int, int], int]:
        """Counts frequency of all adjacent token pairs."""
        counts: dict[tuple[int, int], int] = {}
        for pair in pairwise(tokens):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_tokens(self, tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replaces all occurrences of `pair` in `tokens` with `new_id`."""
        new_tokens: list[int] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def train(self, text: str, num_merges: int = 50, min_frequency: int = 2) -> None:
        """Trains BPE merge rules on raw text."""
        raw_bytes = list(text.encode("utf-8"))
        # Map raw bytes to initial token IDs (offset by special tokens)
        tokens = [self._byte_offset + b for b in raw_bytes]

        for _ in range(num_merges):
            stats = self._get_stats(tokens)
            if not stats:
                break

            # Find the most frequent adjacent pair
            best_pair, best_count = max(stats.items(), key=lambda x: x[1])
            if best_count < min_frequency:
                # Stop if no pair occurs frequently enough
                break

            new_id = self.vocab_size
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            tokens = self._merge_tokens(tokens, best_pair, new_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encodes text into subword token IDs using learned merge rules."""
        raw_bytes = list(text.encode("utf-8"))
        tokens = [self._byte_offset + b for b in raw_bytes]

        # Iteratively apply merges in the order they were created
        for pair, new_id in self.merges.items():
            tokens = self._merge_tokens(tokens, pair, new_id)

        if add_special_tokens:
            tokens = [self._special_tokens["<BOS>"]] + tokens + [self._special_tokens["<EOS>"]]

        return tokens

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        """Decodes token IDs back into a UTF-8 string."""
        byte_chunks: list[bytes] = []
        for t in tokens:
            if t in self._special_tokens_inv:
                if not skip_special_tokens:
                    byte_chunks.append(self._special_tokens_inv[t].encode("utf-8"))
            elif t in self.vocab:
                byte_chunks.append(self.vocab[t])
            else:
                # Fallback for unknown IDs
                if not skip_special_tokens:
                    byte_chunks.append(b"<UNK>")

        full_bytes = b"".join(byte_chunks)
        return full_bytes.decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        """Serializes the tokenizer vocabulary and merge rules to JSON."""
        # Convert tuple keys in merges to string representations for JSON
        merges_serialized = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        vocab_serialized = {str(k): list(v) for k, v in self.vocab.items()}

        payload = {
            "special_tokens": self._special_tokens,
            "merges": merges_serialized,
            "vocab": vocab_serialized,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EducationalBPETokenizer":
        """Loads a saved EducationalBPETokenizer from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)

        tokenizer = cls()
        tokenizer._special_tokens = data["special_tokens"]
        tokenizer._special_tokens_inv = {v: k for k, v in tokenizer._special_tokens.items()}

        tokenizer.merges = {}
        for k_str, v in data["merges"].items():
            p1, p2 = map(int, k_str.split(","))
            tokenizer.merges[(p1, p2)] = v

        tokenizer.vocab = {}
        for k_str, byte_list in data["vocab"].items():
            tokenizer.vocab[int(k_str)] = bytes(byte_list)

        return tokenizer
