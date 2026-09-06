"""
Libra Tokenizers - Production Hugging Face Byte-Level BPE Tokenizer
High-performance Rust-backed tokenizer for scaling beyond small toy experiments.
"""

import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from packages.core.tokenizer.base import BaseTokenizer


class HFTokenizer(BaseTokenizer):
    """Production wrapper around Hugging Face Byte-Level BPE Tokenizer."""

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self._special_tokens: dict[str, int] = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }

        if tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            # Initialize new Byte-Level BPE tokenizer
            self._tokenizer = Tokenizer(models.BPE())
            self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self._tokenizer.decoder = decoders.ByteLevel()

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def special_tokens(self) -> dict[str, int]:
        return self._special_tokens

    def train_from_iterator(self, texts: list[str], vocab_size: int = 1000) -> None:
        """Trains the tokenizer on an iterator of strings."""
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        self._tokenizer.train_from_iterator(texts, trainer=trainer)

    def train_from_files(self, files: list[str], vocab_size: int = 1000) -> None:
        """Trains the tokenizer directly from file paths."""
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        self._tokenizer.train(files, trainer=trainer)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        encoding = self._tokenizer.encode(text)
        ids = encoding.ids
        if add_special_tokens:
            ids = [self.bos_token_id or 2] + ids + [self.eos_token_id or 3]
        return ids

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._tokenizer.save(path)

    @classmethod
    def load(cls, path: str) -> "HFTokenizer":
        tok = Tokenizer.from_file(path)
        return cls(tokenizer=tok)
