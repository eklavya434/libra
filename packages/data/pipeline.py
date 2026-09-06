"""
Libra Data - End-to-End Data Preprocessing & Sharding Pipeline
Coordinates: Quota Validation -> Cleaning -> Deduplication -> Split -> Tokenization -> Binary Sharding.
"""

import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from packages.core.tokenizer.base import BaseTokenizer
from packages.data.cleaning import clean_text
from packages.data.config import DataPipelineConfig
from packages.data.deduplication import deduplicate_paragraphs
from packages.data.quota import validate_storage_quota


@dataclass
class DataPipelineReport:
    raw_chars: int
    cleaned_chars: int
    duplicates_removed: int
    train_tokens: int
    val_tokens: int
    train_shard_path: str
    val_shard_path: str
    train_shard_size_kb: float
    val_shard_size_kb: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DataPipeline:
    """Orchestrates data preparation and saves compact binary token shards."""

    def __init__(self, config: DataPipelineConfig, tokenizer: BaseTokenizer) -> None:
        self.config = config
        self.tokenizer = tokenizer

    def process_text(self, raw_text: str) -> DataPipelineReport:
        """Processes raw text into cleaned, deduplicated, tokenized binary shards."""
        # 1. Enforce Storage Quota check (estimate ~2x text size in memory/disk)
        estimated_mb = len(raw_text.encode("utf-8")) / (1024 * 1024)
        validate_storage_quota(additional_mb=estimated_mb)

        # 2. Text Cleaning
        cleaned, _clean_stats = clean_text(raw_text, self.config)

        # 3. Deduplication
        if self.config.deduplicate:
            deduped, dedup_stats = deduplicate_paragraphs(cleaned)
            dups_removed = dedup_stats["duplicates_removed"]
        else:
            deduped = cleaned
            dups_removed = 0

        # 4. Train / Validation Split
        paragraphs = [p for p in deduped.split("\n\n") if p.strip()]
        split_idx = max(1, int(len(paragraphs) * self.config.train_ratio))
        train_paragraphs = paragraphs[:split_idx]
        val_paragraphs = paragraphs[split_idx:] if split_idx < len(paragraphs) else paragraphs[-1:]

        train_text = "\n\n".join(train_paragraphs)
        val_text = "\n\n".join(val_paragraphs)

        # 5. Tokenization
        train_tokens = self.tokenizer.encode(train_text, add_special_tokens=True)
        val_tokens = self.tokenizer.encode(val_text, add_special_tokens=True)

        # 6. Save Compact Binary Shards (uint16 format)
        os.makedirs(self.config.output_dir, exist_ok=True)
        train_path = os.path.join(self.config.output_dir, "train.bin")
        val_path = os.path.join(self.config.output_dir, "val.bin")

        train_arr = np.array(train_tokens, dtype=np.uint16)
        val_arr = np.array(val_tokens, dtype=np.uint16)

        train_arr.tofile(train_path)
        val_arr.tofile(val_path)

        train_size_kb = round(os.path.getsize(train_path) / 1024, 2)
        val_size_kb = round(os.path.getsize(val_path) / 1024, 2)

        return DataPipelineReport(
            raw_chars=len(raw_text),
            cleaned_chars=len(deduped),
            duplicates_removed=dups_removed,
            train_tokens=len(train_tokens),
            val_tokens=len(val_tokens),
            train_shard_path=train_path,
            val_shard_path=val_path,
            train_shard_size_kb=train_size_kb,
            val_shard_size_kb=val_size_kb,
        )


def read_binary_shard(shard_path: str) -> list[int]:
    """Reads a binary uint16 token shard back into a list of Python integers."""
    arr = np.fromfile(shard_path, dtype=np.uint16)
    return arr.tolist()
