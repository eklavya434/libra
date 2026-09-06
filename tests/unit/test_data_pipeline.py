"""
Unit tests for Libra Data Pipeline (Quota, Cleaning, Deduplication, and Sharding).
"""

import os

import pytest

from packages.core.tokenizer.educational_bpe import EducationalBPETokenizer
from packages.data.cleaning import clean_text
from packages.data.config import DataPipelineConfig
from packages.data.deduplication import deduplicate_paragraphs
from packages.data.pipeline import DataPipeline, read_binary_shard
from packages.data.quota import (
    MAX_STORAGE_QUOTA_MB,
    QuotaExceededError,
    validate_storage_quota,
)


def test_storage_quota_validation():
    # Safe request within quota
    report = validate_storage_quota(additional_mb=10.0)
    assert report["quota_limit_mb"] == MAX_STORAGE_QUOTA_MB
    assert report["remaining_mb"] > 0

    # Excessive request violating 15 GB quota
    with pytest.raises(QuotaExceededError):
        validate_storage_quota(additional_mb=25000.0)


def test_text_cleaning():
    config = DataPipelineConfig(min_line_length=8)
    dirty_text = (
        "Line 1 with valid characters.\n"
        "   Line 2   with   irregular    spaces.   \n"
        "\x00\x07Hidden control characters in this line.\n"
        "tiny\n"  # should be filtered (< 8 chars)
        "\n\n\n\n"
        "Line 3 after blank lines."
    )

    cleaned, stats = clean_text(dirty_text, config)

    assert "\x00" not in cleaned
    assert "\x07" not in cleaned
    assert "tiny" not in cleaned
    assert "Line 2 with irregular spaces." in cleaned
    assert stats["lines_removed"] >= 1
    assert stats["chars_after"] < stats["chars_before"]


def test_deduplication():
    doc = (
        "Paragraph 1 about artificial intelligence.\n\n"
        "Paragraph 2 about machine learning.\n\n"
        "Paragraph 1 about artificial intelligence.\n\n"  # duplicate
        "paragraph 1 about ARTIFICIAL intelligence.   \n\n"  # case/whitespace duplicate
        "Paragraph 3 about neural networks."
    )

    deduped, stats = deduplicate_paragraphs(doc)

    assert stats["total_paragraphs"] == 5
    assert stats["unique_paragraphs"] == 3
    assert stats["duplicates_removed"] == 2
    assert stats["duplication_rate_pct"] == 40.0
    assert "Paragraph 3 about neural networks." in deduped


def test_pipeline_end_to_end(tmp_path):
    tokenizer = EducationalBPETokenizer()
    tokenizer.train("The quick brown fox jumps over the lazy dog.", num_merges=5)

    config = DataPipelineConfig(
        output_dir=str(tmp_path),
        train_ratio=0.75,
        deduplicate=True,
    )
    pipeline = DataPipeline(config, tokenizer)

    raw_sample = (
        "The quick brown fox jumps over the lazy dog.\n\n"
        "The quick brown fox jumps over the lazy dog.\n\n"  # duplicate
        "Another unique sentence about computer science.\n\n"
        "Final sentence for validation testing."
    )

    report = pipeline.process_text(raw_sample)

    assert report.duplicates_removed == 1
    assert report.train_tokens > 0
    assert report.val_tokens > 0
    assert os.path.exists(report.train_shard_path)
    assert os.path.exists(report.val_shard_path)

    # Verify binary shards can be read back losslessly
    train_tokens = read_binary_shard(report.train_shard_path)
    val_tokens = read_binary_shard(report.val_shard_path)
    assert len(train_tokens) == report.train_tokens
    assert len(val_tokens) == report.val_tokens
