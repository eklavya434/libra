"""
Tests for SequenceBinner & Left-Padding (packages/core/batch/sequence_binner.py)
"""

import torch

from packages.core.batch.sequence_binner import (
    BinningStrategy,
    SequenceBinner,
    calculate_padding_waste,
)


def test_calculate_padding_waste_comparison():
    # Prompt lengths with high variance (e.g. 10, 15, 20, 150, 160)
    lengths = [10, 12, 15, 18, 20, 22, 140, 150, 160]
    stats = calculate_padding_waste(lengths, max_batch_size=4, max_tokens_per_bucket=512)

    assert stats["num_sequences"] == 9
    assert stats["naive"]["padding_tokens"] > stats["binned"]["padding_tokens"]
    assert stats["naive"]["waste_pct"] > stats["binned"]["waste_pct"]
    assert stats["efficiency_gain_pct"] > 0.0
    assert stats["estimated_speedup"] >= 1.0


def test_dynamic_length_binning_groups_similar_lengths():
    # 6 short sequences and 3 long sequences
    seqs = [
        [1, 2, 3],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [10] * 50,
        [11] * 52,
        [12] * 55,
    ]
    binner = SequenceBinner(
        strategy=BinningStrategy.DYNAMIC_LENGTH,
        max_batch_size=4,
        max_tokens_per_bucket=200,
    )
    buckets = binner.bin_sequences(seqs)

    assert len(buckets) >= 2
    # The short sequences should not be batched with the 50+ token sequences
    short_bucket = buckets[0]
    assert short_bucket.max_length <= 10
    assert short_bucket.waste_ratio < 0.5


def test_left_padding_alignment_for_decoder_models():
    seqs = [
        [101, 102],  # len 2
        [101, 102, 103, 104],  # len 4
    ]
    binner = SequenceBinner(strategy=BinningStrategy.DYNAMIC_LENGTH, max_batch_size=2)
    buckets = binner.bin_sequences(seqs)
    bucket = buckets[0]

    padded = binner.pad_bucket(bucket, pad_token_id=0, side="left")
    assert padded.side == "left"
    assert padded.input_ids.shape == (2, 4)

    # First sequence has 2 pad tokens on the left: [0, 0, 101, 102]
    assert torch.equal(padded.input_ids[0], torch.tensor([0, 0, 101, 102]))
    assert torch.equal(padded.attention_mask[0], torch.tensor([0, 0, 1, 1]))
    assert torch.equal(padded.position_ids[0], torch.tensor([0, 0, 0, 1]))

    # Second sequence has no pad tokens: [101, 102, 103, 104]
    assert torch.equal(padded.input_ids[1], torch.tensor([101, 102, 103, 104]))
    assert torch.equal(padded.attention_mask[1], torch.tensor([1, 1, 1, 1]))
    assert torch.equal(padded.position_ids[1], torch.tensor([0, 1, 2, 3]))

    # Both sequences end at the exact same rightmost position index (3)
    assert padded.input_ids[0, -1] == 102
    assert padded.input_ids[1, -1] == 104


def test_fixed_buckets_binning():
    seqs = [
        [1] * 10,  # bucket 0-32
        [2] * 20,  # bucket 0-32
        [3] * 60,  # bucket 33-128
        [4] * 200,  # bucket 129-512
    ]
    binner = SequenceBinner(
        strategy=BinningStrategy.FIXED_BUCKETS,
        max_batch_size=4,
        length_thresholds=[32, 128, 512],
    )
    buckets = binner.bin_sequences(seqs)
    assert len(buckets) == 3
