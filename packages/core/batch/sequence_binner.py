"""
Libra Core - Dynamic Sequence Binning & Left-Padding Engine
Reference: "Effective batching strategies for Transformer inference with variable sequence lengths"

Autoregressive decoder models suffer quadratic attention and memory waste when variable-length
prompts are padded naively. SequenceBinner groups sequences into length-coherent buckets and applies
left-padding so that next-token generation occurs uniformly at the rightmost token boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


class BinningStrategy(str, Enum):
    NAIVE = "naive"
    DYNAMIC_LENGTH = "dynamic_length"
    FIXED_BUCKETS = "fixed_buckets"


@dataclass
class BatchBucket:
    """Represents a single binned batch of sequences sharing a coherent length."""

    bucket_id: int
    prompt_indices: list[int]
    sequences: list[list[int]]
    max_length: int
    useful_tokens: int
    total_tokens: int
    padding_tokens: int
    waste_ratio: float

    @property
    def batch_size(self) -> int:
        return len(self.sequences)


@dataclass
class PaddedBatch:
    """A vectorized batch prepared with left-padding for decoder autoregressive generation."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    original_indices: list[int]
    sequence_lengths: list[int]
    pad_token_id: int
    side: str = "left"


def calculate_padding_waste(
    sequence_lengths: list[int],
    max_batch_size: int = 16,
    max_tokens_per_bucket: int = 1024,
) -> dict[str, Any]:
    """
    Computes comparative FLOP and memory metrics between naive uniform batching
    and dynamic length-sorted sequence binning.
    """
    if not sequence_lengths:
        return {
            "num_sequences": 0,
            "naive": {
                "total_tokens": 0,
                "padding_tokens": 0,
                "waste_pct": 0.0,
                "quadratic_waste_pct": 0.0,
            },
            "binned": {
                "total_tokens": 0,
                "padding_tokens": 0,
                "waste_pct": 0.0,
                "quadratic_waste_pct": 0.0,
            },
            "efficiency_gain_pct": 0.0,
            "estimated_speedup": 1.0,
        }

    n = len(sequence_lengths)
    sum_lengths = sum(sequence_lengths)
    max_len = max(sequence_lengths)

    # 1. Naive Batching (All sequences in chunks of max_batch_size, padded to chunk max_len)
    naive_total_tokens = 0
    naive_quadratic_cost = 0
    for i in range(0, n, max_batch_size):
        chunk = sequence_lengths[i : i + max_batch_size]
        chunk_max = max(chunk)
        naive_total_tokens += len(chunk) * chunk_max
        naive_quadratic_cost += len(chunk) * (chunk_max**2)

    naive_padding_tokens = naive_total_tokens - sum_lengths
    naive_waste_pct = (
        (naive_padding_tokens / naive_total_tokens * 100.0) if naive_total_tokens > 0 else 0.0
    )
    useful_quadratic_cost = sum(l**2 for l in sequence_lengths)
    naive_quad_waste_pct = (
        ((naive_quadratic_cost - useful_quadratic_cost) / naive_quadratic_cost * 100.0)
        if naive_quadratic_cost > 0
        else 0.0
    )

    # 2. Dynamic Length Binning (Sorted sequences partitioned into compact buckets)
    binner = SequenceBinner(
        strategy=BinningStrategy.DYNAMIC_LENGTH,
        max_batch_size=max_batch_size,
        max_tokens_per_bucket=max_tokens_per_bucket,
    )
    # Mock dummy token lists
    dummy_seqs = [[0] * l for l in sequence_lengths]
    buckets = binner.bin_sequences(dummy_seqs)

    binned_total_tokens = sum(b.total_tokens for b in buckets)
    binned_padding_tokens = sum(b.padding_tokens for b in buckets)
    binned_waste_pct = (
        (binned_padding_tokens / binned_total_tokens * 100.0) if binned_total_tokens > 0 else 0.0
    )

    binned_quadratic_cost = sum(b.batch_size * (b.max_length**2) for b in buckets)
    binned_quad_waste_pct = (
        ((binned_quadratic_cost - useful_quadratic_cost) / binned_quadratic_cost * 100.0)
        if binned_quadratic_cost > 0
        else 0.0
    )

    efficiency_gain_pct = max(0.0, naive_waste_pct - binned_waste_pct)
    estimated_speedup = (
        (naive_quadratic_cost / max(1, binned_quadratic_cost)) if binned_quadratic_cost > 0 else 1.0
    )

    return {
        "num_sequences": n,
        "max_sequence_length": max_len,
        "average_sequence_length": round(sum_lengths / n, 2),
        "naive": {
            "total_tokens": naive_total_tokens,
            "useful_tokens": sum_lengths,
            "padding_tokens": naive_padding_tokens,
            "waste_pct": round(naive_waste_pct, 2),
            "quadratic_waste_pct": round(naive_quad_waste_pct, 2),
        },
        "binned": {
            "total_tokens": binned_total_tokens,
            "useful_tokens": sum_lengths,
            "padding_tokens": binned_padding_tokens,
            "waste_pct": round(binned_waste_pct, 2),
            "quadratic_waste_pct": round(binned_quad_waste_pct, 2),
            "num_buckets": len(buckets),
        },
        "efficiency_gain_pct": round(efficiency_gain_pct, 2),
        "estimated_speedup": round(estimated_speedup, 2),
    }


class SequenceBinner:
    """
    Partitions variable-length sequences into uniform or dynamic buckets
    and vectors them with left-padding for autoregressive LLMs.
    """

    def __init__(
        self,
        strategy: BinningStrategy = BinningStrategy.DYNAMIC_LENGTH,
        max_batch_size: int = 16,
        max_tokens_per_bucket: int = 1024,
        length_thresholds: list[int] | None = None,
    ) -> None:
        self.strategy = strategy
        self.max_batch_size = max(1, max_batch_size)
        self.max_tokens_per_bucket = max(64, max_tokens_per_bucket)
        self.length_thresholds = sorted(length_thresholds or [32, 128, 512, 1024])

    def bin_sequences(self, sequences: list[list[int]]) -> list[BatchBucket]:
        """Partitions input sequences into an optimal list of BatchBucket objects."""
        if not sequences:
            return []

        if self.strategy == BinningStrategy.NAIVE:
            return self._bin_naive(sequences)
        elif self.strategy == BinningStrategy.FIXED_BUCKETS:
            return self._bin_fixed_thresholds(sequences)
        else:
            return self._bin_dynamic_length(sequences)

    def _bin_naive(self, sequences: list[list[int]]) -> list[BatchBucket]:
        """Simple chunking without length sorting (simulates standard naive batching)."""
        buckets: list[BatchBucket] = []
        n = len(sequences)

        for i in range(0, n, self.max_batch_size):
            chunk = sequences[i : i + self.max_batch_size]
            indices = list(range(i, i + len(chunk)))
            max_len = max(len(s) for s in chunk) if chunk else 0
            useful = sum(len(s) for s in chunk)
            total = len(chunk) * max_len
            pad = total - useful
            waste = (pad / total) if total > 0 else 0.0

            buckets.append(
                BatchBucket(
                    bucket_id=len(buckets),
                    prompt_indices=indices,
                    sequences=chunk,
                    max_length=max_len,
                    useful_tokens=useful,
                    total_tokens=total,
                    padding_tokens=pad,
                    waste_ratio=waste,
                )
            )
        return buckets

    def _bin_dynamic_length(self, sequences: list[list[int]]) -> list[BatchBucket]:
        """
        Sorts sequences by length and greedily clusters them into buckets
        bounded by max_batch_size and max_tokens_per_bucket.
        """
        # Sort indices by sequence length
        indexed_seqs = sorted(enumerate(sequences), key=lambda x: len(x[1]))

        buckets: list[BatchBucket] = []
        current_indices: list[int] = []
        current_seqs: list[list[int]] = []

        for orig_idx, seq in indexed_seqs:
            seq_len = len(seq)
            new_max_len = max([len(s) for s in current_seqs] + [seq_len])
            candidate_size = len(current_seqs) + 1
            candidate_total_tokens = candidate_size * new_max_len

            # Check if adding this sequence exceeds budget or length disparity threshold
            min_l = len(current_seqs[0]) if current_seqs else seq_len
            ratio_exceeded = (min_l > 0 and (seq_len / min_l) > 2.5) or (
                min_l == 0 and seq_len > 10
            )

            if current_seqs and (
                candidate_size > self.max_batch_size
                or candidate_total_tokens > self.max_tokens_per_bucket
                or ratio_exceeded
            ):
                # Finalize current bucket
                max_l = max(len(s) for s in current_seqs)
                useful = sum(len(s) for s in current_seqs)
                total = len(current_seqs) * max_l
                pad = total - useful
                buckets.append(
                    BatchBucket(
                        bucket_id=len(buckets),
                        prompt_indices=current_indices,
                        sequences=current_seqs,
                        max_length=max_l,
                        useful_tokens=useful,
                        total_tokens=total,
                        padding_tokens=pad,
                        waste_ratio=(pad / total) if total > 0 else 0.0,
                    )
                )
                current_indices = []
                current_seqs = []

            current_indices.append(orig_idx)
            current_seqs.append(seq)

        if current_seqs:
            max_l = max(len(s) for s in current_seqs)
            useful = sum(len(s) for s in current_seqs)
            total = len(current_seqs) * max_l
            pad = total - useful
            buckets.append(
                BatchBucket(
                    bucket_id=len(buckets),
                    prompt_indices=current_indices,
                    sequences=current_seqs,
                    max_length=max_l,
                    useful_tokens=useful,
                    total_tokens=total,
                    padding_tokens=pad,
                    waste_ratio=(pad / total) if total > 0 else 0.0,
                )
            )

        return buckets

    def _bin_fixed_thresholds(self, sequences: list[list[int]]) -> list[BatchBucket]:
        """Bins sequences into predetermined threshold intervals (e.g. 0-32, 33-128, etc.)."""
        intervals: list[tuple[int, int]] = []
        prev = 0
        for thresh in self.length_thresholds:
            intervals.append((prev, thresh))
            prev = thresh + 1
        intervals.append((prev, 100_000))  # Catch-all

        grouped: dict[int, list[tuple[int, list[int]]]] = {i: [] for i in range(len(intervals))}
        for orig_idx, seq in enumerate(sequences):
            seq_len = len(seq)
            placed = False
            for group_idx, (low, high) in enumerate(intervals):
                if low <= seq_len <= high:
                    grouped[group_idx].append((orig_idx, seq))
                    placed = True
                    break
            if not placed:
                grouped[len(intervals) - 1].append((orig_idx, seq))

        buckets: list[BatchBucket] = []
        for group_idx in sorted(grouped.keys()):
            items = grouped[group_idx]
            if not items:
                continue
            # Chunk items within the same threshold by max_batch_size
            for i in range(0, len(items), self.max_batch_size):
                chunk = items[i : i + self.max_batch_size]
                chunk_indices = [c[0] for c in chunk]
                chunk_seqs = [c[1] for c in chunk]
                max_l = max(len(s) for s in chunk_seqs)
                useful = sum(len(s) for s in chunk_seqs)
                total = len(chunk_seqs) * max_l
                pad = total - useful
                buckets.append(
                    BatchBucket(
                        bucket_id=len(buckets),
                        prompt_indices=chunk_indices,
                        sequences=chunk_seqs,
                        max_length=max_l,
                        useful_tokens=useful,
                        total_tokens=total,
                        padding_tokens=pad,
                        waste_ratio=(pad / total) if total > 0 else 0.0,
                    )
                )

        return buckets

    def pad_bucket(
        self,
        bucket: BatchBucket,
        pad_token_id: int = 0,
        side: str = "left",
    ) -> PaddedBatch:
        """
        Pads sequences in a bucket to bucket.max_length.
        Decoder-only generation requires 'left' padding so rightmost token aligns across batch.
        """
        b_size = bucket.batch_size
        max_l = max(1, bucket.max_length)

        input_ids = torch.full((b_size, max_l), fill_value=pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((b_size, max_l), dtype=torch.long)
        position_ids = torch.zeros((b_size, max_l), dtype=torch.long)
        lengths: list[int] = []

        for row, seq in enumerate(bucket.sequences):
            l = len(seq)
            lengths.append(l)
            if l == 0:
                continue

            seq_tensor = torch.tensor(seq, dtype=torch.long)
            if side == "left":
                # Left padding: pad at [0, max_l - l), real tokens at [max_l - l, max_l)
                start_col = max_l - l
                input_ids[row, start_col:] = seq_tensor
                attention_mask[row, start_col:] = 1
                # Position IDs for real tokens start from 0 to l - 1
                position_ids[row, start_col:] = torch.arange(l, dtype=torch.long)
            else:
                # Right padding (standard for encoder models, NOT autoregressive decoders)
                input_ids[row, :l] = seq_tensor
                attention_mask[row, :l] = 1
                position_ids[row, :l] = torch.arange(l, dtype=torch.long)

        return PaddedBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            original_indices=bucket.prompt_indices,
            sequence_lengths=lengths,
            pad_token_id=pad_token_id,
            side=side,
        )
