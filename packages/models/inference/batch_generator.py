"""
Libra Models - High-Throughput Vectorized Batch Generator
Coordinates dynamic sequence binning, left-padding tensor creation,
autoregressive forward stepping, and telemetry calculation.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Coroutine

import torch
import torch.nn as nn

from packages.core.batch.sequence_binner import (
    BinningStrategy,
    SequenceBinner,
    calculate_padding_waste,
)
from packages.core.batch.worker_queue import (
    BatchJob,
    BatchJobItemResult,
    BatchJobTelemetry,
)


class BatchGenerator:
    """
    Executes vectorized batch inference across binned sequence buckets.
    Ensures Left-Padding so decoder autoregressive steps occur uniformly.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        tokenizer: Any | None = None,
        max_batch_size: int = 16,
        max_tokens_per_bucket: int = 1024,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_tokens_per_bucket = max_tokens_per_bucket

    def _tokenize(self, text: str) -> list[int]:
        """Encodes text to token IDs using tokenizer or UTF-8 byte fallback."""
        if self.tokenizer and hasattr(self.tokenizer, "encode"):
            try:
                return self.tokenizer.encode(text)
            except Exception:
                pass
        # Fallback: simple deterministic byte-level encoding
        encoded = list(text.encode("utf-8"))
        return encoded if encoded else [0]

    def _decode(self, token_ids: list[int]) -> str:
        """Decodes token IDs back to string."""
        if self.tokenizer and hasattr(self.tokenizer, "decode"):
            try:
                return self.tokenizer.decode(token_ids)
            except Exception:
                pass
        # Byte decoding fallback
        valid_bytes = bytes([t % 256 for t in token_ids if t > 0])
        return valid_bytes.decode("utf-8", errors="replace")

    async def execute_job(
        self,
        job: BatchJob,
        progress_callback: (Callable[[float, int, int], Coroutine[Any, Any, None]] | None) = None,
    ) -> None:
        """
        Executes a BatchJob end-to-end:
        1. Tokenizes prompts
        2. Bins into length buckets
        3. Generates completions per bucket using left-padded tensors
        4. Reconstructs original sequence ordering
        5. Computes throughput and padding telemetry
        """
        start_time = time.perf_counter()
        prompts = job.prompts
        n_prompts = len(prompts)

        if n_prompts == 0:
            job.telemetry = BatchJobTelemetry()
            return

        # 1. Tokenize all prompts
        tokenized_prompts = [self._tokenize(p) for p in prompts]
        prompt_lengths = [len(seq) for seq in tokenized_prompts]

        # 2. Bin sequences according to strategy
        binner = SequenceBinner(
            strategy=job.binning_strategy,
            max_batch_size=self.max_batch_size,
            max_tokens_per_bucket=self.max_tokens_per_bucket,
        )
        buckets = binner.bin_sequences(tokenized_prompts)
        total_buckets = len(buckets)
        job.total_buckets = total_buckets

        # Calculate theoretical waste comparison
        waste_stats = calculate_padding_waste(
            sequence_lengths=prompt_lengths,
            max_batch_size=self.max_batch_size,
            max_tokens_per_bucket=self.max_tokens_per_bucket,
        )

        # 3. Process each bucket
        ordered_results: list[BatchJobItemResult | None] = [None] * n_prompts
        total_generated_tokens = 0
        total_prompt_tokens = sum(prompt_lengths)

        for b_idx, bucket in enumerate(buckets):
            bucket_start_t = time.perf_counter()
            # Prepare left-padded batch
            padded = binner.pad_bucket(bucket, pad_token_id=0, side="left")
            batch_size = bucket.batch_size

            # Autoregressive generation simulation / execution
            new_tokens_per_seq: list[list[int]] = []
            if self.model is not None and hasattr(self.model, "forward"):
                # Forward pass with real PyTorch model
                generated_ids = self._generate_pytorch_batch(
                    padded=padded,
                    max_new_tokens=job.max_new_tokens,
                    temperature=job.temperature,
                )
                new_tokens_per_seq = generated_ids
            else:
                # Educational mock generator
                # Yields contextual synthetic completions quickly on CPU
                await asyncio.sleep(0.02 * batch_size)
                for row_idx, orig_len in enumerate(padded.sequence_lengths):
                    mock_tokens = [
                        ((orig_len + step + b_idx) % 90) + 33 for step in range(job.max_new_tokens)
                    ]
                    new_tokens_per_seq.append(mock_tokens)

            bucket_elapsed_ms = (time.perf_counter() - bucket_start_t) * 1000.0
            per_item_ms = bucket_elapsed_ms / max(1, batch_size)

            # Map results back to original indices
            for item_idx, orig_idx in enumerate(bucket.prompt_indices):
                gen_ids = new_tokens_per_seq[item_idx]
                total_generated_tokens += len(gen_ids)
                comp_text = (
                    self._decode(gen_ids)
                    if self.model is not None
                    else f"Batch response for prompt #{orig_idx + 1}: processed via bucket {b_idx + 1}/{total_buckets}."
                )
                p_len = padded.sequence_lengths[item_idx]

                ordered_results[orig_idx] = BatchJobItemResult(
                    index=orig_idx,
                    prompt=prompts[orig_idx],
                    completion=comp_text,
                    prompt_tokens=p_len,
                    completion_tokens=len(gen_ids),
                    latency_ms=per_item_ms,
                    status="success",
                )

            # Notify progress
            if progress_callback:
                prog = (b_idx + 1) / total_buckets
                await progress_callback(prog, b_idx + 1, total_buckets)

        # 4. Finalize job results
        valid_results = [r for r in ordered_results if r is not None]
        job.results = valid_results

        # 5. Telemetry
        wall_time = max(0.001, time.perf_counter() - start_time)
        total_tokens = total_prompt_tokens + total_generated_tokens
        throughput = total_tokens / wall_time

        job.telemetry = BatchJobTelemetry(
            total_prompts=n_prompts,
            completed_prompts=len(valid_results),
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_generated_tokens,
            wall_clock_time_sec=wall_time,
            throughput_tokens_per_sec=throughput,
            padding_waste_pct=waste_stats["binned"]["waste_pct"]
            if job.binning_strategy != BinningStrategy.NAIVE
            else waste_stats["naive"]["waste_pct"],
            speedup_factor=waste_stats["estimated_speedup"],
            num_buckets=total_buckets,
        )

    def _generate_pytorch_batch(
        self,
        padded: Any,
        max_new_tokens: int,
        temperature: float,
    ) -> list[list[int]]:
        """Vectorized generation with real PyTorch model."""
        input_ids = padded.input_ids.clone()
        generated: list[list[int]] = [[] for _ in range(input_ids.size(0))]

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.model(input_ids)
                # Next token logits are at the rightmost position
                next_token_logits = logits[:, -1, :]
                if temperature > 0:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                input_ids = torch.cat([input_ids, next_tokens], dim=1)
                for b_i in range(input_ids.size(0)):
                    generated[b_i].append(int(next_tokens[b_i, 0].item()))

        return generated
