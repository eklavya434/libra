"""
Libra Inference - Continuous (Iteration-Level) Batching Engine
Reference: Yu et al., 2022 ("Orca: A Distributed Serving System for Transformer-Based Generative Models")

Interleaves prefill (prompt processing) and decode (single-token generation) iterations,
allowing requests to join and exit dynamically without waiting for the longest sequence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from packages.models.components.paged_cache import PagedKVCache


class SequenceStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class SequenceRequest:
    """Represents an active client inference request."""

    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    generated_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    arrival_time: float = field(default_factory=time.perf_counter)
    start_time: float | None = None
    finish_time: float | None = None

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    @property
    def latency_ms(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.finish_time or time.perf_counter()
        return (end - self.start_time) * 1000.0


@dataclass
class BatchIterationRecord:
    """Telemetry recorded per iteration of the continuous batching loop."""

    iteration_idx: int
    num_running: int
    num_waiting: int
    num_finished: int
    tokens_generated_in_step: int
    allocated_blocks: int
    free_blocks: int
    elapsed_ms: float


class ContinuousBatchingEngine:
    """First-principles continuous iteration-level batching engine."""

    def __init__(
        self,
        paged_cache: PagedKVCache,
        step_generator_fn: Callable[[list[SequenceRequest]], list[int]] | None = None,
    ) -> None:
        self.paged_cache = paged_cache
        self.step_generator_fn = step_generator_fn
        self.waiting_queue: list[SequenceRequest] = []
        self.running_batch: list[SequenceRequest] = []
        self.completed_requests: list[SequenceRequest] = []
        self.iteration_history: list[BatchIterationRecord] = []

    def add_request(
        self,
        request_id: str,
        prompt: str,
        prompt_token_ids: list[int],
        max_new_tokens: int = 16,
    ) -> SequenceRequest:
        """Enqueues a new inference request."""
        req = SequenceRequest(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
        )
        self.waiting_queue.append(req)
        return req

    def can_admit_request(self, req: SequenceRequest) -> bool:
        """Checks if the paged allocator has enough free blocks to admit the prompt."""
        needed_blocks = (
            len(req.prompt_token_ids) + self.paged_cache.block_size - 1
        ) // self.paged_cache.block_size
        return self.paged_cache.allocator.num_free_blocks >= needed_blocks

    def step(self) -> BatchIterationRecord:
        """Executes a single continuous batching iteration.

        1. Admits new waiting requests if blocks are available (Prefill phase).
        2. Advances active running sequences by 1 token (Decode phase).
        3. Evicts finished sequences and frees their physical blocks immediately.
        """
        t0 = time.perf_counter()

        # 1. Admit waiting requests
        admitted = []
        for req in list(self.waiting_queue):
            if self.can_admit_request(req):
                req.status = SequenceStatus.RUNNING
                req.start_time = time.perf_counter()
                self.paged_cache.create_sequence(req.request_id)
                # Allocate initial prompt token slots
                for _ in range(len(req.prompt_token_ids)):
                    table = self.paged_cache.block_tables[req.request_id]
                    if table.num_tokens % self.paged_cache.block_size == 0:
                        bid = self.paged_cache.allocator.allocate()
                        table.physical_block_ids.append(bid)
                    table.num_tokens += 1

                self.running_batch.append(req)
                admitted.append(req)

        for req in admitted:
            self.waiting_queue.remove(req)

        # 2. Generate next tokens
        tokens_generated = 0
        if self.running_batch:
            if self.step_generator_fn is not None:
                next_tokens = self.step_generator_fn(self.running_batch)
            else:
                # Default mock generator producing sequential token IDs
                next_tokens = [100 + req.total_tokens for req in self.running_batch]

            for req, token in zip(self.running_batch, next_tokens, strict=False):
                req.generated_token_ids.append(token)
                tokens_generated += 1

                # Update physical block table for new token
                table = self.paged_cache.block_tables[req.request_id]
                if table.num_tokens % self.paged_cache.block_size == 0:
                    bid = self.paged_cache.allocator.allocate()
                    table.physical_block_ids.append(bid)
                table.num_tokens += 1

                # Check finish conditions
                if len(req.generated_token_ids) >= req.max_new_tokens:
                    req.status = SequenceStatus.FINISHED
                    req.finish_time = time.perf_counter()

        # 3. Evict finished sequences
        finished = [req for req in self.running_batch if req.status == SequenceStatus.FINISHED]
        for req in finished:
            self.running_batch.remove(req)
            self.paged_cache.free_sequence(req.request_id)
            self.completed_requests.append(req)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        record = BatchIterationRecord(
            iteration_idx=len(self.iteration_history),
            num_running=len(self.running_batch),
            num_waiting=len(self.waiting_queue),
            num_finished=len(self.completed_requests),
            tokens_generated_in_step=tokens_generated,
            allocated_blocks=self.paged_cache.allocator.num_allocated_blocks,
            free_blocks=self.paged_cache.allocator.num_free_blocks,
            elapsed_ms=round(elapsed_ms, 3),
        )
        self.iteration_history.append(record)
        return record

    def run_until_complete(self, max_iterations: int = 500) -> list[BatchIterationRecord]:
        """Runs the batching loop until all queued and running requests are finished."""
        while (self.waiting_queue or self.running_batch) and len(
            self.iteration_history
        ) < max_iterations:
            self.step()
        return self.iteration_history
