"""
Tests for BatchGenerator (packages/models/inference/batch_generator.py)
"""

import pytest

from packages.core.batch.sequence_binner import BinningStrategy
from packages.core.batch.worker_queue import BatchJob
from packages.models.inference.batch_generator import BatchGenerator


@pytest.mark.asyncio
async def test_batch_generator_preserves_prompt_order():
    generator = BatchGenerator(max_batch_size=2)

    prompts = [
        "Tiny",
        "A somewhat longer prompt with more words and detail",
        "Medium prompt here",
        "Short",
    ]

    job = BatchJob(
        job_id="test_gen_order",
        name="Order Verification",
        prompts=prompts,
        max_new_tokens=8,
        binning_strategy=BinningStrategy.DYNAMIC_LENGTH,
    )

    progress_steps = []

    async def cb(prog, curr_b, total_b):
        progress_steps.append((prog, curr_b, total_b))

    await generator.execute_job(job, cb)

    assert len(job.results) == 4
    # Ensure items are in exact prompt order 0, 1, 2, 3
    for i, res in enumerate(job.results):
        assert res.index == i
        assert res.prompt == prompts[i]
        assert len(res.completion) > 0
        assert res.completion_tokens == 8

    assert job.telemetry.total_prompts == 4
    assert job.telemetry.total_completion_tokens == 32
    assert job.telemetry.throughput_tokens_per_sec > 0.0
    assert len(progress_steps) > 0
