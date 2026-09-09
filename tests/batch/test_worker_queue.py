"""
Tests for BatchJobScheduler & Async Worker Queue (packages/core/batch/worker_queue.py)
"""

import asyncio

import pytest

from packages.core.batch.worker_queue import (
    BatchJobPriority,
    BatchJobScheduler,
    BatchJobStatus,
)


@pytest.mark.asyncio
async def test_scheduler_lifecycle_and_execution():
    scheduler = BatchJobScheduler(max_concurrency=1)

    # Register a fast mock processor
    async def mock_processor(job, progress_cb):
        await progress_cb(0.5, 1, 2)
        await asyncio.sleep(0.01)
        await progress_cb(1.0, 2, 2)

    scheduler.register_processor(mock_processor)
    await scheduler.start()

    job = await scheduler.submit_job(
        prompts=["Prompt 1", "Prompt 2"],
        name="Test Lifecycle",
        priority=BatchJobPriority.NORMAL,
    )
    assert job.status in [BatchJobStatus.QUEUED, BatchJobStatus.PROCESSING]

    # Wait for completion
    for _ in range(50):
        if job.status == BatchJobStatus.COMPLETED:
            break
        await asyncio.sleep(0.02)

    assert job.status == BatchJobStatus.COMPLETED
    assert job.progress == 1.0
    assert job.finished_at is not None

    await scheduler.stop()


@pytest.mark.asyncio
async def test_scheduler_job_cancellation():
    scheduler = BatchJobScheduler(max_concurrency=1)

    # Processor that sleeps
    async def slow_processor(job, progress_cb):
        await asyncio.sleep(1.0)

    scheduler.register_processor(slow_processor)
    await scheduler.start()

    job = await scheduler.submit_job(
        prompts=["Slow 1", "Slow 2"],
        name="Cancellation Test",
    )

    await asyncio.sleep(0.02)
    success = await scheduler.cancel_job(job.job_id)
    assert success is True
    assert job.status == BatchJobStatus.CANCELLED
    assert job.error_message == "Job was cancelled by user."

    await scheduler.stop()


@pytest.mark.asyncio
async def test_scheduler_priority_dispatch():
    # Don't start workers yet, queue jobs in reverse priority
    scheduler = BatchJobScheduler(max_concurrency=1)
    execution_order = []

    async def tracking_processor(job, progress_cb):
        execution_order.append(job.name)

    scheduler.register_processor(tracking_processor)

    # Queue LOW priority then HIGH priority
    job_low = await scheduler.submit_job(
        prompts=["Low"],
        name="LOW_JOB",
        priority=BatchJobPriority.LOW,
    )
    job_high = await scheduler.submit_job(
        prompts=["High"],
        name="HIGH_JOB",
        priority=BatchJobPriority.HIGH,
    )

    # Now start workers
    await scheduler.start()

    for _ in range(50):
        if len(execution_order) == 2:
            break
        await asyncio.sleep(0.02)

    # HIGH priority should execute before LOW priority
    assert execution_order == ["HIGH_JOB", "LOW_JOB"]

    await scheduler.stop()
