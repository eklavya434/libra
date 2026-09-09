"""
Libra Core - Asynchronous Worker Queue & Batch Job State Machine
Provides non-blocking job submission, priority scheduling, graceful cancellation,
and real-time progress callbacks for high-throughput LLM workloads.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

from packages.core.batch.sequence_binner import BinningStrategy


class BatchJobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchJobPriority(int, Enum):
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class BatchJobItemResult:
    index: int
    prompt: str
    completion: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    status: str = "success"
    error: str | None = None


@dataclass
class BatchJobTelemetry:
    total_prompts: int = 0
    completed_prompts: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    wall_clock_time_sec: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    padding_waste_pct: float = 0.0
    speedup_factor: float = 1.0
    num_buckets: int = 0


@dataclass
class BatchJob:
    job_id: str
    name: str
    prompts: list[str]
    model_name: str = "libra-llama-tied"
    max_new_tokens: int = 32
    temperature: float = 0.7
    binning_strategy: BinningStrategy = BinningStrategy.DYNAMIC_LENGTH
    priority: BatchJobPriority = BatchJobPriority.NORMAL
    status: BatchJobStatus = BatchJobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    progress: float = 0.0  # 0.0 to 1.0
    current_bucket: int = 0
    total_buckets: int = 0
    results: list[BatchJobItemResult] = field(default_factory=list)
    telemetry: BatchJobTelemetry = field(default_factory=BatchJobTelemetry)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "model_name": self.model_name,
            "num_prompts": len(self.prompts),
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "binning_strategy": self.binning_strategy.value,
            "priority": self.priority.name,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": round(self.progress, 3),
            "current_bucket": self.current_bucket,
            "total_buckets": self.total_buckets,
            "results": [
                {
                    "index": r.index,
                    "prompt": r.prompt,
                    "completion": r.completion,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "latency_ms": round(r.latency_ms, 2),
                    "status": r.status,
                    "error": r.error,
                }
                for r in self.results
            ],
            "telemetry": {
                "total_prompts": self.telemetry.total_prompts,
                "completed_prompts": self.telemetry.completed_prompts,
                "total_prompt_tokens": self.telemetry.total_prompt_tokens,
                "total_completion_tokens": self.telemetry.total_completion_tokens,
                "wall_clock_time_sec": round(self.telemetry.wall_clock_time_sec, 3),
                "throughput_tokens_per_sec": round(self.telemetry.throughput_tokens_per_sec, 2),
                "padding_waste_pct": round(self.telemetry.padding_waste_pct, 2),
                "speedup_factor": round(self.telemetry.speedup_factor, 2),
                "num_buckets": self.telemetry.num_buckets,
            },
            "error_message": self.error_message,
        }


class BatchJobScheduler:
    """
    In-memory asynchronous worker pool managing batch inference job queues,
    concurrency ceilings, cancellation, and subscriber notifications.
    """

    def __init__(self, max_concurrency: int = 1) -> None:
        self.max_concurrency = max_concurrency
        self.jobs: dict[str, BatchJob] = {}
        self._queue: asyncio.PriorityQueue[tuple[int, float, str]] = asyncio.PriorityQueue()
        self._subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}
        self._workers: list[asyncio.Task[None]] = []
        self._running = False
        self._processor_fn: (
            Callable[
                [BatchJob, Callable[[float, int, int], Coroutine[Any, Any, None]]],
                Coroutine[Any, Any, None],
            ]
            | None
        ) = None

    def register_processor(
        self,
        processor_fn: Callable[
            [BatchJob, Callable[[float, int, int], Coroutine[Any, Any, None]]],
            Coroutine[Any, Any, None],
        ],
    ) -> None:
        """Hooks the actual batch generation executor into the scheduler."""
        self._processor_fn = processor_fn

    async def start(self) -> None:
        """Starts worker pool background tasks."""
        if self._running:
            return
        self._running = True
        for i in range(self.max_concurrency):
            task = asyncio.create_task(self._worker_loop(i))
            self._workers.append(task)

    async def stop(self) -> None:
        """Gracefully halts worker tasks."""
        self._running = False
        for task in self._workers:
            task.cancel()
        self._workers.clear()

    async def submit_job(
        self,
        prompts: list[str],
        name: str = "Batch Workload",
        model_name: str = "libra-llama-tied",
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        binning_strategy: BinningStrategy = BinningStrategy.DYNAMIC_LENGTH,
        priority: BatchJobPriority = BatchJobPriority.NORMAL,
    ) -> BatchJob:
        """Submits a new batch job to the priority queue."""
        job_id = f"batch_{uuid.uuid4().hex[:8]}"
        job = BatchJob(
            job_id=job_id,
            name=name,
            prompts=prompts,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            binning_strategy=binning_strategy,
            priority=priority,
            status=BatchJobStatus.QUEUED,
        )
        self.jobs[job_id] = job
        # Enqueue with tuple (priority_int, creation_time, job_id)
        await self._queue.put((priority.value, job.created_at, job_id))
        await self._broadcast_update(job_id, {"event": "queued", "job": job.to_dict()})
        return job

    def get_job(self, job_id: str) -> BatchJob | None:
        return self.jobs.get(job_id)

    def list_jobs(self) -> list[BatchJob]:
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancels a queued or processing job."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        if job.status in [
            BatchJobStatus.COMPLETED,
            BatchJobStatus.FAILED,
            BatchJobStatus.CANCELLED,
        ]:
            return False

        job.status = BatchJobStatus.CANCELLED
        job.finished_at = time.time()
        job.error_message = "Job was cancelled by user."
        await self._broadcast_update(job_id, {"event": "cancelled", "job": job.to_dict()})
        return True

    def subscribe(self, job_id: str) -> asyncio.Queue[dict[str, Any]]:
        """Registers a queue for SSE updates for a specific job."""
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        if job_id not in self._subscribers:
            self._subscribers[job_id] = []
        self._subscribers[job_id].append(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue[dict[str, Any]]) -> None:
        if job_id in self._subscribers and q in self._subscribers[job_id]:
            self._subscribers[job_id].remove(q)
            if not self._subscribers[job_id]:
                del self._subscribers[job_id]

    async def _broadcast_update(self, job_id: str, data: dict[str, Any]) -> None:
        if job_id in self._subscribers:
            for q in list(self._subscribers[job_id]):
                try:
                    q.put_nowait(data)
                except asyncio.QueueFull:
                    pass

    async def _worker_loop(self, worker_id: int) -> None:
        """Worker loop processing jobs from the priority queue."""
        while self._running:
            try:
                _prio, _t, job_id = await self._queue.get()
                job = self.jobs.get(job_id)
                if not job or job.status == BatchJobStatus.CANCELLED:
                    self._queue.task_done()
                    continue

                job.status = BatchJobStatus.PROCESSING
                job.started_at = time.time()
                await self._broadcast_update(job_id, {"event": "started", "job": job.to_dict()})

                async def progress_callback(prog: float, curr_b: int, total_b: int) -> None:
                    job.progress = min(1.0, max(0.0, prog))
                    job.current_bucket = curr_b
                    job.total_buckets = total_b
                    await self._broadcast_update(
                        job_id,
                        {
                            "event": "progress",
                            "progress": round(job.progress, 3),
                            "current_bucket": curr_b,
                            "total_buckets": total_b,
                            "job": job.to_dict(),
                        },
                    )

                if self._processor_fn:
                    try:
                        await self._processor_fn(job, progress_callback)
                        if job.status != BatchJobStatus.CANCELLED:
                            job.status = BatchJobStatus.COMPLETED
                            job.finished_at = time.time()
                            job.progress = 1.0
                    except Exception as e:
                        if job.status != BatchJobStatus.CANCELLED:
                            job.status = BatchJobStatus.FAILED
                            job.error_message = str(e)
                            job.finished_at = time.time()
                else:
                    # Default mock fallback processor if none hooked
                    await asyncio.sleep(0.05)
                    job.status = BatchJobStatus.COMPLETED
                    job.finished_at = time.time()
                    job.progress = 1.0

                await self._broadcast_update(job_id, {"event": "finished", "job": job.to_dict()})
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.1)


# Global singleton scheduler instance
_global_scheduler: BatchJobScheduler | None = None


def get_batch_scheduler(max_concurrency: int = 1) -> BatchJobScheduler:
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = BatchJobScheduler(max_concurrency=max_concurrency)
    return _global_scheduler
