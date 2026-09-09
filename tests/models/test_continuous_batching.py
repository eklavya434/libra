"""
Unit Tests for Phase 31: Continuous (Iteration-Level) Batching Engine
"""

from packages.models.components.paged_cache import PagedKVCache
from packages.models.inference.continuous_batching import (
    ContinuousBatchingEngine,
    SequenceStatus,
)


def test_continuous_batching_interleaved_lifecycle():
    cache = PagedKVCache(
        num_layers=2,
        num_blocks=16,
        block_size=4,
        num_kv_heads=2,
        head_dim=16,
    )
    engine = ContinuousBatchingEngine(paged_cache=cache)

    # Add two requests with different lengths
    req1 = engine.add_request(
        "req-1", prompt="Short prompt", prompt_token_ids=[1, 2], max_new_tokens=3
    )
    req2 = engine.add_request(
        "req-2", prompt="Long prompt example", prompt_token_ids=[1, 2, 3, 4], max_new_tokens=6
    )

    assert len(engine.waiting_queue) == 2
    assert len(engine.running_batch) == 0

    # Step 1: Both admitted
    engine.step()
    assert len(engine.running_batch) == 2
    assert req1.status == SequenceStatus.RUNNING
    assert req2.status == SequenceStatus.RUNNING
    assert len(req1.generated_token_ids) == 1
    assert len(req2.generated_token_ids) == 1

    # Step 2:
    engine.step()
    assert len(req1.generated_token_ids) == 2
    assert len(req2.generated_token_ids) == 2

    # Step 3: req1 reaches max_new_tokens (3) and finishes, req2 continues
    engine.step()
    assert len(req1.generated_token_ids) == 3
    assert req1.status == SequenceStatus.FINISHED
    assert req1 in engine.completed_requests
    assert req1 not in engine.running_batch
    assert req2 in engine.running_batch

    # Run remaining steps to finish req2
    timeline = engine.run_until_complete()
    assert req2.status == SequenceStatus.FINISHED
    assert len(engine.completed_requests) == 2
    assert len(engine.running_batch) == 0
    assert len(engine.waiting_queue) == 0
    # All physical blocks should be freed back to pool
    assert cache.allocator.num_allocated_blocks == 0
    assert cache.allocator.num_free_blocks == 16
