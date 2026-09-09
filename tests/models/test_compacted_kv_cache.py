"""
Unit tests for CompactedKVCache (H2O Heavy-Hitter Oracle & StreamingLLM attention sinks).
"""

import pytest
import torch

from packages.models.components.compacted_kv_cache import CompactedKVCache


def test_compacted_kv_cache_initialization_and_validation():
    # Valid
    cache = CompactedKVCache(n_layers=2, max_budget=128, n_sink=4, n_recent=32)
    assert cache.max_budget == 128
    assert cache.n_sink == 4
    assert cache.n_recent == 32
    assert cache.n_heavy == 128 - 4 - 32

    # Invalid: budget smaller than sink + recent
    with pytest.raises(ValueError):
        CompactedKVCache(n_layers=2, max_budget=30, n_sink=10, n_recent=25)


def test_compacted_kv_cache_under_budget():
    cache = CompactedKVCache(n_layers=2, max_budget=64, n_sink=4, n_recent=16)

    k1 = torch.randn(1, 2, 20, 32)
    v1 = torch.randn(1, 2, 20, 32)
    k_out, v_out = cache.update(k1, v1, layer_idx=0)

    assert k_out.size(2) == 20
    assert v_out.size(2) == 20
    assert cache.get_seq_len(0) == 20
    assert cache.total_evicted_tokens == 0


def test_compacted_kv_cache_compaction_and_retention():
    max_budget = 40
    n_sink = 4
    n_recent = 16
    cache = CompactedKVCache(
        n_layers=1,
        max_budget=max_budget,
        n_sink=n_sink,
        n_recent=n_recent,
        n_heavy=10,
    )

    # Add 60 tokens total (exceeding budget of 40)
    k1 = torch.randn(1, 2, 60, 16)
    v1 = torch.randn(1, 2, 60, 16)

    # Provide synthetic attention weights: make position 10 have massive attention score
    synthetic_attn = torch.zeros(1, 2, 60, 60)
    synthetic_attn[:, :, :, 10] = 50.0  # token 10 is a heavy hitter

    k_out, v_out = cache.update(k1, v1, layer_idx=0, current_attn_weights=synthetic_attn)

    # Cache should be compacted down to <= max_budget
    assert k_out.size(2) <= max_budget
    assert v_out.size(2) <= max_budget
    assert cache.get_seq_len(0) <= max_budget
    assert cache.total_evicted_tokens > 0

    stats = cache.get_stats()
    assert stats["cached_tokens"] <= max_budget
    assert stats["total_tokens_seen"] == 60
    assert stats["evicted_tokens"] > 0
    assert stats["memory_savings_pct"] > 0.0


def test_compacted_kv_cache_reset_and_memory():
    cache = CompactedKVCache(n_layers=2, max_budget=32, n_sink=2, n_recent=8)
    k = torch.randn(1, 2, 10, 16)
    v = torch.randn(1, 2, 10, 16)
    cache.update(k, v, layer_idx=0)
    cache.update(k, v, layer_idx=1)

    assert cache.memory_bytes() > 0

    cache.reset()
    assert cache.get_seq_len(0) == 0
    assert cache.get_seq_len(1) == 0
    assert cache.memory_bytes() == 0
    assert cache.total_tokens_seen == 0
