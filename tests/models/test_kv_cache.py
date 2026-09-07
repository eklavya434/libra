"""
Tests for Key-Value (KV) Cache
Phase 23: Key-Value (KV) Cache Optimization and Grouped-Query Attention (MQA/GQA)
"""

import torch

from packages.models.components.kv_cache import KVCache


def test_kv_cache_initialization():
    cache = KVCache(n_layers=4)
    assert cache.n_layers == 4
    assert cache.get_seq_len() == 0
    assert len(cache.k_cache) == 4
    assert len(cache.v_cache) == 4
    assert all(k is None for k in cache.k_cache)
    assert all(v is None for v in cache.v_cache)


def test_kv_cache_update_prefill_and_decode():
    cache = KVCache(n_layers=2)
    B, n_kv_heads, head_dim = 1, 2, 16

    # Step 1: Prefill (prompt length = 4)
    k_prompt = torch.randn(B, n_kv_heads, 4, head_dim)
    v_prompt = torch.randn(B, n_kv_heads, 4, head_dim)

    k_out_0, v_out_0 = cache.update(k_prompt, v_prompt, layer_idx=0)
    assert k_out_0.shape == (B, n_kv_heads, 4, head_dim)
    assert v_out_0.shape == (B, n_kv_heads, 4, head_dim)
    assert torch.equal(k_out_0, k_prompt)

    # Layer 1 prefill
    cache.update(k_prompt, v_prompt, layer_idx=1)
    assert cache.get_seq_len() == 4

    # Step 2: Single-token decode step (length = 1)
    k_new = torch.randn(B, n_kv_heads, 1, head_dim)
    v_new = torch.randn(B, n_kv_heads, 1, head_dim)

    k_out_decode, v_out_decode = cache.update(k_new, v_new, layer_idx=0)
    assert k_out_decode.shape == (B, n_kv_heads, 5, head_dim)
    assert v_out_decode.shape == (B, n_kv_heads, 5, head_dim)
    # Check that past 4 tokens were preserved and 5th token matches k_new
    assert torch.equal(k_out_decode[:, :, :4, :], k_prompt)
    assert torch.equal(k_out_decode[:, :, 4:5, :], k_new)


def test_kv_cache_memory_and_reset():
    cache = KVCache(n_layers=2)
    k = torch.randn(1, 2, 4, 16, dtype=torch.float32)
    v = torch.randn(1, 2, 4, 16, dtype=torch.float32)

    cache.update(k, v, layer_idx=0)
    cache.update(k, v, layer_idx=1)

    # Memory = 2 layers * 2 (k+v) * (1 * 2 * 4 * 16 elements) * 4 bytes/float32
    expected_bytes = 2 * 2 * (1 * 2 * 4 * 16) * 4
    assert cache.memory_bytes() == expected_bytes
    assert cache.get_seq_len() == 4

    # Reset
    cache.reset()
    assert cache.get_seq_len() == 0
    assert cache.memory_bytes() == 0
    assert cache.k_cache[0] is None
    assert cache.v_cache[0] is None
