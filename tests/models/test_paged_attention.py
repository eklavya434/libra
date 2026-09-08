"""
Unit Tests for Phase 31: PagedAttention & Block Memory Allocation
"""

import math
import torch
import pytest

from packages.models.components.paged_attention import paged_attention_decode
from packages.models.components.paged_cache import (
    BlockAllocator,
    PagedKVCache,
    PhysicalBlockPool,
    SequenceBlockTable,
)


def test_block_allocator_lifo_free():
    allocator = BlockAllocator(num_blocks=4)
    assert allocator.num_free_blocks == 4
    assert allocator.num_allocated_blocks == 0

    b0 = allocator.allocate()
    b1 = allocator.allocate()
    assert allocator.num_free_blocks == 2
    assert allocator.num_allocated_blocks == 2
    assert b0 == 0 and b1 == 1

    allocator.free(b0)
    assert allocator.num_free_blocks == 3
    assert allocator.num_allocated_blocks == 1


def test_paged_kv_cache_append_and_gather():
    cache = PagedKVCache(
        num_layers=2,
        num_blocks=8,
        block_size=4,
        num_kv_heads=2,
        head_dim=16,
    )

    table = cache.create_sequence("seq-1")
    assert table.num_tokens == 0

    # Append 10 tokens across 2 layers
    keys_layer0 = torch.randn(10, 2, 16)
    values_layer0 = torch.randn(10, 2, 16)

    cache.append_sequence_kv("seq-1", layer_idx=0, keys=keys_layer0, values=values_layer0)
    cache.append_sequence_kv("seq-1", layer_idx=1, keys=keys_layer0, values=values_layer0)

    assert table.num_tokens == 10
    # 10 tokens with block_size=4 requires ceil(10/4) = 3 physical blocks
    assert len(table.physical_block_ids) == 3

    # Gather contiguous tensor and verify exact numerical identity
    gathered_k, gathered_v = cache.get_contiguous_kv("seq-1", layer_idx=0)
    # gathered shape: (num_kv_heads, num_tokens, head_dim)
    expected_k = keys_layer0.transpose(0, 1)
    expected_v = values_layer0.transpose(0, 1)
    assert torch.allclose(gathered_k, expected_k, atol=1e-5)
    assert torch.allclose(gathered_v, expected_v, atol=1e-5)

    metrics = cache.memory_metrics()
    assert metrics["used_tokens"] == 10
    assert metrics["allocated_blocks"] == 3
    assert metrics["internal_frag_tokens"] == (3 * 4) - 10  # 2 wasted slots


def test_paged_attention_decode_numerical_equivalence():
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    block_size = 4
    num_tokens = 9

    pool = PhysicalBlockPool(
        num_blocks=4,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    table = SequenceBlockTable(seq_id="seq-test", block_size=block_size)
    table.physical_block_ids = [0, 1, 2]
    table.num_tokens = num_tokens

    # Populate physical blocks with test vectors
    torch.manual_seed(42)
    pool.key_blocks.normal_()
    pool.value_blocks.normal_()

    query = torch.randn(num_heads, head_dim)

    # 1. Run paged attention kernel
    paged_out = paged_attention_decode(query, pool, table)

    # 2. Compute reference unpaged attention
    gathered_k = []
    gathered_v = []
    for i in range(num_tokens):
        phys_id, off = table.get_token_physical_location(i)
        gathered_k.append(pool.key_blocks[phys_id, :, off, :])
        gathered_v.append(pool.value_blocks[phys_id, :, off, :])

    # Shapes: (num_tokens, num_kv_heads, head_dim) -> (num_heads, num_tokens, head_dim)
    k_ref = torch.stack(gathered_k, dim=0).transpose(0, 1)  # (kv_heads, T, D)
    v_ref = torch.stack(gathered_v, dim=0).transpose(0, 1)

    queries_per_kv = num_heads // num_kv_heads
    k_ref = torch.repeat_interleave(k_ref, repeats=queries_per_kv, dim=0)  # (heads, T, D)
    v_ref = torch.repeat_interleave(v_ref, repeats=queries_per_kv, dim=0)

    # Q @ K^T / sqrt(D)
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.sum(query.unsqueeze(1) * k_ref, dim=-1) * scale  # (heads, T)
    weights = torch.softmax(scores, dim=-1)  # (heads, T)
    ref_out = torch.sum(weights.unsqueeze(-1) * v_ref, dim=1)  # (heads, D)

    # Verify exact match between PagedAttention and standard attention
    assert torch.allclose(paged_out, ref_out, atol=1e-5)
