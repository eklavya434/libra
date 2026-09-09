"""
Libra Models - PagedAttention Kernel
Reference: Kwon et al., 2023 ("vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention")

Computes multi-head attention directly from non-contiguous physical KV memory blocks
using sequence block tables, avoiding tensor materialization and eliminating fragmentation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from packages.models.components.paged_cache import PhysicalBlockPool, SequenceBlockTable


def paged_attention_decode(
    query: torch.Tensor,
    pool: PhysicalBlockPool,
    block_table: SequenceBlockTable,
    scale: float | None = None,
) -> torch.Tensor:
    """Computes single-token decoding attention across paged KV blocks.

    Args:
        query: Query tensor of shape (num_heads, head_dim) or (1, num_heads, 1, head_dim)
        pool: PhysicalBlockPool containing non-contiguous physical blocks
        block_table: SequenceBlockTable mapping logical blocks to physical IDs
        scale: Optional attention scale factor (default: 1 / sqrt(head_dim))

    Returns:
        Output tensor of shape (num_heads, head_dim)
    """
    # Normalize query shape to (num_heads, head_dim)
    if query.ndim == 4:
        query = query.squeeze(0).squeeze(1)  # (num_heads, head_dim)
    elif query.ndim == 3:
        query = query.squeeze(0)

    num_heads, head_dim = query.shape
    num_kv_heads = pool.num_kv_heads
    num_tokens = block_table.num_tokens
    block_size = block_table.block_size
    queries_per_kv = num_heads // num_kv_heads

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    if num_tokens == 0:
        return torch.zeros((num_heads, head_dim), dtype=query.dtype, device=query.device)

    # 1. Compute dot-product attention scores across physical blocks
    # scores shape: (num_heads, num_tokens)
    scores = torch.empty((num_heads, num_tokens), dtype=query.dtype, device=query.device)

    for tok_idx in range(num_tokens):
        phys_block_id, offset = block_table.get_token_physical_location(tok_idx)
        # Key vector: (num_kv_heads, head_dim)
        k_vec = pool.key_blocks[phys_block_id, :, offset, :]

        # GQA expansion if num_heads > num_kv_heads
        if queries_per_kv > 1:
            k_vec = torch.repeat_interleave(k_vec, repeats=queries_per_kv, dim=0)

        # Dot product with query heads: (num_heads,)
        head_scores = torch.sum(query * k_vec, dim=-1) * scale
        scores[:, tok_idx] = head_scores

    # 2. Softmax across sequence tokens
    attn_weights = F.softmax(scores, dim=-1)  # (num_heads, num_tokens)

    # 3. Weighted aggregation of value vectors
    output = torch.zeros((num_heads, head_dim), dtype=query.dtype, device=query.device)

    for tok_idx in range(num_tokens):
        phys_block_id, offset = block_table.get_token_physical_location(tok_idx)
        v_vec = pool.value_blocks[phys_block_id, :, offset, :]

        if queries_per_kv > 1:
            v_vec = torch.repeat_interleave(v_vec, repeats=queries_per_kv, dim=0)

        # Weight vector for current token across heads: (num_heads, 1)
        w = attn_weights[:, tok_idx].unsqueeze(-1)
        output += w * v_vec

    return output
