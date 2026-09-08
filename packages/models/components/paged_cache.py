"""
Libra Models - Paged KV Cache Architecture
Reference: Kwon et al., 2023 ("vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention")

Implements virtual memory page allocation for LLM inference:
1. PhysicalBlockPool: Fixed-size non-contiguous memory blocks for keys and values.
2. BlockTable: Per-sequence page table mapping logical block indices to physical block IDs.
3. BlockAllocator: Free-list physical block manager with reference counting for Prefix Caching.
4. PagedKVCache: High-level cache manager eliminating internal & external fragmentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import torch


@dataclass
class PhysicalBlockPool:
    """Pre-allocated pool of physical Key and Value memory blocks."""

    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype = torch.float32
    device: str = "cpu"

    key_blocks: torch.Tensor = field(init=False)
    value_blocks: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        shape = (self.num_blocks, self.num_kv_heads, self.block_size, self.head_dim)
        self.key_blocks = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.value_blocks = torch.zeros(shape, dtype=self.dtype, device=self.device)

    def total_memory_bytes(self) -> int:
        return (
            self.key_blocks.element_size() * self.key_blocks.nelement()
            + self.value_blocks.element_size() * self.value_blocks.nelement()
        )


class BlockAllocator:
    """Manages physical block allocation, freeing, and reference counting."""

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self.free_blocks: list[int] = list(range(num_blocks))
        # Reference count per physical block (for prefix caching / copy-on-write)
        self.ref_counts: dict[int, int] = {i: 0 for i in range(num_blocks)}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_allocated_blocks(self) -> int:
        return self.num_blocks - len(self.free_blocks)

    def allocate(self) -> int:
        """Allocates a physical block from the free list."""
        if not self.free_blocks:
            raise MemoryError("Out of physical KV blocks (PhysicalBlockPool exhausted)")
        block_id = self.free_blocks.pop(0)
        self.ref_counts[block_id] = 1
        return block_id

    def free(self, block_id: int) -> None:
        """Decrements reference count and returns block to free list when count hits 0."""
        if block_id not in self.ref_counts or self.ref_counts[block_id] <= 0:
            return
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            self.free_blocks.append(block_id)

    def add_reference(self, block_id: int) -> None:
        """Increments reference count for shared prefix caching."""
        if block_id in self.ref_counts and self.ref_counts[block_id] > 0:
            self.ref_counts[block_id] += 1


@dataclass
class SequenceBlockTable:
    """Logical to physical page table for a single sequence."""

    seq_id: str
    block_size: int
    physical_block_ids: list[int] = field(default_factory=list)
    num_tokens: int = 0

    @property
    def num_logical_blocks(self) -> int:
        if self.num_tokens == 0:
            return 0
        return (self.num_tokens + self.block_size - 1) // self.block_size

    def get_physical_block_id(self, logical_block_idx: int) -> int:
        return self.physical_block_ids[logical_block_idx]

    def get_token_physical_location(self, token_idx: int) -> tuple[int, int]:
        """Resolves token index to (physical_block_id, offset_in_block)."""
        logical_idx = token_idx // self.block_size
        offset = token_idx % self.block_size
        physical_id = self.physical_block_ids[logical_idx]
        return physical_id, offset


class PagedKVCache:
    """Multi-layer Paged KV Cache Manager."""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int = 16,
        num_kv_heads: int = 4,
        head_dim: int = 64,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device

        # Physical memory pool per transformer layer
        self.layer_pools = [
            PhysicalBlockPool(
                num_blocks=num_blocks,
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(num_layers)
        ]

        self.allocator = BlockAllocator(num_blocks=num_blocks)
        self.block_tables: dict[str, SequenceBlockTable] = {}

    def create_sequence(self, seq_id: str) -> SequenceBlockTable:
        """Registers a new sequence in the block table."""
        if seq_id in self.block_tables:
            raise ValueError(f"Sequence {seq_id} already exists in block table")
        table = SequenceBlockTable(seq_id=seq_id, block_size=self.block_size)
        self.block_tables[seq_id] = table
        return table

    def free_sequence(self, seq_id: str) -> None:
        """Frees all physical blocks mapped to a sequence."""
        if seq_id not in self.block_tables:
            return
        table = self.block_tables.pop(seq_id)
        for block_id in table.physical_block_ids:
            self.allocator.free(block_id)

    def append_token_kv(
        self,
        seq_id: str,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Appends a single token Key and Value vector to the paged cache.

        Args:
            seq_id: Sequence identifier
            layer_idx: Transformer layer index
            key: Tensor of shape (num_kv_heads, head_dim)
            value: Tensor of shape (num_kv_heads, head_dim)
        """
        table = self.block_tables[seq_id]
        token_idx = table.num_tokens

        # Check if we need to allocate a new physical block
        if token_idx % self.block_size == 0:
            if layer_idx == 0:
                # Allocate a new physical block for the sequence
                new_block_id = self.allocator.allocate()
                table.physical_block_ids.append(new_block_id)

        logical_idx = token_idx // self.block_size
        physical_id = table.physical_block_ids[logical_idx]
        offset = token_idx % self.block_size

        pool = self.layer_pools[layer_idx]
        pool.key_blocks[physical_id, :, offset, :] = key
        pool.value_blocks[physical_id, :, offset, :] = value

        if layer_idx == self.num_layers - 1:
            table.num_tokens += 1

    def append_sequence_kv(
        self,
        seq_id: str,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Appends multiple tokens (e.g. during prefill) for a given layer.

        Args:
            seq_id: Sequence identifier
            layer_idx: Transformer layer index
            keys: Tensor of shape (num_tokens, num_kv_heads, head_dim)
            values: Tensor of shape (num_tokens, num_kv_heads, head_dim)
        """
        num_toks = keys.size(0)
        table = self.block_tables[seq_id]
        initial_toks = table.num_tokens

        for i in range(num_toks):
            token_idx = initial_toks + i
            if token_idx % self.block_size == 0 and layer_idx == 0:
                new_block_id = self.allocator.allocate()
                table.physical_block_ids.append(new_block_id)

            logical_idx = token_idx // self.block_size
            physical_id = table.physical_block_ids[logical_idx]
            offset = token_idx % self.block_size

            pool = self.layer_pools[layer_idx]
            pool.key_blocks[physical_id, :, offset, :] = keys[i]
            pool.value_blocks[physical_id, :, offset, :] = values[i]

        if layer_idx == self.num_layers - 1:
            table.num_tokens += num_toks

    def get_contiguous_kv(
        self,
        seq_id: str,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gathers non-contiguous paged blocks into a contiguous tensor (for testing/verification).

        Returns:
            Tuple of (keys, values) of shape (num_kv_heads, num_tokens, head_dim)
        """
        table = self.block_tables[seq_id]
        num_toks = table.num_tokens
        if num_toks == 0:
            empty = torch.empty((self.num_kv_heads, 0, self.head_dim), device=self.device)
            return empty, empty

        pool = self.layer_pools[layer_idx]
        gathered_k = []
        gathered_v = []

        for i in range(num_toks):
            phys_id, offset = table.get_token_physical_location(i)
            gathered_k.append(pool.key_blocks[phys_id, :, offset, :])
            gathered_v.append(pool.value_blocks[phys_id, :, offset, :])

        # Stack into (num_kv_heads, num_tokens, head_dim)
        k_tensor = torch.stack(gathered_k, dim=1)
        v_tensor = torch.stack(gathered_v, dim=1)
        return k_tensor, v_tensor

    def memory_metrics(self) -> dict[str, int | float]:
        """Calculates memory metrics, utilization, and fragmentation statistics."""
        total_slots = self.num_blocks * self.block_size
        allocated_blocks = self.allocator.num_allocated_blocks
        allocated_slots = allocated_blocks * self.block_size
        used_slots = sum(t.num_tokens for t in self.block_tables.values())

        internal_fragmentation_slots = allocated_slots - used_slots
        internal_frag_percent = (
            (internal_fragmentation_slots / allocated_slots * 100.0)
            if allocated_slots > 0
            else 0.0
        )

        return {
            "total_blocks": self.num_blocks,
            "allocated_blocks": allocated_blocks,
            "free_blocks": self.allocator.num_free_blocks,
            "block_size": self.block_size,
            "total_capacity_tokens": total_slots,
            "used_tokens": used_slots,
            "internal_frag_tokens": internal_fragmentation_slots,
            "internal_frag_percent": round(internal_frag_percent, 2),
            "memory_bytes_per_layer": self.layer_pools[0].total_memory_bytes(),
            "total_memory_bytes": sum(p.total_memory_bytes() for p in self.layer_pools),
        }
