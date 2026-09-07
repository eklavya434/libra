"""
Libra Models - Dynamic Key-Value (KV) Cache

Stores past Key and Value projection states across transformer layers to eliminate
redundant computations during autoregressive generation, converting per-token
decoding complexity from O(T) to O(1).
"""

from __future__ import annotations

import torch


class KVCache:
    """Manages Key and Value projection tensor states across transformer layers."""

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers
        self.k_cache: list[torch.Tensor | None] = [None] * n_layers
        self.v_cache: list[torch.Tensor | None] = [None] * n_layers

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new key and value states and returns the cumulative sequence.

        Args:
            key_states: Tensor of shape (B, n_kv_heads, seq_len, head_dim)
            value_states: Tensor of shape (B, n_kv_heads, seq_len, head_dim)
            layer_idx: Layer index in [0, n_layers - 1]

        Returns:
            Tuple of (cumulative_keys, cumulative_values)
        """
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = key_states
            self.v_cache[layer_idx] = value_states
        else:
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], key_states], dim=2)
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], value_states], dim=2)

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def get_seq_len(self, layer_idx: int = 0) -> int:
        """Returns the current cached sequence length for a given layer."""
        if self.k_cache[layer_idx] is None:
            return 0
        return self.k_cache[layer_idx].size(2)

    def reset(self) -> None:
        """Flushes all cached key and value states."""
        self.k_cache = [None] * self.n_layers
        self.v_cache = [None] * self.n_layers

    def memory_bytes(self) -> int:
        """Calculates total memory occupied by cached tensors in bytes."""
        total = 0
        for k, v in zip(self.k_cache, self.v_cache, strict=False):
            if k is not None:
                total += k.element_size() * k.nelement()
            if v is not None:
                total += v.element_size() * v.nelement()
        return total
