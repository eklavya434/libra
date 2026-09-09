"""
Libra Models - Dynamic Compacted KV Cache (H2O & StreamingLLM Eviction)

Implements dynamic KV cache compaction engineered from first principles:
1. Attention Sinks (StreamingLLM): Preserves first N_sink tokens to anchor attention normalization.
2. Rolling Recent Window: Preserves trailing N_recent tokens for local grammar and syntax coherence.
3. Heavy-Hitter Oracle (H2O): Retains top N_heavy historically significant tokens based on
   accumulated attention scores, evicting transient noise tokens to bound memory to O(B).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


class CompactedKVCache:
    """
    Budget-constrained Key-Value Cache implementing Heavy-Hitter Oracle (H2O)
    and StreamingLLM attention sinks for bounded-memory autoregressive decoding.
    """

    def __init__(
        self,
        n_layers: int,
        max_budget: int = 512,
        n_sink: int = 4,
        n_recent: int = 64,
        n_heavy: Optional[int] = None,
    ) -> None:
        if max_budget < n_sink + n_recent:
            raise ValueError(
                f"max_budget ({max_budget}) must be >= n_sink ({n_sink}) + n_recent ({n_recent})"
            )

        self.n_layers = n_layers
        self.max_budget = max_budget
        self.n_sink = n_sink
        self.n_recent = n_recent
        self.n_heavy = n_heavy if n_heavy is not None else (max_budget - n_sink - n_recent)

        self.k_cache: List[Optional[torch.Tensor]] = [None] * n_layers
        self.v_cache: List[Optional[torch.Tensor]] = [None] * n_layers
        # Cumulative attention scores per token position: List[Optional[Tensor (B, seq_len)]]
        self.attn_scores: List[Optional[torch.Tensor]] = [None] * n_layers

        self.total_tokens_seen: int = 0
        self.total_evicted_tokens: int = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        current_attn_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Appends new key and value projection states, updates accumulated attention scores,
        and compacts cache when sequence length exceeds max_budget.

        Args:
            key_states: Tensor of shape (B, n_kv_heads, new_seq_len, head_dim)
            value_states: Tensor of shape (B, n_kv_heads, new_seq_len, head_dim)
            layer_idx: Layer index in [0, n_layers - 1]
            current_attn_weights: Optional attention weights (B, n_heads, query_len, total_len)

        Returns:
            Tuple of (compacted_keys, compacted_values)
        """
        B, n_kv_heads, new_len, head_dim = key_states.shape

        # 1. Concatenate into cache
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = key_states
            self.v_cache[layer_idx] = value_states
            self.attn_scores[layer_idx] = torch.zeros(
                (B, new_len), dtype=torch.float32, device=key_states.device
            )
            if layer_idx == 0:
                self.total_tokens_seen = new_len
        else:
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], key_states], dim=2)
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], value_states], dim=2)

            new_scores = torch.zeros((B, new_len), dtype=torch.float32, device=key_states.device)
            self.attn_scores[layer_idx] = torch.cat(
                [self.attn_scores[layer_idx], new_scores], dim=1
            )
            if layer_idx == 0:
                self.total_tokens_seen += new_len

        # 2. Accumulate attention scores if provided
        if current_attn_weights is not None:
            # Aggregate across query positions and attention heads -> (B, total_len)
            # current_attn_weights: (B, heads, q_len, total_len)
            summed = current_attn_weights.sum(dim=(1, 2))  # (B, total_len)
            total_len = self.attn_scores[layer_idx].shape[1]
            if summed.shape[1] == total_len:
                self.attn_scores[layer_idx] = self.attn_scores[layer_idx] + summed

        # 3. Compact if exceeds max_budget
        current_len = self.k_cache[layer_idx].size(2)
        if current_len > self.max_budget:
            self._compact_layer(layer_idx)

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def _compact_layer(self, layer_idx: int) -> None:
        """Evicts low-importance middle tokens while retaining sinks, recent window, and heavy hitters."""
        k = self.k_cache[layer_idx]
        v = self.v_cache[layer_idx]
        scores = self.attn_scores[layer_idx]
        if k is None or v is None:
            return

        total_len = k.size(2)
        if total_len <= self.max_budget:
            return

        # Sinks: [0 : n_sink]
        sink_indices = list(range(self.n_sink))

        # Recent: [total_len - n_recent : total_len]
        recent_indices = list(range(total_len - self.n_recent, total_len))

        # Candidate middle positions
        middle_start = self.n_sink
        middle_end = total_len - self.n_recent
        middle_indices = list(range(middle_start, middle_end))

        # Select top-k heavy hitters from middle candidate positions
        if self.n_heavy > 0 and middle_indices and scores is not None:
            middle_scores = scores[0, middle_indices]  # (num_candidates,)
            k_heavy = min(self.n_heavy, len(middle_indices))
            _, top_local_idx = torch.topk(middle_scores, k=k_heavy, largest=True)
            heavy_indices = [middle_indices[i.item()] for i in top_local_idx]
        else:
            heavy_indices = []

        # Union and sort retained indices to preserve temporal causal structure
        retained_set = set(sink_indices) | set(heavy_indices) | set(recent_indices)
        retained_indices = sorted(list(retained_set))

        idx_tensor = torch.tensor(retained_indices, dtype=torch.long, device=k.device)

        evicted_count = total_len - len(retained_indices)
        if layer_idx == 0:
            self.total_evicted_tokens += evicted_count

        # Slice cached tensors
        self.k_cache[layer_idx] = torch.index_select(k, dim=2, index=idx_tensor)
        self.v_cache[layer_idx] = torch.index_select(v, dim=2, index=idx_tensor)
        if scores is not None:
            self.attn_scores[layer_idx] = torch.index_select(scores, dim=1, index=idx_tensor)

    def get_seq_len(self, layer_idx: int = 0) -> int:
        """Returns the current cached sequence length for a given layer."""
        if self.k_cache[layer_idx] is None:
            return 0
        return self.k_cache[layer_idx].size(2)

    def reset(self) -> None:
        """Flushes cached keys, values, and attention statistics."""
        self.k_cache = [None] * self.n_layers
        self.v_cache = [None] * self.n_layers
        self.attn_scores = [None] * self.n_layers
        self.total_tokens_seen = 0
        self.total_evicted_tokens = 0

    def memory_bytes(self) -> int:
        """Calculates total memory occupied by cached tensors in bytes."""
        total = 0
        for k, v in zip(self.k_cache, self.v_cache, strict=False):
            if k is not None:
                total += k.element_size() * k.nelement()
            if v is not None:
                total += v.element_size() * v.nelement()
        return total

    def get_stats(self) -> dict:
        """Returns compaction metrics and compression telemetry."""
        cached_len = self.get_seq_len(0)
        uncompressed_bytes = 0
        if self.k_cache[0] is not None:
            element_size = self.k_cache[0].element_size()
            dim_size = self.k_cache[0].size(1) * self.k_cache[0].size(3)
            # uncompressed size if all total_tokens_seen were kept
            uncompressed_bytes = (
                self.total_tokens_seen * dim_size * element_size * 2 * self.n_layers
            )

        current_bytes = self.memory_bytes()
        savings_pct = (
            round((1.0 - (current_bytes / uncompressed_bytes)) * 100.0, 1)
            if uncompressed_bytes > 0
            else 0.0
        )

        return {
            "cached_tokens": cached_len,
            "total_tokens_seen": self.total_tokens_seen,
            "evicted_tokens": self.total_evicted_tokens,
            "max_budget": self.max_budget,
            "n_sink": self.n_sink,
            "n_recent": self.n_recent,
            "n_heavy": self.n_heavy,
            "memory_bytes": current_bytes,
            "uncompressed_bytes": uncompressed_bytes,
            "memory_savings_pct": max(0.0, savings_pct),
        }
