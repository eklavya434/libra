"""
Unit tests for Rotary Positional Embeddings (RoPE).
Proves the core mathematical property: dot product depends only on relative distance.
"""

import torch

from packages.models.components.rope import RotaryEmbedding


def test_rope_relative_position_invariance():
    """Mathematically proves that dot_product(q_m, k_n) == dot_product(q_{m+d}, k_{n+d})."""
    head_dim = 32
    max_len = 64
    rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=max_len)

    # Identical query and key features
    q_vec = torch.randn(1, 1, 1, head_dim)
    k_vec = torch.randn(1, 1, 1, head_dim)

    # Full RoPE application across sequence of length 20
    q_seq = q_vec.repeat(1, 1, 20, 1)
    k_seq = k_vec.repeat(1, 1, 20, 1)
    q_rot, k_rot = rope(q_seq, k_seq, seq_len=20)

    # Dot product at (m=5, n=2) -> relative distance 3
    score_5_2 = (q_rot[:, :, 5, :] * k_rot[:, :, 2, :]).sum()

    # Dot product at (m=10, n=7) -> relative distance 3
    score_10_7 = (q_rot[:, :, 10, :] * k_rot[:, :, 7, :]).sum()

    # Dot product at (m=15, n=12) -> relative distance 3
    score_15_12 = (q_rot[:, :, 15, :] * k_rot[:, :, 12, :]).sum()

    # The scores MUST be identical because relative distance (m - n = 3) is identical!
    assert torch.isclose(score_5_2, score_10_7, atol=1e-5)
    assert torch.isclose(score_10_7, score_15_12, atol=1e-5)
