"""
Unit Tests for Phase 30: RoPE Scaling & Long Context
Tests:
- Linear position interpolation mathematical properties
- Dynamic NTK-aware RoPE base frequency adjustment
- YaRN frequency band partitioning & attention temperature scaling
- ScaledRotaryEmbedding module consistency with base RoPE when scale=1.0
- Attention forward pass with scaled RoPE in ModernTransformerLM
"""

import math

import torch

from packages.models.components.rope import precompute_freqs_cis
from packages.models.components.rope_scaling import (
    ScaledRotaryEmbedding,
    ScalingType,
    compute_base_freqs,
    compute_freqs_dynamic_ntk,
    compute_freqs_linear,
    compute_freqs_yarn,
)
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM


def test_compute_base_freqs():
    dim = 64
    theta_base = 10000.0
    freqs = compute_base_freqs(dim, theta_base)

    assert freqs.shape == (dim // 2,)
    # Highest frequency at index 0 should be 1.0 (base^0 = 1)
    assert math.isclose(freqs[0].item(), 1.0, rel_tol=1e-5)
    # Lowest frequency at index d/2 - 1 should be 1 / 10000^(62/64)
    expected_lowest = 1.0 / (theta_base ** (62.0 / 64.0))
    assert math.isclose(freqs[-1].item(), expected_lowest, rel_tol=1e-5)


def test_compute_freqs_linear_identity():
    dim = 32
    seq_len = 128
    cos_scaled, sin_scaled = compute_freqs_linear(dim, seq_len, scale=1.0, theta_base=10000.0)
    cos_base, sin_base = precompute_freqs_cis(dim, seq_len, theta_base=10000.0)

    # When scale=1.0, linear scaling must match standard RoPE exactly
    assert torch.allclose(cos_scaled, cos_base, atol=1e-5)
    assert torch.allclose(sin_scaled, sin_base, atol=1e-5)


def test_compute_freqs_linear_scaling():
    dim = 32
    seq_len = 256
    scale = 2.0
    cos_scaled, sin_scaled = compute_freqs_linear(dim, seq_len, scale=scale, theta_base=10000.0)

    # Position 2 at scale 2 should correspond to position 1 at scale 1
    cos_base, sin_base = precompute_freqs_cis(dim, seq_len, theta_base=10000.0)
    assert torch.allclose(cos_scaled[2], cos_base[1], atol=1e-5)
    assert torch.allclose(sin_scaled[2], sin_base[1], atol=1e-5)


def test_compute_freqs_dynamic_ntk():
    dim = 64
    orig_len = 512
    # Within training length: base is not scaled
    cos_short, _ = compute_freqs_dynamic_ntk(dim, seq_len=256, max_train_seq_len=orig_len)
    cos_orig, _ = compute_freqs_dynamic_ntk(dim, seq_len=512, max_train_seq_len=orig_len)
    assert torch.allclose(cos_short, cos_orig[:256], atol=1e-5)

    # Beyond training length: frequencies scale dynamically
    cos_long, _ = compute_freqs_dynamic_ntk(dim, seq_len=1024, max_train_seq_len=orig_len)
    assert cos_long.shape == (1024, dim)
    # Positions in long sequence should have lower effective frequency (longer wavelength)
    assert not torch.allclose(cos_long[:512], cos_orig, atol=1e-4)


def test_compute_freqs_yarn():
    dim = 64
    seq_len = 1024
    scale = 4.0
    orig_len = 256

    cos, sin, attn_factor = compute_freqs_yarn(
        dim=dim,
        max_seq_len=seq_len,
        scale=scale,
        max_train_seq_len=orig_len,
        beta_fast=32.0,
        beta_slow=1.0,
    )

    assert cos.shape == (seq_len, dim)
    assert sin.shape == (seq_len, dim)
    # YaRN attention temperature factor should scale according to 1 / sqrt(0.1 * ln(4) + 1)
    expected_factor = 1.0 / math.sqrt(0.1 * math.log(scale) + 1.0)
    assert math.isclose(attn_factor, expected_factor, rel_tol=1e-4)


def test_scaled_rotary_embedding_forward():
    dim = 32
    seq_len = 64
    batch_size = 2
    n_heads = 4

    emb = ScaledRotaryEmbedding(
        head_dim=dim,
        max_seq_len=seq_len,
        scaling_type=ScalingType.YARN,
        scale=2.0,
    )

    xq = torch.randn(batch_size, n_heads, seq_len, dim)
    xk = torch.randn(batch_size, n_heads, seq_len, dim)

    out_q, out_k = emb(xq, xk, seq_len=seq_len, start_pos=0)
    assert out_q.shape == xq.shape
    assert out_k.shape == xk.shape
    # Norm of vectors is preserved under 2D Givens rotations
    assert torch.allclose(torch.norm(out_q, dim=-1), torch.norm(xq, dim=-1), atol=1e-4)


def test_modern_transformer_with_yarn_scaling():
    config = ModernTransformerConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        max_context_length=128,
        original_max_seq_len=64,
        rope_scaling_type="yarn",
        rope_scale=2.0,
    )

    model = ModernTransformerLM(config)
    input_ids = torch.randint(0, 100, (2, 80))  # Exceeds original 64-token training window
    logits, loss = model(input_ids)

    assert logits.shape == (2, 80, 100)
    assert loss is None
