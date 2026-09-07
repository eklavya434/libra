"""
Tests for Grouped-Query Attention (GQA), Multi-Query Attention (MQA),
and Exact Equivalence between Cached and Cacheless Generation.
Phase 23: Key-Value (KV) Cache Optimization and Grouped-Query Attention (MQA/GQA)
"""

import pytest
import torch

from packages.models.generation import generate, generate_with_cache
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernCausalAttention, ModernTransformerLM


def test_gqa_config_validation():
    # Valid GQA config
    cfg = ModernTransformerConfig(d_model=64, n_heads=4, n_kv_heads=2)
    assert cfg.num_queries_per_kv == 2
    assert cfg.kv_dim == 32

    # Valid MQA config
    cfg_mqa = ModernTransformerConfig(d_model=64, n_heads=4, n_kv_heads=1)
    assert cfg_mqa.num_queries_per_kv == 4
    assert cfg_mqa.kv_dim == 16

    # Invalid: n_heads not divisible by n_kv_heads
    with pytest.raises(ValueError, match="must be divisible"):
        ModernTransformerConfig(d_model=64, n_heads=4, n_kv_heads=3)


def test_gqa_attention_shapes():
    # MHA: n_heads=4, n_kv=4
    cfg_mha = ModernTransformerConfig(
        vocab_size=50, d_model=32, n_heads=4, n_kv_heads=4, n_layers=1
    )
    attn_mha = ModernCausalAttention(cfg_mha)
    assert attn_mha.k_proj.out_features == 32
    assert attn_mha.v_proj.out_features == 32

    # GQA: n_heads=4, n_kv=2
    cfg_gqa = ModernTransformerConfig(
        vocab_size=50, d_model=32, n_heads=4, n_kv_heads=2, n_layers=1
    )
    attn_gqa = ModernCausalAttention(cfg_gqa)
    assert attn_gqa.k_proj.out_features == 16
    assert attn_gqa.v_proj.out_features == 16

    # MQA: n_heads=4, n_kv=1
    cfg_mqa = ModernTransformerConfig(
        vocab_size=50, d_model=32, n_heads=4, n_kv_heads=1, n_layers=1
    )
    attn_mqa = ModernCausalAttention(cfg_mqa)
    assert attn_mqa.k_proj.out_features == 8
    assert attn_mqa.v_proj.out_features == 8

    # Forward pass tensor shape checks
    x = torch.randn(2, 6, 32)
    out_gqa = attn_gqa(x)
    assert out_gqa.shape == (2, 6, 32)


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
def test_cache_vs_nocache_exact_equivalence(n_kv_heads):
    """
    Verify 100% mathematical equivalence: greedy autoregressive generation
    with KV cache produces the EXACT same tokens as cacheless generation.
    """
    torch.manual_seed(42)
    cfg = ModernTransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_kv_heads=n_kv_heads,
        n_layers=2,
        max_context_length=64,
    )
    model = ModernTransformerLM(cfg)
    model.eval()

    prompt = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    max_new_tokens = 10

    with torch.no_grad():
        out_nocache = generate(
            model, prompt.clone(), max_new_tokens=max_new_tokens, temperature=0.0
        )
        out_cache, _ = generate_with_cache(
            model, prompt.clone(), max_new_tokens=max_new_tokens, temperature=0.0
        )

    assert torch.equal(out_nocache, out_cache), (
        f"Mismatch for n_kv_heads={n_kv_heads}!\n"
        f"NoCache: {out_nocache.tolist()}\n"
        f"Cache:   {out_cache.tolist()}"
    )
