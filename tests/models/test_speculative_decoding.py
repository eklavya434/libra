"""
Tests for First-Principles Speculative Decoding Engine (Phase 21)
"""

import torch

from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.models.speculative import (
    SpeculativeDecoder,
    standard_autoregressive_generate,
)


def create_models() -> tuple[ModernTransformerLM, ModernTransformerLM]:
    """Creates deterministic lightweight target and draft models sharing vocab and embedding space."""
    torch.manual_seed(42)
    # Draft model: 1 layer, fast
    draft_cfg = ModernTransformerConfig(
        vocab_size=64,
        d_model=32,
        n_layers=1,
        n_heads=2,
        max_context_length=64,
        hidden_dim=64,
    )
    # Target model: 3 layers, authoritative
    target_cfg = ModernTransformerConfig(
        vocab_size=64,
        d_model=32,
        n_layers=3,
        n_heads=2,
        max_context_length=64,
        hidden_dim=64,
    )

    draft_m = ModernTransformerLM(draft_cfg)
    target_m = ModernTransformerLM(target_cfg)
    return draft_m, target_m


def test_speculative_decoding_exact_mathematical_equivalence():
    """
    Core theorem of speculative decoding:
    Under greedy decoding (temperature=0.0), speculative decoding MUST produce
    the EXACT SAME token sequence as standard target model autoregressive generation.
    """
    draft_m, target_m = create_models()

    prompt_tokens = [1, 5, 12, 20]
    max_new_tokens = 16

    # 1. Baseline standard autoregressive generation on Target Model
    baseline_tokens, baseline_passes = standard_autoregressive_generate(
        target_m, prompt_tokens, max_new_tokens=max_new_tokens, temperature=0.0
    )

    # 2. Speculative decoding with Lookahead K=3
    decoder = SpeculativeDecoder(
        target_model=target_m,
        draft_model=draft_m,
        lookahead_k=3,
        temperature=0.0,
    )
    spec_result = decoder.generate(prompt_tokens, max_new_tokens=max_new_tokens)

    # Invariant: Output tokens must match 100% token-for-token
    assert spec_result.output_tokens == baseline_tokens
    assert spec_result.tokens_generated == len(baseline_tokens)
    assert spec_result.baseline_forward_passes == baseline_passes


def test_speculative_decoding_telemetry():
    """Verifies speculative telemetry metrics (acceptance rate, passes, speedup)."""
    draft_m, target_m = create_models()

    prompt_tokens = [2, 10, 15]
    decoder = SpeculativeDecoder(
        target_model=target_m,
        draft_model=draft_m,
        lookahead_k=2,
        temperature=0.0,
    )
    result = decoder.generate(prompt_tokens, max_new_tokens=10)

    assert 0.0 <= result.acceptance_rate <= 1.0
    assert result.draft_tokens_proposed >= 0
    assert result.draft_tokens_accepted <= result.draft_tokens_proposed
    assert result.target_forward_passes <= result.baseline_forward_passes
    assert result.passes_saved >= 0
    assert result.theoretical_speedup >= 1.0


def test_speculative_decoding_different_k():
    """Verifies speculative decoding across various lookahead window sizes K."""
    draft_m, target_m = create_models()
    prompt_tokens = [4, 8, 16]

    for k in [1, 2, 4]:
        decoder = SpeculativeDecoder(
            target_model=target_m,
            draft_model=draft_m,
            lookahead_k=k,
            temperature=0.0,
        )
        result = decoder.generate(prompt_tokens, max_new_tokens=8)
        assert len(result.output_tokens) == 8


def test_speculative_decoding_temperature():
    """Verifies that non-zero temperature executes without error."""
    draft_m, target_m = create_models()
    prompt_tokens = [3, 9, 27]

    decoder = SpeculativeDecoder(
        target_model=target_m,
        draft_model=draft_m,
        lookahead_k=3,
        temperature=0.7,
    )
    result = decoder.generate(prompt_tokens, max_new_tokens=6)
    assert len(result.output_tokens) == 6
