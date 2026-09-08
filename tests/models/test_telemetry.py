"""
Tests for First-Principles Token Telemetry & Token-Level Metrics
Verifies Shannon surprisal, entropy, top-k candidates, sequence perplexity, and generation.
"""

import math

import pytest
import torch

from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.models.telemetry import (
    TokenTelemetry,
    aggregate_sequence_telemetry,
    analyze_sequence_telemetry,
    compute_entropy_bits,
    compute_step_telemetry,
    compute_surprisal_bits,
    stream_generate_with_telemetry,
)


def test_surprisal_calculation():
    """Verifies exact Shannon self-information / surprisal: I(w) = -log2(p(w))."""
    # Deterministic event: p = 1.0 -> 0 bits of information
    assert math.isclose(compute_surprisal_bits(1.0), 0.0, abs_tol=1e-6)

    # 50% probability: 1 bit of information
    assert math.isclose(compute_surprisal_bits(0.5), 1.0, abs_tol=1e-6)

    # 25% probability: 2 bits of information
    assert math.isclose(compute_surprisal_bits(0.25), 2.0, abs_tol=1e-6)

    # 12.5% probability: 3 bits of information
    assert math.isclose(compute_surprisal_bits(0.125), 3.0, abs_tol=1e-6)

    # Near-zero probability is clamped gracefully without crash
    surprisal_zero = compute_surprisal_bits(0.0)
    assert surprisal_zero > 30.0  # -log2(1e-12) ~ 39.86 bits


def test_entropy_calculation():
    """Verifies Shannon entropy boundary values and mathematical properties."""
    # Deterministic one-hot distribution: H(P) = 0 bits
    one_hot = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert math.isclose(compute_entropy_bits(one_hot), 0.0, abs_tol=1e-6)

    # Uniform distribution over 4 items: H(P) = log2(4) = 2.0 bits
    uniform_4 = torch.tensor([0.25, 0.25, 0.25, 0.25])
    assert math.isclose(compute_entropy_bits(uniform_4), 2.0, abs_tol=1e-6)

    # Uniform distribution over 8 items: H(P) = log2(8) = 3.0 bits
    uniform_8 = torch.full((8,), 1.0 / 8.0)
    assert math.isclose(compute_entropy_bits(uniform_8), 3.0, abs_tol=1e-6)

    # Permutation invariance: changing token order does not alter distribution entropy
    p1 = torch.tensor([0.7, 0.2, 0.1])
    p2 = torch.tensor([0.1, 0.7, 0.2])
    assert math.isclose(compute_entropy_bits(p1), compute_entropy_bits(p2), abs_tol=1e-6)


def test_step_telemetry_top_k():
    """Verifies step telemetry produces sorted top-k candidates with valid probabilities."""
    # Construct logits where token 3 has highest logit, token 1 second, token 0 third
    logits = torch.tensor([1.0, 2.0, 0.5, 4.0, -1.0])
    chosen_id = 3

    telem = compute_step_telemetry(
        logits=logits,
        chosen_token_id=chosen_id,
        index=0,
        latency_ms=12.5,
        candidate_top_k=3,
        temperature=1.0,
    )

    assert telem.index == 0
    assert telem.token_id == 3
    assert telem.latency_ms == 12.5
    assert telem.prob > 0.5  # Token 3 should dominate
    assert telem.surprisal_bits == pytest.approx(-math.log2(telem.prob), rel=1e-3)
    assert telem.entropy_bits > 0.0

    # Verify top-k candidates are sorted descending
    assert len(telem.top_k) == 3
    assert telem.top_k[0].token_id == 3
    assert telem.top_k[1].token_id == 1
    assert telem.top_k[2].token_id == 0
    assert telem.top_k[0].prob >= telem.top_k[1].prob >= telem.top_k[2].prob


def test_sequence_aggregation_perplexity():
    """Verifies that SequenceTelemetry calculates perplexity = 2^(mean_surprisal) accurately."""
    # Construct 3 synthetic token telemetries with known surprisals: 2.0, 3.0, 1.0
    # Mean surprisal = 2.0 bits -> Perplexity = 2^2 = 4.0
    t0 = TokenTelemetry(
        index=0,
        token_id=10,
        token_text="Hello",
        prob=0.25,
        logprob=math.log(0.25),
        surprisal_bits=2.0,
        entropy_bits=2.5,
        latency_ms=10.0,
        top_k=[],
    )
    t1 = TokenTelemetry(
        index=1,
        token_id=20,
        token_text=" world",
        prob=0.125,
        logprob=math.log(0.125),
        surprisal_bits=3.0,
        entropy_bits=2.2,
        latency_ms=12.0,
        top_k=[],
    )
    t2 = TokenTelemetry(
        index=2,
        token_id=30,
        token_text="!",
        prob=0.5,
        logprob=math.log(0.5),
        surprisal_bits=1.0,
        entropy_bits=1.8,
        latency_ms=8.0,
        top_k=[],
    )

    seq = aggregate_sequence_telemetry([t0, t1, t2], total_duration_ms=30.0)

    assert seq.total_tokens == 3
    assert seq.text == "Hello world!"
    assert math.isclose(seq.mean_surprisal_bits, 2.0, abs_tol=1e-4)
    assert math.isclose(seq.perplexity, 4.0, abs_tol=1e-4)
    assert seq.max_surprisal_token.token_id == 20
    assert seq.min_surprisal_token.token_id == 30
    assert seq.tokens_per_second == pytest.approx(3 / 0.03, rel=1e-2)


def test_stream_generate_with_telemetry():
    """Verifies autoregressive generation with real-time telemetry yield."""
    torch.manual_seed(42)
    config = ModernTransformerConfig(
        vocab_size=64,
        d_model=32,
        n_heads=2,
        n_layers=2,
        max_context_length=64,
    )
    model = ModernTransformerLM(config)
    model.eval()

    prompt = torch.tensor([[1, 5, 12]], dtype=torch.long)
    max_new = 4

    gen = stream_generate_with_telemetry(
        model=model,
        idx=prompt,
        max_new_tokens=max_new,
        temperature=0.7,
        candidate_top_k=3,
    )

    tokens = []
    for tok in gen:
        assert isinstance(tok, TokenTelemetry)
        assert tok.prob > 0.0
        assert tok.surprisal_bits >= 0.0
        assert tok.latency_ms >= 0.0
        assert len(tok.top_k) == 3
        tokens.append(tok)

    assert len(tokens) == max_new
    for i, tok in enumerate(tokens):
        assert tok.index == i


def test_analyze_sequence_telemetry():
    """Verifies teacher-forcing evaluation across an existing sequence."""
    torch.manual_seed(42)
    config = ModernTransformerConfig(
        vocab_size=64,
        d_model=32,
        n_heads=2,
        n_layers=2,
        max_context_length=64,
    )
    model = ModernTransformerLM(config)
    model.eval()

    input_tokens = torch.tensor([[2, 10, 15, 22, 31]], dtype=torch.long)
    seq_telem = analyze_sequence_telemetry(model, input_tokens, candidate_top_k=4)

    # 5 tokens in -> 4 transition predictions (token 1 predicts 2, 2 predicts 3, etc.)
    assert seq_telem.total_tokens == 4
    assert len(seq_telem.tokens) == 4
    assert seq_telem.perplexity > 0.0
    assert seq_telem.mean_surprisal_bits > 0.0
    assert seq_telem.max_surprisal_token is not None
