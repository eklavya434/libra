"""
Unit tests for Libra Model Cost Tracker (Phase 9).
Verifies exact pricing calculations, local zero-cost guarantees, and custom model handling.
"""

from __future__ import annotations

import pytest

from packages.providers.cost import (
    calculate_cost,
    get_model_pricing,
)


def test_zero_cost_guarantee_for_local_and_mock():
    """All local and mock models MUST strictly return 0.0 cost."""
    local_models = [
        "ollama/llama3.2:1b",
        "llama3.2:1b",
        "huggingface/gpt2",
        "gpt2",
        "mock-model",
        "libra-educational-tiny",
        "custom-cpu-model",
    ]
    for m in local_models:
        cost = calculate_cost(m, prompt_tokens=1_000_000, completion_tokens=1_000_000)
        assert cost["total_cost_usd"] == 0.0
        assert cost["prompt_cost_usd"] == 0.0
        assert cost["completion_cost_usd"] == 0.0
        assert cost["total_tokens"] == 2_000_000


def test_cloud_pricing_lookup():
    """Verify known cloud models match expected token pricing."""
    gpt4o_pricing = get_model_pricing("gpt-4o")
    assert gpt4o_pricing.prompt_cost_per_1m == 2.50
    assert gpt4o_pricing.completion_cost_per_1m == 10.00

    claude_haiku = get_model_pricing("claude-3-5-haiku-20241022")
    assert claude_haiku.prompt_cost_per_1m == 0.80
    assert claude_haiku.completion_cost_per_1m == 4.00

    deepseek_v3 = get_model_pricing("deepseek-chat")
    assert deepseek_v3.prompt_cost_per_1m == 0.14
    assert deepseek_v3.completion_cost_per_1m == 0.28


def test_exact_cost_calculation():
    """Verify exact dollar calculation for sample token counts."""
    cost = calculate_cost("gpt-4o-mini", prompt_tokens=10_000, completion_tokens=2_000)
    assert pytest.approx(cost["prompt_cost_usd"], rel=1e-4) == 0.0015
    assert pytest.approx(cost["completion_cost_usd"], rel=1e-4) == 0.0012
    assert pytest.approx(cost["total_cost_usd"], rel=1e-4) == 0.0027


def test_prefix_matching_for_versions():
    """Models with date stamps or versions should match base model pricing."""
    pricing = get_model_pricing("gpt-4o-2024-08-06")
    assert pricing.prompt_cost_per_1m == 2.50
