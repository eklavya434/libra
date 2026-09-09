"""
Libra Providers - Cost Tracking & Token Economics Engine

Maintains verified pricing per 1M tokens (input/output) and calculates
exact per-request execution costs, enforcing our Zero-Cost policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelPricing:
    """Pricing rates per 1,000,000 tokens in USD."""

    input_per_million: float
    output_per_million: float
    is_free: bool = False

    @property
    def prompt_cost_per_1m(self) -> float:
        return self.input_per_million

    @property
    def completion_cost_per_1m(self) -> float:
        return self.output_per_million


# Verified official rates (as of late 2024 / 2025)
MODEL_PRICING_TABLE: dict[str, ModelPricing] = {
    # --- ZERO-COST LOCAL & MOCK MODELS ($0.00 / ₹0.00) ---
    "libra-llama-tied": ModelPricing(0.0, 0.0, is_free=True),
    "libra-tiny-llm": ModelPricing(0.0, 0.0, is_free=True),
    "libra-mock-v1": ModelPricing(0.0, 0.0, is_free=True),
    "llama3.2:1b": ModelPricing(0.0, 0.0, is_free=True),
    "llama3.2:3b": ModelPricing(0.0, 0.0, is_free=True),
    "deepseek-r1:1.5b": ModelPricing(0.0, 0.0, is_free=True),
    "qwen2.5:0.5b": ModelPricing(0.0, 0.0, is_free=True),
    "qwen2.5:1.5b": ModelPricing(0.0, 0.0, is_free=True),
    "qwen2.5:7b": ModelPricing(0.0, 0.0, is_free=True),
    "hf/gpt2": ModelPricing(0.0, 0.0, is_free=True),
    # --- OPENAI COMMERCIAL MODELS ---
    "gpt-4o": ModelPricing(2.50, 10.00),
    "gpt-4o-mini": ModelPricing(0.15, 0.60),
    "o1": ModelPricing(15.00, 60.00),
    "o1-mini": ModelPricing(3.00, 12.00),
    # --- ANTHROPIC CLAUDE COMMERCIAL MODELS ---
    "claude-3-5-sonnet-20241022": ModelPricing(3.00, 15.00),
    "claude-3-5-haiku-20241022": ModelPricing(0.80, 4.00),
    # --- GOOGLE GEMINI MODELS ---
    "gemini-1.5-flash": ModelPricing(0.075, 0.30),
    "gemini-1.5-pro": ModelPricing(1.25, 5.00),
    "gemini-2.0-flash": ModelPricing(0.10, 0.40),
    "gemini-2.5-flash": ModelPricing(0.075, 0.30),
    "gemini-2.5-pro": ModelPricing(1.25, 5.00),
    "gemini-flash-latest": ModelPricing(0.075, 0.30),
    "gemini-pro-latest": ModelPricing(1.25, 5.00),
    # --- DEEPSEEK COMMERCIAL API ---
    "deepseek-chat": ModelPricing(0.14, 0.28),
    "deepseek-reasoner": ModelPricing(0.55, 2.19),
    # --- GROQ CLOUD LPU ---
    "llama-3.3-70b-versatile": ModelPricing(0.59, 0.79),
    "llama-3.1-8b-instant": ModelPricing(0.05, 0.08),
}


def get_model_pricing(model_id: str) -> ModelPricing:
    """Retrieve pricing for a given model ID with sensible default fallbacks."""
    model_clean = model_id.lower().strip()
    if model_clean in MODEL_PRICING_TABLE:
        return MODEL_PRICING_TABLE[model_clean]

    # Partial match heuristics
    for key, pricing in MODEL_PRICING_TABLE.items():
        if key in model_clean or model_clean in key:
            return pricing

    # If it's a local Ollama, HF, Libra, or custom local model, it's free
    if (
        ":" in model_clean
        or "libra" in model_clean
        or "mock" in model_clean
        or "ollama" in model_clean
        or "huggingface" in model_clean
        or "hf/" in model_clean
        or "cpu" in model_clean
        or "local" in model_clean
    ):
        return ModelPricing(0.0, 0.0, is_free=True)

    # Generic default for external unknown models
    return ModelPricing(1.0, 2.0, is_free=False)


def calculate_cost(
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    """Calculate USD cost for a token transaction."""
    pricing = get_model_pricing(model_id)
    total_tokens = prompt_tokens + completion_tokens

    if pricing.is_free:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_cost_usd": 0.0,
            "completion_cost_usd": 0.0,
            "total_cost_usd": 0.0,
            "is_free": True,
            "model": model_id,
        }

    prompt_cost = (prompt_tokens / 1_000_000) * pricing.input_per_million
    comp_cost = (completion_tokens / 1_000_000) * pricing.output_per_million
    total_cost = prompt_cost + comp_cost

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_cost_usd": round(prompt_cost, 6),
        "completion_cost_usd": round(comp_cost, 6),
        "total_cost_usd": round(total_cost, 6),
        "is_free": False,
        "model": model_id,
    }
