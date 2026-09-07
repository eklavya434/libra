"""
Libra Routing - Dynamic Model Router & Fallback Orchestrator

Routes user prompts to optimal model execution tiers based on query complexity,
intent, latency constraints, cost budgets, and provider health, with automated
multi-level fallback cascades.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from packages.providers.router import ProviderRouter, get_router
from packages.routing.classifier import (
    ExecutionTier,
    QueryClassification,
    QueryClassifier,
    get_query_classifier,
)

logger = logging.getLogger(__name__)


class RoutingPolicy(str, Enum):
    """User-selectable routing policies."""

    AUTO = "auto"  # Dynamic intent- and complexity-based routing
    FAST = "fast"  # Enforce Fast/Local tier (low latency, zero cost)
    BALANCED = "balanced"  # Enforce Balanced tier (standard capability)
    QUALITY = "quality"  # Enforce Frontier Reasoning tier (maximum capability)
    MULTI_AGENT = "multi_agent"  # Enforce Collaborative Multi-Agent ensemble


class RoutingDecision(BaseModel):
    """Structured decision explaining selected tier, model, provider, and fallback path."""

    classification: QueryClassification
    selected_tier: ExecutionTier
    selected_model: str
    selected_provider: str
    fallback_chain: list[dict[str, str]] = Field(
        default_factory=list,
        description="Ordered list of (provider, model) pairs to attempt on failure",
    )
    estimated_latency_tier: str
    estimated_cost_tier: str
    policy_applied: RoutingPolicy
    routing_rationale: str


class DynamicRouter:
    """Intelligently routes inference requests across model tiers with fallback guarantees."""

    # Default model mapping per tier
    TIER_MODEL_MAP: ClassVar[dict[ExecutionTier, dict[str, Any]]] = {
        ExecutionTier.FAST_LOCAL: {
            "primary": ("mock-provider", "mock-model"),
            "local_alt": ("ollama", "qwen2.5:0.5b"),
            "latency": "< 50ms",
            "cost": "$0 / ₹0 (Local CPU)",
        },
        ExecutionTier.BALANCED: {
            "primary": ("mock-provider", "mock-model"),
            "cloud_alt": ("openai", "gpt-4o-mini"),
            "local_alt": ("ollama", "llama3.2:3b"),
            "latency": "200ms - 800ms",
            "cost": "$0 / ₹0 (Local / Mock)",
        },
        ExecutionTier.FRONTIER_REASONING: {
            "primary": ("mock-provider", "mock-model"),
            "cloud_alt": ("deepseek", "deepseek-r1"),
            "local_alt": ("ollama", "qwen2.5-coder:7b"),
            "latency": "1.0s - 3.5s",
            "cost": "$0 / ₹0 (Local / Mock)",
        },
        ExecutionTier.MULTI_AGENT: {
            "primary": ("mock-provider", "mock-model"),
            "team_mode": True,
            "latency": "3.0s - 10.0s",
            "cost": "$0 / ₹0 (Local / Mock)",
        },
    }

    def __init__(
        self,
        classifier: QueryClassifier | None = None,
        provider_router: ProviderRouter | None = None,
    ) -> None:
        self.classifier = classifier or get_query_classifier()
        self.provider_router = provider_router or get_router()

    def plan_routing(
        self,
        prompt: str,
        policy: RoutingPolicy = RoutingPolicy.AUTO,
        prefer_local: bool = True,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> RoutingDecision:
        """Determines tier, model, provider, and fallback chain without executing."""
        classification = self.classifier.classify(prompt, conversation_history)

        # 1. Resolve Tier based on Policy
        if policy == RoutingPolicy.FAST:
            selected_tier = ExecutionTier.FAST_LOCAL
            rationale = "Forced Fast/Local policy by user configuration."
        elif policy == RoutingPolicy.BALANCED:
            selected_tier = ExecutionTier.BALANCED
            rationale = "Forced Balanced policy by user configuration."
        elif policy == RoutingPolicy.QUALITY:
            selected_tier = ExecutionTier.FRONTIER_REASONING
            rationale = "Forced Quality/Frontier reasoning policy by user configuration."
        elif policy == RoutingPolicy.MULTI_AGENT:
            selected_tier = ExecutionTier.MULTI_AGENT
            rationale = "Forced Multi-Agent team execution policy by user configuration."
        else:
            selected_tier = classification.recommended_tier
            rationale = f"Dynamic policy: {classification.rationale}"

        # 2. Resolve Model and Provider
        tier_info = self.TIER_MODEL_MAP[selected_tier]
        primary_provider, primary_model = tier_info["primary"]

        # 3. Construct Fallback Cascade Chain
        fallbacks: list[dict[str, str]] = []
        if "local_alt" in tier_info:
            fallbacks.append(
                {"provider": tier_info["local_alt"][0], "model": tier_info["local_alt"][1]}
            )
        if "cloud_alt" in tier_info:
            fallbacks.append(
                {"provider": tier_info["cloud_alt"][0], "model": tier_info["cloud_alt"][1]}
            )
        # Universal zero-cost safety net
        fallbacks.append({"provider": "mock-provider", "model": "mock-model"})

        # Avoid redundant duplicate fallbacks
        seen = set()
        deduped_fallbacks: list[dict[str, str]] = []
        for fb in fallbacks:
            pair = (fb["provider"], fb["model"])
            if pair not in seen and pair != (primary_provider, primary_model):
                seen.add(pair)
                deduped_fallbacks.append(fb)

        return RoutingDecision(
            classification=classification,
            selected_tier=selected_tier,
            selected_model=primary_model,
            selected_provider=primary_provider,
            fallback_chain=deduped_fallbacks,
            estimated_latency_tier=tier_info["latency"],
            estimated_cost_tier=tier_info["cost"],
            policy_applied=policy,
            routing_rationale=rationale,
        )

    async def route_and_execute(
        self,
        messages: list[dict[str, Any]],
        policy: RoutingPolicy = RoutingPolicy.AUTO,
        prefer_local: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Plans routing and executes completion with multi-level fallback resilience."""
        # Extract prompt from last user message
        prompt = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                prompt = m.get("content", "")
                break

        decision = self.plan_routing(
            prompt=prompt,
            policy=policy,
            prefer_local=prefer_local,
            conversation_history=messages[:-1] if len(messages) > 1 else None,
        )

        # Attempt primary provider
        attempts = [
            {"provider": decision.selected_provider, "model": decision.selected_model}
        ] + decision.fallback_chain
        last_error = None

        for attempt in attempts:
            prov_name = attempt["provider"]
            model_name = attempt["model"]
            try:
                provider = self.provider_router.get_provider(prov_name)
                # Check health for network/daemon providers
                if hasattr(provider, "health"):
                    health = await provider.health()
                    if not health.get("healthy", True) and not health.get("connected", False):
                        continue

                result = await provider.chat(messages, model=model_name, **kwargs)
                return {
                    "content": result.get("content", ""),
                    "model_used": model_name,
                    "provider_used": prov_name,
                    "tier_used": decision.selected_tier.value,
                    "routing_decision": decision.model_dump(),
                    "fallback_triggered": (prov_name != decision.selected_provider),
                    "usage": result.get("usage", {}),
                }
            except Exception as e:  # noqa: BLE001
                logger.warning("Routing attempt failed for %s/%s: %s", prov_name, model_name, e)
                last_error = e

        # Final safety net
        mock = self.provider_router.get_provider("mock-provider")
        result = await mock.chat(messages, model="mock-model", **kwargs)
        return {
            "content": result.get("content", ""),
            "model_used": "mock-model",
            "provider_used": "mock-provider",
            "tier_used": ExecutionTier.FAST_LOCAL.value,
            "routing_decision": decision.model_dump(),
            "fallback_triggered": True,
            "error_fallback": str(last_error) if last_error else None,
            "usage": result.get("usage", {}),
        }


_default_dynamic_router: DynamicRouter | None = None


def get_dynamic_router() -> DynamicRouter:
    """Singleton getter for DynamicRouter."""
    global _default_dynamic_router
    if _default_dynamic_router is None:
        _default_dynamic_router = DynamicRouter()
    return _default_dynamic_router
