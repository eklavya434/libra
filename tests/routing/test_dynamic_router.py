"""
Tests for Dynamic Tier Router & Fallback Cascading (Phase 21)
"""

import pytest

from packages.routing.classifier import ExecutionTier
from packages.routing.dynamic_router import (
    DynamicRouter,
    RoutingPolicy,
    get_dynamic_router,
)


@pytest.fixture
def router() -> DynamicRouter:
    return get_dynamic_router()


def test_routing_policy_auto(router: DynamicRouter):
    decision = router.plan_routing("Hi!", policy=RoutingPolicy.AUTO)
    assert decision.selected_tier == ExecutionTier.FAST_LOCAL
    assert decision.policy_applied == RoutingPolicy.AUTO
    assert len(decision.fallback_chain) >= 1


def test_routing_policy_forced_fast(router: DynamicRouter):
    complex_prompt = "Write a distributed compiler in Rust with formal proof."
    decision = router.plan_routing(complex_prompt, policy=RoutingPolicy.FAST)
    assert decision.selected_tier == ExecutionTier.FAST_LOCAL
    assert decision.policy_applied == RoutingPolicy.FAST


def test_routing_policy_forced_frontier(router: DynamicRouter):
    simple_prompt = "Hello"
    decision = router.plan_routing(simple_prompt, policy=RoutingPolicy.QUALITY)
    assert decision.selected_tier == ExecutionTier.FRONTIER_REASONING
    assert decision.policy_applied == RoutingPolicy.QUALITY


def test_routing_policy_forced_multi_agent(router: DynamicRouter):
    decision = router.plan_routing("Design system", policy=RoutingPolicy.MULTI_AGENT)
    assert decision.selected_tier == ExecutionTier.MULTI_AGENT


@pytest.mark.asyncio
async def test_route_and_execute_success(router: DynamicRouter):
    messages = [{"role": "user", "content": "What is 2+2?"}]
    result = await router.route_and_execute(messages, policy=RoutingPolicy.AUTO)

    assert "content" in result
    assert "model_used" in result
    assert "routing_decision" in result
    assert result["tier_used"] in [t.value for t in ExecutionTier]


@pytest.mark.asyncio
async def test_route_and_execute_fallback(router: DynamicRouter):
    messages = [{"role": "user", "content": "Hello"}]
    # Force a tier and ensure execution succeeds with mock provider
    result = await router.route_and_execute(messages, policy=RoutingPolicy.BALANCED)
    assert "content" in result
    assert result["model_used"] is not None
