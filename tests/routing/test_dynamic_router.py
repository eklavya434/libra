"""
Tests for Dynamic Tier Router & Fallback Cascading (Phase 21)

The router must only ever reach **real** providers. It must never fabricate a
mock/echo response when no provider can serve the request: execution raises an
actionable error instead.
"""

import pytest

from packages.providers.mock import MockProvider
from packages.providers.router import ProviderRouter
from packages.routing.classifier import ExecutionTier
from packages.routing.dynamic_router import (
    DynamicRouter,
    RoutingPolicy,
)


class _AllMockRouter(ProviderRouter):
    """Stub that maps every (provider, model) pair to the real MockProvider so
    routing logic is testable without external engines."""

    def get_provider(self, name: str):
        return MockProvider(name=name)


@pytest.fixture
def stub_router() -> DynamicRouter:
    return DynamicRouter(provider_router=_AllMockRouter())


def test_routing_policy_auto(stub_router: DynamicRouter):
    decision = stub_router.plan_routing("Hi!", policy=RoutingPolicy.AUTO)
    assert decision.selected_tier == ExecutionTier.FAST_LOCAL
    assert decision.policy_applied == RoutingPolicy.AUTO
    assert len(decision.fallback_chain) >= 0

    # No mock provider anywhere in the plan.
    all_pairs = [(decision.selected_provider, decision.selected_model)] + [
        (f["provider"], f["model"]) for f in decision.fallback_chain
    ]
    assert all(prov != "mock-provider" for prov, _ in all_pairs)


def test_routing_policy_forced_fast(stub_router: DynamicRouter):
    complex_prompt = "Write a distributed compiler in Rust with formal proof."
    decision = stub_router.plan_routing(complex_prompt, policy=RoutingPolicy.FAST)
    assert decision.selected_tier == ExecutionTier.FAST_LOCAL
    assert decision.policy_applied == RoutingPolicy.FAST
    assert decision.selected_provider != "mock-provider"


def test_routing_policy_forced_frontier(stub_router: DynamicRouter):
    simple_prompt = "Hello"
    decision = stub_router.plan_routing(simple_prompt, policy=RoutingPolicy.QUALITY)
    assert decision.selected_tier == ExecutionTier.FRONTIER_REASONING
    assert decision.policy_applied == RoutingPolicy.QUALITY
    assert decision.selected_provider != "mock-provider"


def test_routing_policy_forced_multi_agent(stub_router: DynamicRouter):
    decision = stub_router.plan_routing("Design system", policy=RoutingPolicy.MULTI_AGENT)
    assert decision.selected_tier == ExecutionTier.MULTI_AGENT
    assert decision.selected_provider != "mock-provider"


@pytest.mark.asyncio
async def test_route_and_execute_success(stub_router: DynamicRouter):
    messages = [{"role": "user", "content": "What is 2+2?"}]
    result = await stub_router.route_and_execute(messages, policy=RoutingPolicy.AUTO)

    assert "content" in result
    assert result["content"], "expected a real (stubbed) provider response, not empty text"
    assert "model_used" in result
    assert "routing_decision" in result
    assert result["tier_used"] in [t.value for t in ExecutionTier]
    assert result["provider_used"] != "mock-provider"


@pytest.mark.asyncio
async def test_route_and_execute_fallback(stub_router: DynamicRouter):
    messages = [{"role": "user", "content": "Hello"}]
    result = await stub_router.route_and_execute(messages, policy=RoutingPolicy.BALANCED)
    assert "content" in result
    assert result["content"]
    assert result["model_used"] is not None


@pytest.mark.asyncio
async def test_route_and_execute_never_fabricates_mock_output():
    """When every attempt fails, the router must raise, never echo a mock reply."""
    from packages.providers.base import BaseProvider

    class _FailingProvider(BaseProvider):
        @property
        def name(self) -> str:
            return "stub-fail"

        async def chat(self, messages, model="x", **kwargs):
            raise RuntimeError("simulated provider outage")

        async def health(self):
            return {"healthy": True, "connected": True}

        async def list_models(self):
            return []

        async def stream(self, messages, model="x", **kwargs):
            if False:
                yield ""

        async def embeddings(self, texts, model=None):
            return [[0.0] * 8 for _ in texts]

        def capabilities(self):
            return {"supports_text": True}

    failing_router = ProviderRouter()
    failing_router._providers["ollama"] = _FailingProvider()
    router = DynamicRouter(provider_router=failing_router)

    messages = [{"role": "user", "content": "Hello"}]
    with pytest.raises(RuntimeError) as excinfo:
        await router.route_and_execute(messages, policy=RoutingPolicy.AUTO)
    assert "No routing attempt succeeded" in str(excinfo.value)
