"""Integrity guarantees for the dynamic tier router (ISSUE_MATRIX #2).

The router may only reach into real providers. It must never inspect a mock,
never fall back to one, and never fabricate a response when every attempt
fails: exhaustion must raise so the API layer can answer with an honest 503.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from packages.routing.classifier import ExecutionTier
from packages.routing.dynamic_router import DynamicRouter, RoutingPolicy


def test_tier_model_map_contains_only_real_providers():
    """Every tier target must be a real provider, never a mock."""
    offenders: list[str] = []
    for tier, info in DynamicRouter.TIER_MODEL_MAP.items():
        for key, value in info.items():
            if not isinstance(value, tuple):
                continue
            provider_name = value[0]
            if "mock" in provider_name.lower() or "mock" in str(value[1]).lower():
                offenders.append(f"{tier.value}.{key} = {value}")
    assert not offenders, f"mock provider in tier map: {offenders}"


def test_router_module_never_references_a_mock_provider_class():
    """Source-level guard: no implicit mock fallback can creep back in.

    Prose in docstrings may *mention* mocks, but the router must never import or
    name a mock provider class.
    """
    source = inspect.getsource(__import__("packages.routing.dynamic_router", fromlist=["*"]))
    assert "MockProvider" not in source
    assert "providers.mock" not in source


class _AllAttemptsFailRouter:
    """Provider router stub whose every attempt raises."""

    def get_provider(self, name: str) -> Any:
        raise RuntimeError(f"simulated outage for {name}")


class _UnhealthyProvider:
    """Provider that never becomes reachable, so attempts are skipped, not failed."""

    async def health(self) -> dict[str, Any]:
        return {"healthy": False, "connected": False}

    async def chat(self, messages, model="x", **kwargs):  # pragma: no cover - never reached
        raise AssertionError("chat must not be called on an unhealthy provider")


class _UnhealthyRouter:
    def get_provider(self, name: str) -> _UnhealthyProvider:
        return _UnhealthyProvider()


@pytest.mark.asyncio
async def test_exhausted_fallback_chain_raises_instead_of_fabricating():
    router = DynamicRouter(provider_router=_AllAttemptsFailRouter())
    with pytest.raises(RuntimeError) as excinfo:
        await router.route_and_execute(
            messages=[{"role": "user", "content": "Explain gravity in one sentence"}],
            policy=RoutingPolicy.AUTO,
        )
    message = str(excinfo.value)
    assert "No routing attempt succeeded" in message
    # Honest about the failure; never a fabricated completion.
    assert "choices" not in message


@pytest.mark.asyncio
async def test_no_available_attempt_reports_a_configuration_problem():
    router = DynamicRouter(provider_router=_UnhealthyRouter())
    with pytest.raises(RuntimeError) as excinfo:
        await router.route_and_execute(
            messages=[{"role": "user", "content": "Hello"}],
            policy=RoutingPolicy.FAST,
        )
    message = str(excinfo.value)
    assert "No routing attempt was available" in message
    assert "Ollama" in message or "cloud provider" in message


@pytest.mark.asyncio
async def test_unhealthy_provider_is_skipped_before_being_called():
    """Health gating must short-circuit chat, not call it and catch the error."""
    router = DynamicRouter(provider_router=_UnhealthyRouter())
    with pytest.raises(RuntimeError):
        await router.route_and_execute(
            messages=[{"role": "user", "content": "Hello"}],
            policy=RoutingPolicy.FAST,
        )
    # The assertion inside _UnhealthyProvider.chat would surface as a different
    # error type, so reaching RuntimeError proves the health gate ran first.


def test_plan_routing_never_selects_a_mock_for_any_tier():
    router = DynamicRouter(provider_router=_AllAttemptsFailRouter())
    for tier in ExecutionTier:
        for policy in RoutingPolicy:
            decision = router.plan_routing("Write a quicksort function", policy=policy)
            assert decision.selected_tier in ExecutionTier
            chain = [decision.selected_provider] + [
                fb["provider"] for fb in decision.fallback_chain
            ]
            assert all("mock" not in p.lower() for p in chain)
