"""
Tests for Web Search Providers (Mock and DuckDuckGo)
"""

import pytest

from packages.rag.web_search import (
    DuckDuckGoSearchProvider,
    MockSearchProvider,
    get_web_search_provider,
)


@pytest.mark.anyio
async def test_mock_search_provider_keywords():
    provider = MockSearchProvider()
    results = await provider.search("transformer attention mechanism", max_results=2)
    assert len(results) > 0
    assert any("Transformer" in r.title for r in results)
    assert results[0].url.startswith("http")
    assert results[0].source == "mock"


@pytest.mark.anyio
async def test_mock_search_provider_fallback():
    provider = MockSearchProvider()
    results = await provider.search("obscure xyz term 123", max_results=3)
    assert len(results) == 3
    assert results[0].source == "mock"


@pytest.mark.anyio
async def test_duckduckgo_provider_fallback_to_mock():
    # Verify DuckDuckGo provider falls back gracefully to Mock
    provider = DuckDuckGoSearchProvider(timeout_sec=0.001)
    results = await provider.search("SQLite WAL", max_results=2)
    assert len(results) > 0
    assert any("SQLite" in r.title or "SQLite" in r.snippet for r in results)


def test_get_web_search_provider():
    p1 = get_web_search_provider("mock")
    assert isinstance(p1, MockSearchProvider)

    p2 = get_web_search_provider()
    assert p2 is not None
