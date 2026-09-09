"""
Libra RAG & Search Package - Web Search Providers

Supports:
1. DuckDuckGoSearchProvider: Free, zero-API-key web search over HTTPS.
2. MockSearchProvider: Fast, deterministic offline fallback for air-gapped environments.
"""

from __future__ import annotations

import re
import urllib.parse
from abc import ABC, abstractmethod
from typing import Optional

import httpx
from pydantic import BaseModel, Field


class WebSearchResult(BaseModel):
    """Represents a single web search result item."""

    title: str = Field(..., description="Title of the web page")
    url: str = Field(..., description="Canonical URL of the source")
    snippet: str = Field(..., description="Summary or text snippet")
    source: str = Field("web", description="Origin provider (e.g. duckduckgo, mock)")


class BaseWebSearchProvider(ABC):
    """Abstract interface for web search providers."""

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        """Asynchronously search the web for query."""
        pass

    def search_sync(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        """Synchronous wrapper for search."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.search(query, max_results))

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, self.search(query, max_results)).result()


class MockSearchProvider(BaseWebSearchProvider):
    """
    Deterministic offline mock search provider.
    Zero network egress, 100% free, ideal for CI and air-gapped development.
    """

    def __init__(self) -> None:
        self.knowledge_base = [
            {
                "keywords": ["transformer", "attention", "rope", "swiglu"],
                "title": "Modern Transformer Architectures & Attention Innovations",
                "url": "https://arxiv.org/abs/1706.03762",
                "snippet": "Transformer architectures utilize Scaled Dot-Product Attention, Rotary Position Embeddings (RoPE), and SwiGLU activation functions for efficient autoregressive language modeling.",
            },
            {
                "keywords": ["sqlite", "wal", "database", "concurrency"],
                "title": "SQLite Write-Ahead Logging (WAL) Technical Specification",
                "url": "https://www.sqlite.org/wal.html",
                "snippet": "SQLite WAL mode improves concurrency by allowing readers to query the database concurrently while writers append changes to the separate WAL log file.",
            },
            {
                "keywords": ["bm25", "rag", "hybrid", "search", "rrf"],
                "title": "Hybrid Search: Combining BM25 Lexical and Dense Vector Retrieval",
                "url": "https://en.wikipedia.org/wiki/Okapi_BM25",
                "snippet": "Reciprocal Rank Fusion (RRF) combines BM25 sparse search with dense embeddings to balance exact technical keyword matching with deep semantic recall.",
            },
            {
                "keywords": ["cpu", "intel", "i5", "avx2", "simd"],
                "title": "Intel Core Architecture & Vector Extensions Optimization",
                "url": "https://software.intel.com/content/www/us/en/develop/articles/avx2-optimization.html",
                "snippet": "AVX2 SIMD instructions on modern Intel Core processors accelerate vectorized dot-product computations for in-memory neural network inference without dedicated GPUs.",
            },
            {
                "keywords": ["deep", "research", "agent", "planning"],
                "title": "Autonomous Research Agents: Multi-Step Decomposition and Synthesis",
                "url": "https://openai.com/research/deep-research",
                "snippet": "Deep research agents recursively decompose complex topics into targeted sub-queries, gather and verify multi-source citations, and compile comprehensive structured reports.",
            },
        ]

    async def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        q_lower = query.lower()
        results: list[WebSearchResult] = []

        for item in self.knowledge_base:
            match_score = sum(1 for kw in item["keywords"] if kw in q_lower)
            if match_score > 0:
                results.append(
                    WebSearchResult(
                        title=item["title"],
                        url=item["url"],
                        snippet=item["snippet"],
                        source="mock",
                    )
                )

        # If no specific keyword matched, return general knowledge hits
        if not results:
            for item in self.knowledge_base[:max_results]:
                results.append(
                    WebSearchResult(
                        title=f"{item['title']} - Search for '{query}'",
                        url=item["url"],
                        snippet=f"Relevant research regarding {query}: {item['snippet']}",
                        source="mock",
                    )
                )

        return results[:max_results]


class DuckDuckGoSearchProvider(BaseWebSearchProvider):
    """
    Live web search provider using DuckDuckGo.
    Free, no API key required, with automatic fallback to MockSearchProvider.
    """

    def __init__(self, timeout_sec: float = 6.0) -> None:
        self.timeout_sec = timeout_sec
        self.fallback = MockSearchProvider()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }

    async def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        if not query.strip():
            return []

        # First attempt DuckDuckGo instant answers API
        try:
            encoded_query = urllib.parse.quote_plus(query.strip())
            api_url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"

            async with httpx.AsyncClient(timeout=self.timeout_sec, follow_redirects=True) as client:
                res = await client.get(api_url, headers=self.headers)
                if res.status_code == 200:
                    data = res.json()
                    results: list[WebSearchResult] = []

                    # Abstract text
                    abstract = data.get("AbstractText", "")
                    abstract_url = data.get("AbstractURL", "")
                    heading = data.get("Heading", query)
                    if abstract and abstract_url:
                        results.append(
                            WebSearchResult(
                                title=heading,
                                url=abstract_url,
                                snippet=abstract,
                                source="duckduckgo",
                            )
                        )

                    # Related topics
                    for topic in data.get("RelatedTopics", []):
                        if isinstance(topic, dict) and "Text" in topic and "FirstURL" in topic:
                            results.append(
                                WebSearchResult(
                                    title=topic["Text"].split(" - ")[0]
                                    if " - " in topic["Text"]
                                    else heading,
                                    url=topic["FirstURL"],
                                    snippet=topic["Text"],
                                    source="duckduckgo",
                                )
                            )
                        if len(results) >= max_results:
                            break

                    if results:
                        return results[:max_results]
        except Exception:
            pass

        # Second attempt: HTML search scraping
        try:
            html_url = (
                f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(query.strip())}"
            )
            async with httpx.AsyncClient(timeout=self.timeout_sec, follow_redirects=True) as client:
                res = await client.get(html_url, headers=self.headers)
                if res.status_code == 200:
                    html = res.text
                    results = self._parse_ddg_html(html, max_results)
                    if results:
                        return results
        except Exception:
            pass

        # Fallback to deterministic mock provider if offline or blocked
        return await self.fallback.search(query, max_results=max_results)

    def _parse_ddg_html(self, html: str, max_results: int) -> list[WebSearchResult]:
        results: list[WebSearchResult] = []
        # Pattern to capture result links and snippets from DuckDuckGo HTML
        link_matches = list(
            re.finditer(r'<a class="result__url"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>', html)
        )
        snippet_matches = list(re.finditer(r'<a class="result__snippet"[^>]*>([\s\S]*?)</a>', html))

        for i in range(min(len(link_matches), len(snippet_matches), max_results)):
            url = link_matches[i].group(1).strip()
            # Unescape DDG redirect if present
            if "/l/?uddg=" in url:
                parsed_params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                url = parsed_params.get("uddg", [url])[0]

            title = re.sub(r"<[^>]+>", "", link_matches[i].group(2)).strip()
            snippet = re.sub(r"<[^>]+>", "", snippet_matches[i].group(1)).strip()

            if url and snippet:
                results.append(
                    WebSearchResult(
                        title=title or "Web Search Result",
                        url=url,
                        snippet=snippet,
                        source="duckduckgo",
                    )
                )

        return results


_global_web_search_provider: Optional[BaseWebSearchProvider] = None


def get_web_search_provider(provider_type: Optional[str] = None) -> BaseWebSearchProvider:
    """Returns the configured Web Search provider (default: DuckDuckGo with Mock fallback)."""
    global _global_web_search_provider
    if provider_type == "mock":
        return MockSearchProvider()

    if _global_web_search_provider is None:
        _global_web_search_provider = DuckDuckGoSearchProvider()
    return _global_web_search_provider
