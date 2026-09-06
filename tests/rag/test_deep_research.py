"""
Tests for Deep Research Agent Workflow
"""

import pytest
from packages.rag.deep_research import DeepResearchAgent, DeepResearchReport
from packages.rag.web_search import MockSearchProvider
from packages.rag.vector_store import InMemoryVectorStore
from packages.rag.hybrid import HybridRetriever


@pytest.mark.anyio
async def test_deep_research_agent_workflow():
    mock_search = MockSearchProvider()
    store = InMemoryVectorStore()
    retriever = HybridRetriever(vector_store=store)

    # Ingest a local reference document
    retriever.add_document(
        title="Local Engineering Guide",
        content="Intel Core i5-12450H CPU operates efficiently with AVX2 vector SIMD instructions for local neural network inference.",
    )

    agent = DeepResearchAgent(search_provider=mock_search, retriever=retriever)

    # Run deep research on a technical query
    report: DeepResearchReport = await agent.execute_research(
        topic="Modern Transformer Architectures and Hardware Sizing",
        max_iterations=2,
        max_sources_per_query=2,
    )

    assert report.topic == "Modern Transformer Architectures and Hardware Sizing"
    assert report.iterations_completed == 2
    assert len(report.sections) == 2
    assert len(report.sources) > 0

    # Verify report formatting
    assert "# Deep Research Dossier" in report.markdown_report
    assert "## 1. Executive Summary" in report.markdown_report
    assert "## 4. Source Index & Citations" in report.markdown_report
    assert report.executive_summary != ""
