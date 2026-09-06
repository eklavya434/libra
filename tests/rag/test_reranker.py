"""
Tests for Multi-Factor Heuristic Re-Ranker
"""

import pytest
from packages.rag.models import DocumentChunk, SearchResult
from packages.rag.reranker import HeuristicReRanker


def test_reranker_phrase_match():
    reranker = HeuristicReRanker()
    chunk_exact = DocumentChunk(
        id="c_exact",
        doc_id="d1",
        doc_title="D1",
        text="The key innovation in transformers is multi head self attention mechanisms.",
    )
    chunk_scattered = DocumentChunk(
        id="c_scattered",
        doc_id="d2",
        doc_title="D2",
        text="Self esteem and multi tasking can draw attention to other heads in the network.",
    )

    results = [
        SearchResult(chunk=chunk_scattered, score=0.85, rank=1),
        SearchResult(chunk=chunk_exact, score=0.80, rank=2),
    ]

    # Re-ranking on "multi head self attention"
    reranked = reranker.rerank(
        query="multi head self attention",
        results=results,
        top_k=2,
    )

    assert len(reranked) == 2
    # chunk_exact has full phrase match and high coverage; it should be promoted to rank 1
    assert reranked[0].chunk.id == "c_exact"
    assert reranked[0].rerank_score is not None
    assert reranked[0].rank == 1


def test_reranker_empty_results():
    reranker = HeuristicReRanker()
    assert reranker.rerank(query="test", results=[]) == []
