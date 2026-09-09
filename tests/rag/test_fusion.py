"""
Tests for Search Fusion Algorithms (Reciprocal Rank Fusion and Weighted Score Fusion)
"""

from packages.rag.fusion import reciprocal_rank_fusion, weighted_score_fusion
from packages.rag.models import DocumentChunk, SearchResult


def _make_result(cid: str, score: float, rank: int, mode: str = "dense") -> SearchResult:
    chunk = DocumentChunk(
        id=cid,
        doc_id="doc_test",
        doc_title="Test Doc",
        text=f"Text for chunk {cid}",
    )
    res = SearchResult(
        chunk=chunk,
        score=score,
        rank=rank,
        retrieval_mode=mode,
    )
    if mode == "dense":
        res.dense_score = score
        res.dense_rank = rank
    elif mode == "bm25":
        res.bm25_score = score
        res.bm25_rank = rank
    return res


def test_rrf_both_systems():
    dense_results = [
        _make_result("c1", 0.90, 1, mode="dense"),
        _make_result("c2", 0.80, 2, mode="dense"),
        _make_result("c3", 0.70, 3, mode="dense"),
    ]
    bm25_results = [
        _make_result("c2", 4.5, 1, mode="bm25"),
        _make_result("c4", 3.2, 2, mode="bm25"),
        _make_result("c1", 2.1, 3, mode="bm25"),
    ]

    fused = reciprocal_rank_fusion([dense_results, bm25_results], k=60, top_k=5)
    assert len(fused) == 4

    # c2 is rank 2 in dense and rank 1 in bm25 -> 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    # c1 is rank 1 in dense and rank 3 in bm25 -> 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
    # So c2 should be #1, followed closely by c1!
    assert fused[0].chunk.id == "c2"
    assert fused[1].chunk.id == "c1"
    assert fused[0].rrf_score is not None
    assert fused[0].rank == 1
    assert fused[1].rank == 2


def test_rrf_empty_inputs():
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[]], k=60) == []


def test_weighted_score_fusion():
    dense_results = [
        _make_result("c1", 0.90, 1, mode="dense"),
        _make_result("c2", 0.30, 2, mode="dense"),
    ]
    bm25_results = [
        _make_result("c2", 10.0, 1, mode="bm25"),
        _make_result("c1", 1.0, 2, mode="bm25"),
    ]

    # When alpha = 1.0 (all dense), c1 must win
    dense_fused = weighted_score_fusion(dense_results, bm25_results, alpha=1.0)
    assert dense_fused[0].chunk.id == "c1"

    # When alpha = 0.0 (all bm25), c2 must win
    bm25_fused = weighted_score_fusion(dense_results, bm25_results, alpha=0.0)
    assert bm25_fused[0].chunk.id == "c2"
