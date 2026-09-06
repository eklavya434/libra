"""
Libra RAG Package - Search Fusion Algorithms

Implements:
1. Reciprocal Rank Fusion (RRF):
     RRF(d) = sum_{m in M} 1 / (k + r_m(d))
2. Linear Weighted Score Fusion:
     Score(d) = alpha * S_dense_norm(d) + (1 - alpha) * S_bm25_norm(d)
"""

from __future__ import annotations

from typing import Optional
from packages.rag.models import DocumentChunk, SearchResult


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    k: int = 60,
    top_k: int = 5,
) -> list[SearchResult]:
    """
    Combines multiple ranked lists of SearchResults using Reciprocal Rank Fusion.

    Args:
        ranked_lists: A list of result lists from different retrieval engines (e.g. [dense, bm25]).
        k: Smoothing constant to calibrate the impact of high vs low ranks (default 60).
        top_k: Maximum number of merged results to return.

    Returns:
        Consolidated and re-ranked list of SearchResults with RRF scoring metadata.
    """
    if not ranked_lists:
        return []

    # Map chunk_id -> {rrf_score, chunk, dense_score, dense_rank, bm25_score, bm25_rank}
    fused_candidates: dict[str, dict] = {}

    for system_idx, rlist in enumerate(ranked_lists):
        for rank_idx, result in enumerate(rlist, start=1):
            cid = result.chunk.id
            if cid not in fused_candidates:
                fused_candidates[cid] = {
                    "chunk": result.chunk,
                    "rrf_score": 0.0,
                    "dense_score": None,
                    "dense_rank": None,
                    "bm25_score": None,
                    "bm25_rank": None,
                }

            # Reciprocal rank increment: 1 / (k + rank)
            rrf_increment = 1.0 / (k + rank_idx)
            fused_candidates[cid]["rrf_score"] += rrf_increment

            if result.retrieval_mode == "dense" or result.dense_score is not None:
                fused_candidates[cid]["dense_score"] = result.dense_score or result.score
                fused_candidates[cid]["dense_rank"] = rank_idx
            if result.retrieval_mode == "bm25" or result.bm25_score is not None:
                fused_candidates[cid]["bm25_score"] = result.bm25_score or result.score
                fused_candidates[cid]["bm25_rank"] = rank_idx

    # Sort descending by rrf_score
    sorted_items = sorted(
        fused_candidates.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )[:top_k]

    results: list[SearchResult] = []
    for final_rank, item in enumerate(sorted_items, start=1):
        results.append(
            SearchResult(
                chunk=item["chunk"],
                score=float(round(item["rrf_score"], 6)),
                rank=final_rank,
                dense_score=item["dense_score"],
                dense_rank=item["dense_rank"],
                bm25_score=item["bm25_score"],
                bm25_rank=item["bm25_rank"],
                rrf_score=float(round(item["rrf_score"], 6)),
                retrieval_mode="hybrid",
            )
        )

    return results


def weighted_score_fusion(
    dense_results: list[SearchResult],
    bm25_results: list[SearchResult],
    alpha: float = 0.5,
    top_k: int = 5,
) -> list[SearchResult]:
    """
    Combines dense and sparse results using min-max normalized weighted linear interpolation.
    Score = alpha * S_dense_norm + (1 - alpha) * S_bm25_norm
    """
    def _normalize(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        s_min = min(scores.values())
        s_max = max(scores.values())
        span = s_max - s_min
        if span <= 1e-7:
            return {k: 1.0 if s_max > 0 else 0.0 for k in scores}
        return {k: (v - s_min) / span for k, v in scores.items()}

    dense_dict = {r.chunk.id: (r.dense_score if r.dense_score is not None else r.score) for r in dense_results}
    bm25_dict = {r.chunk.id: (r.bm25_score if r.bm25_score is not None else r.score) for r in bm25_results}

    dense_ranks = {r.chunk.id: r.rank for r in dense_results}
    bm25_ranks = {r.chunk.id: r.rank for r in bm25_results}

    all_chunks: dict[str, DocumentChunk] = {}
    for r in dense_results:
        all_chunks[r.chunk.id] = r.chunk
    for r in bm25_results:
        all_chunks[r.chunk.id] = r.chunk

    dense_norm = _normalize(dense_dict)
    bm25_norm = _normalize(bm25_dict)

    combined_scores: dict[str, float] = {}
    for cid in all_chunks:
        d_val = dense_norm.get(cid, 0.0)
        b_val = bm25_norm.get(cid, 0.0)
        combined_scores[cid] = (alpha * d_val) + ((1.0 - alpha) * b_val)

    sorted_cids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results: list[SearchResult] = []
    for rank_idx, (cid, score) in enumerate(sorted_cids, start=1):
        results.append(
            SearchResult(
                chunk=all_chunks[cid],
                score=float(round(score, 4)),
                rank=rank_idx,
                dense_score=dense_dict.get(cid),
                dense_rank=dense_ranks.get(cid),
                bm25_score=bm25_dict.get(cid),
                bm25_rank=bm25_ranks.get(cid),
                retrieval_mode="hybrid",
            )
        )

    return results
