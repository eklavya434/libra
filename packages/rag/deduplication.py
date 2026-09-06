"""
Libra RAG Package - Chunk Deduplication & Diversity Filtering

Implements:
1. Lexical Jaccard Deduplication
2. Maximal Marginal Relevance (MMR) Diversification
"""

from __future__ import annotations

import re
from typing import Optional

from packages.rag.models import SearchResult


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower()))


def jaccard_similarity(text1: str, text2: str) -> float:
    """Computes token-level Jaccard similarity index between two texts: |A & B| / |A | B|."""
    set1 = _token_set(text1)
    set2 = _token_set(text2)
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


class ChunkDeduplicator:
    """
    Filters redundant and near-duplicate chunks to maximize diversity
    in the retrieved LLM context window.
    """

    def __init__(self, max_jaccard_threshold: float = 0.70, mmr_lambda: float = 0.70):
        """
        Args:
            max_jaccard_threshold: Maximum allowed lexical overlap between any two returned chunks.
            mmr_lambda: Weight for relevance vs diversity in MMR (1.0 = relevance only, 0.0 = diversity only).
        """
        self.max_jaccard_threshold = max_jaccard_threshold
        self.mmr_lambda = mmr_lambda

    def deduplicate(
        self,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Greedy threshold deduplication: iterates through ranked results and rejects
        any chunk that exceeds the Jaccard threshold with an already accepted chunk.
        """
        if not results:
            return []

        accepted: list[SearchResult] = []
        for cand in results:
            cand_text = cand.chunk.text
            is_redundant = False
            for prev in accepted:
                sim = jaccard_similarity(cand_text, prev.chunk.text)
                if sim >= self.max_jaccard_threshold:
                    is_redundant = True
                    break

            if not is_redundant:
                accepted.append(cand)
                if len(accepted) >= top_k:
                    break

        # Re-number ranks
        for idx, res in enumerate(accepted, start=1):
            res.rank = idx

        return accepted

    def mmr(
        self,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        r"""
        Maximal Marginal Relevance (MMR) reranking:
        MMR = argmax_{d in R \ S} [ lambda * Score(d) - (1 - lambda) * max_{s in S} Sim(d, s) ]
        """
        if not results or top_k <= 0:
            return []

        unselected = list(results)
        selected: list[SearchResult] = []

        # Pick the highest scoring item first
        first = unselected.pop(0)
        selected.append(first)

        while len(selected) < top_k and unselected:
            best_idx = -1
            best_mmr_score = -float("inf")

            for idx, cand in enumerate(unselected):
                # Max similarity to any already selected chunk
                max_sim = max(
                    jaccard_similarity(cand.chunk.text, sel.chunk.text)
                    for sel in selected
                )
                mmr_val = (self.mmr_lambda * cand.score) - ((1.0 - self.mmr_lambda) * max_sim)

                if mmr_val > best_mmr_score:
                    best_mmr_score = mmr_val
                    best_idx = idx

            if best_idx >= 0:
                selected.append(unselected.pop(best_idx))
            else:
                break

        # Re-index ranks
        for idx, res in enumerate(selected, start=1):
            res.rank = idx

        return selected
