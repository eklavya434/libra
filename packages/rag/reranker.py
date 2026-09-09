"""
Libra RAG Package - Re-Ranking Engine

Implements multi-factor cross-scoring to refine candidate chunk order:
1. Exact phrase matching bonus
2. Query keyword coverage ratio
3. Term proximity / span density
4. Initial retrieval rank prior
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from packages.rag.models import SearchResult


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in re.finditer(r"\b[a-zA-Z0-9_\-\.]+\b", text)]


class BaseReRanker(ABC):
    """Abstract interface for RAG re-rankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Re-rank candidate search results for a given query."""
        pass


class HeuristicReRanker(BaseReRanker):
    """
    Educational Multi-Factor Cross-Scoring Re-Ranker.
    Evaluates phrase match, keyword coverage, and term proximity on CPU in <1ms.
    """

    def __init__(
        self,
        weight_prior: float = 0.35,
        weight_phrase: float = 0.30,
        weight_coverage: float = 0.25,
        weight_proximity: float = 0.10,
    ):
        self.w_prior = weight_prior
        self.w_phrase = weight_phrase
        self.w_coverage = weight_coverage
        self.w_proximity = weight_proximity

    def _phrase_score(self, query_clean: str, text_clean: str) -> float:
        """Calculates exact phrase and sub-phrase matching bonus."""
        if not query_clean or not text_clean:
            return 0.0

        # Full exact query match
        if query_clean in text_clean:
            return 1.0

        q_words = query_clean.split()
        if len(q_words) <= 1:
            return 1.0 if q_words and q_words[0] in text_clean else 0.0

        # Check for matching 2-word or 3-word n-grams
        matches = 0
        total_ngrams = 0
        for n in [3, 2]:
            if len(q_words) >= n:
                for i in range(len(q_words) - n + 1):
                    ngram = " ".join(q_words[i : i + n])
                    total_ngrams += 1
                    if ngram in text_clean:
                        matches += 1

        return (matches / total_ngrams) if total_ngrams > 0 else 0.0

    def _coverage_score(self, q_tokens: set[str], doc_tokens: set[str]) -> float:
        """Calculates fraction of distinct query terms present in the document."""
        if not q_tokens:
            return 0.0
        matched = len(q_tokens.intersection(doc_tokens))
        return matched / len(q_tokens)

    def _proximity_score(self, q_tokens: set[str], doc_words: list[str]) -> float:
        """Calculates term density: how closely query words appear together."""
        if len(q_tokens) <= 1 or not doc_words:
            return 1.0

        positions: list[int] = [idx for idx, word in enumerate(doc_words) if word in q_tokens]
        if len(positions) < 2:
            return 0.0

        # Smallest span containing at least 2 distinct query tokens
        min_span = float("inf")
        for i in range(len(positions) - 1):
            span = positions[i + 1] - positions[i]
            if span < min_span:
                min_span = span

        # If adjacent words, score = 1.0; decaying as span grows up to 30 words
        return max(0.0, 1.0 - (min_span / 30.0))

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not results:
            return []

        q_clean = query.strip().lower()
        q_tokens = _tokenize(q_clean)
        q_token_set = set(q_tokens)

        # Determine min and max of initial retrieval scores for scaling
        raw_scores = [r.score for r in results]
        s_min = min(raw_scores)
        s_max = max(raw_scores)
        s_range = (s_max - s_min) if (s_max - s_min) > 1e-7 else 1.0

        scored_candidates: list[tuple[SearchResult, float]] = []

        for r in results:
            text_clean = r.chunk.text.lower()
            doc_words = _tokenize(text_clean)
            doc_token_set = set(doc_words)

            s_prior = (r.score - s_min) / s_range
            s_phrase = self._phrase_score(q_clean, text_clean)
            s_coverage = self._coverage_score(q_token_set, doc_token_set)
            s_proximity = self._proximity_score(q_token_set, doc_words)

            composite = (
                self.w_prior * s_prior
                + self.w_phrase * s_phrase
                + self.w_coverage * s_coverage
                + self.w_proximity * s_proximity
            )

            scored_candidates.append((r, composite))

        # Sort descending by composite re-ranking score
        scored_candidates.sort(key=lambda item: item[1], reverse=True)

        reranked_results: list[SearchResult] = []
        for rank_idx, (r, comp_score) in enumerate(scored_candidates[:top_k], start=1):
            reranked_results.append(
                SearchResult(
                    chunk=r.chunk,
                    score=float(round(comp_score, 4)),
                    rank=rank_idx,
                    dense_score=r.dense_score,
                    dense_rank=r.dense_rank,
                    bm25_score=r.bm25_score,
                    bm25_rank=r.bm25_rank,
                    rrf_score=r.rrf_score,
                    rerank_score=float(round(comp_score, 4)),
                    retrieval_mode=r.retrieval_mode,
                )
            )

        return reranked_results
