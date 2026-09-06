"""
Libra RAG Package - Okapi BM25 Inverted Index & Sparse Searcher

Mathematical Implementation from First Principles:
  IDF(q_i) = ln(1 + (N - n(q_i) + 0.5) / (n(q_i) + 0.5))
  BM25(D, Q) = sum_{q_i in Q} IDF(q_i) * (f(q_i, D) * (k_1 + 1)) / (f(q_i, D) + k_1 * (1 - b + b * (|D| / avgdl)))
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

from packages.rag.models import DocumentChunk, SearchResult


def stem_token(word: str) -> str:
    """Simple educational suffix stemmer for English inflections."""
    w = word.lower()
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]
    if w.endswith("ed") and len(w) > 4:
        return w[:-2]
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def tokenize_bm25(text: str) -> list[str]:
    """
    Tokenizes and stems text for BM25 indexing.
    Preserves alphanumeric terms, hyphens, and underscores for technical and code terms.
    """
    raw_tokens = [match.group(0).lower() for match in re.finditer(r"\b[a-zA-Z0-9_\-\.]+\b", text)]
    return [stem_token(t) for t in raw_tokens]


class BM25Index:
    """
    Educational Okapi BM25 Inverted Index for sparse lexical retrieval.
    Zero external dependencies, highly optimized for CPU.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Term frequency saturation parameter (default 1.5).
            b: Document length normalization parameter (default 0.75).
        """
        self.k1 = k1
        self.b = b

        # Primary storage
        self.chunks: dict[str, DocumentChunk] = {}
        self.chunk_lengths: dict[str, int] = {}
        self.chunk_term_freqs: dict[str, Counter[str]] = {}

        # Inverted index: term -> set of chunk_ids containing term
        self.inverted_index: dict[str, set[str]] = {}

        # Cached statistics
        self.total_tokens: int = 0
        self.avg_doc_len: float = 0.0

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Add a batch of document chunks to the BM25 inverted index."""
        for chunk in chunks:
            tokens = tokenize_bm25(chunk.text)
            doc_len = len(tokens)
            term_counts = Counter(tokens)

            self.chunks[chunk.id] = chunk
            self.chunk_lengths[chunk.id] = doc_len
            self.chunk_term_freqs[chunk.id] = term_counts
            self.total_tokens += doc_len

            # Update inverted index
            for term in term_counts:
                if term not in self.inverted_index:
                    self.inverted_index[term] = set()
                self.inverted_index[term].add(chunk.id)

        self._recompute_stats()

    def remove_document(self, doc_id: str) -> int:
        """Removes all chunks belonging to a parent document ID."""
        chunk_ids_to_remove = [
            cid for cid, chunk in self.chunks.items() if chunk.doc_id == doc_id
        ]
        for cid in chunk_ids_to_remove:
            self._remove_chunk(cid)

        self._recompute_stats()
        return len(chunk_ids_to_remove)

    def _remove_chunk(self, chunk_id: str) -> None:
        """Internal helper to safely remove a single chunk."""
        if chunk_id not in self.chunks:
            return

        doc_len = self.chunk_lengths.pop(chunk_id, 0)
        self.total_tokens = max(0, self.total_tokens - doc_len)
        term_counts = self.chunk_term_freqs.pop(chunk_id, Counter())
        self.chunks.pop(chunk_id, None)

        for term in term_counts:
            if term in self.inverted_index:
                self.inverted_index[term].discard(chunk_id)
                if not self.inverted_index[term]:
                    del self.inverted_index[term]

    def clear(self) -> None:
        """Clears all indexed data."""
        self.chunks.clear()
        self.chunk_lengths.clear()
        self.chunk_term_freqs.clear()
        self.inverted_index.clear()
        self.total_tokens = 0
        self.avg_doc_len = 0.0

    def _recompute_stats(self) -> None:
        """Recomputes global statistics such as average document length."""
        n = len(self.chunks)
        self.avg_doc_len = (self.total_tokens / n) if n > 0 else 0.0

    def idf(self, term: str) -> float:
        """
        Computes Robertson-Spärck Jones Inverse Document Frequency with smoothing.
        IDF(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))
        """
        n = len(self.chunks)
        if n == 0:
            return 0.0

        doc_freq = len(self.inverted_index.get(term, set()))
        if doc_freq == 0:
            return 0.0

        numerator = n - doc_freq + 0.5
        denominator = doc_freq + 0.5
        return math.log(1.0 + (numerator / denominator))

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Performs Okapi BM25 scoring across indexed chunks.
        Returns top-k ranked SearchResults.
        """
        if not self.chunks or not query.strip():
            return []

        query_tokens = tokenize_bm25(query)
        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)
        n = len(self.chunks)
        avgdl = self.avg_doc_len if self.avg_doc_len > 0 else 1.0

        # Candidate accumulation
        candidate_ids: set[str] = set()
        for term in query_counts:
            candidate_ids.update(self.inverted_index.get(term, set()))

        if not candidate_ids:
            return []

        scores: dict[str, float] = {}
        for chunk_id in candidate_ids:
            chunk_len = self.chunk_lengths[chunk_id]
            term_freqs = self.chunk_term_freqs[chunk_id]

            doc_score = 0.0
            for term, q_tf in query_counts.items():
                f = term_freqs.get(term, 0)
                if f == 0:
                    continue

                term_idf = self.idf(term)

                # Okapi BM25 term weighting
                tf_norm = f * (self.k1 + 1.0) / (
                    f + self.k1 * (1.0 - self.b + self.b * (chunk_len / avgdl))
                )
                doc_score += term_idf * tf_norm

            if doc_score >= min_score:
                scores[chunk_id] = doc_score

        if not scores:
            return []

        sorted_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        results: list[SearchResult] = []
        for rank_idx, (chunk_id, score) in enumerate(sorted_results, start=1):
            chunk = self.chunks[chunk_id]
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(round(score, 4)),
                    rank=rank_idx,
                    bm25_score=float(round(score, 4)),
                    bm25_rank=rank_idx,
                    retrieval_mode="bm25",
                )
            )

        return results
