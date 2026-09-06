"""
Libra RAG Package - Hybrid Retriever

Unifies Dense Semantic Vector Search and Sparse BM25 Lexical Retrieval with
Reciprocal Rank Fusion (RRF), Multi-Factor Re-Ranking, and Chunk Deduplication.
"""

from __future__ import annotations

from typing import Any, Optional

from packages.rag.bm25 import BM25Index
from packages.rag.deduplication import ChunkDeduplicator
from packages.rag.fusion import reciprocal_rank_fusion, weighted_score_fusion
from packages.rag.models import Document, DocumentChunk, SearchResult
from packages.rag.reranker import BaseReRanker, HeuristicReRanker
from packages.rag.vector_store import InMemoryVectorStore, get_vector_store


class HybridRetriever:
    """
    Unified Hybrid Search Engine combining:
    1. Dense Vector Embeddings (Semantic Cosine Similarity)
    2. Okapi BM25 (Sparse Lexical Inverted Index)
    3. Reciprocal Rank Fusion / Weighted Linear Interpolation
    4. Heuristic Multi-Factor Re-Ranking
    5. Jaccard Chunk Deduplication / Diversity Filtering
    """

    def __init__(
        self,
        vector_store: Optional[InMemoryVectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        reranker: Optional[BaseReRanker] = None,
        deduplicator: Optional[ChunkDeduplicator] = None,
    ) -> None:
        self.vector_store = vector_store or get_vector_store()
        self.bm25_index = bm25_index or BM25Index()
        self.reranker = reranker or HeuristicReRanker()
        self.deduplicator = deduplicator or ChunkDeduplicator()

        # If vector store already has chunks, index them in BM25
        if self.vector_store._chunks and self.bm25_index.total_chunks == 0:
            self.bm25_index.add_chunks(self.vector_store._chunks)

    @property
    def total_documents(self) -> int:
        return self.vector_store.total_documents

    @property
    def total_chunks(self) -> int:
        return self.vector_store.total_chunks

    def add_document(
        self,
        title: str,
        content: str,
        doc_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[Document, list[DocumentChunk]]:
        """Splits, embeds in vector store, and indexes in BM25 inverted index."""
        doc, chunks = self.vector_store.add_document(
            title=title, content=content, doc_id=doc_id, metadata=metadata
        )
        if chunks:
            self.bm25_index.add_chunks(chunks)
        return doc, chunks

    def delete_document(self, doc_id: str) -> bool:
        """Purges document from both vector store and BM25 index."""
        v_ok = self.vector_store.delete_document(doc_id)
        self.bm25_index.remove_document(doc_id)
        return v_ok

    def clear(self) -> None:
        """Clears both dense and sparse storage."""
        self.vector_store.clear()
        self.bm25_index.clear()

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        alpha: float = 0.5,
        use_rrf: bool = True,
        use_reranking: bool = True,
        use_deduplication: bool = True,
        rrf_k: int = 60,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Executes search using dense, sparse, or unified hybrid retrieval with optional
        re-ranking and deduplication.
        """
        if self.total_chunks == 0 or not query.strip():
            return []

        # Candidate pool size for second-stage re-ranking
        pool_size = max(top_k * 4, 15)

        candidates: list[SearchResult] = []

        if mode == "dense":
            candidates = self.vector_store.similarity_search(
                query=query, top_k=pool_size, min_score=min_score
            )
        elif mode == "bm25":
            candidates = self.bm25_index.search(
                query=query, top_k=pool_size, min_score=min_score
            )
        else:  # hybrid
            dense_candidates = self.vector_store.similarity_search(
                query=query, top_k=pool_size, min_score=-1.0
            )
            bm25_candidates = self.bm25_index.search(
                query=query, top_k=pool_size, min_score=0.0
            )

            if not dense_candidates and not bm25_candidates:
                return []

            if use_rrf:
                candidates = reciprocal_rank_fusion(
                    ranked_lists=[dense_candidates, bm25_candidates],
                    k=rrf_k,
                    top_k=pool_size,
                )
            else:
                candidates = weighted_score_fusion(
                    dense_results=dense_candidates,
                    bm25_results=bm25_candidates,
                    alpha=alpha,
                    top_k=pool_size,
                )

        if not candidates:
            return []

        # Optional Stage 2: Re-ranking
        if use_reranking and self.reranker is not None:
            candidates = self.reranker.rerank(
                query=query, results=candidates, top_k=pool_size
            )

        # Optional Stage 3: Deduplication
        if use_deduplication and self.deduplicator is not None:
            candidates = self.deduplicator.deduplicate(
                results=candidates, top_k=top_k
            )
        else:
            candidates = candidates[:top_k]

        # Filter by min_score if applicable and re-number ranks
        final_results: list[SearchResult] = []
        for idx, r in enumerate(candidates, start=1):
            if r.score >= min_score:
                r.rank = idx
                final_results.append(r)

        return final_results


_global_hybrid_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    """Singleton getter for the unified hybrid retriever."""
    global _global_hybrid_retriever
    if _global_hybrid_retriever is None:
        _global_hybrid_retriever = HybridRetriever()
    return _global_hybrid_retriever
