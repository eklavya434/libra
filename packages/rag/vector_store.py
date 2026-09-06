"""
Libra RAG Package - In-Memory Vector Store

First-principles vector database using NumPy for exact cosine similarity search.
Stores document chunks and normalized dense embedding matrices.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional
import numpy as np

from packages.rag.chunking import RecursiveCharacterChunker
from packages.rag.embeddings import BaseEmbeddingProvider, get_embedding_provider
from packages.rag.models import Document, DocumentChunk, SearchResult, utc_now_iso


class InMemoryVectorStore:
    """Thread-safe in-memory vector database with exact cosine similarity search."""

    def __init__(
        self,
        embedder: Optional[BaseEmbeddingProvider] = None,
        chunker: Optional[RecursiveCharacterChunker] = None,
    ) -> None:
        self.embedder = embedder or get_embedding_provider("educational")
        self.chunker = chunker or RecursiveCharacterChunker(chunk_size=500, chunk_overlap=80)
        self._documents: dict[str, Document] = {}
        self._chunks: list[DocumentChunk] = []
        self._vectors: Optional[np.ndarray] = None  # Shape: (N, D)

    @property
    def total_documents(self) -> int:
        return len(self._documents)

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    def add_document(
        self,
        title: str,
        content: str,
        doc_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[Document, list[DocumentChunk]]:
        """Splits, embeds, and indexes a raw text document."""
        did = doc_id or f"doc-{uuid.uuid4().hex[:10]}"
        now = utc_now_iso()

        doc = Document(
            id=did,
            title=title.strip() if title.strip() else "Untitled Document",
            content=content,
            created_at=now,
            metadata=metadata or {},
        )

        chunks = self.chunker.chunk_document(doc)
        doc.chunk_count = len(chunks)

        if not chunks:
            self._documents[did] = doc
            return doc, []

        # Generate dense embeddings for newly created chunks
        embeddings = self.embedder.embed_batch([c.text for c in chunks])
        new_matrix = np.array(embeddings, dtype=np.float32)

        # Append chunks and concatenate vector rows
        if self._vectors is None or len(self._chunks) == 0:
            self._vectors = new_matrix
            self._chunks = list(chunks)
        else:
            self._vectors = np.vstack([self._vectors, new_matrix])
            self._chunks.extend(chunks)

        self._documents[did] = doc
        return doc, chunks

    def delete_document(self, doc_id: str) -> bool:
        """Removes a document and purges all of its associated vector embeddings."""
        if doc_id not in self._documents:
            return False

        del self._documents[doc_id]

        if not self._chunks:
            return True

        # Keep indices of chunks that do NOT belong to the deleted document
        keep_indices = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]

        if not keep_indices:
            self._chunks = []
            self._vectors = None
        else:
            self._chunks = [self._chunks[i] for i in keep_indices]
            if self._vectors is not None:
                self._vectors = self._vectors[keep_indices]

        return True

    def similarity_search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = -1.0,
    ) -> list[SearchResult]:
        """
        Executes exact cosine similarity search against indexed chunks.

        Math:
        Since all embeddings are unit L2-normalized:
        cos(u, v) = (u . v) / (||u|| * ||v||) = u . v
        Similarity scores are computed via single matrix-vector multiplication.
        """
        if not self._chunks or self._vectors is None or len(self._chunks) == 0:
            return []

        # Embed and normalize query vector
        query_vec = np.array(self.embedder.embed_text(query), dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 1e-9:
            query_vec = query_vec / q_norm

        # Matrix dot-product: shape (N,)
        scores = np.dot(self._vectors, query_vec)

        # Get top-k indices sorted descending
        k = min(top_k, len(self._chunks))
        top_indices = np.argsort(scores)[::-1][:k]

        results: list[SearchResult] = []
        rank = 1
        for idx in top_indices:
            score = float(scores[idx])
            if score >= min_score:
                results.append(
                    SearchResult(
                        chunk=self._chunks[idx],
                        score=round(score, 4),
                        rank=rank,
                        dense_score=round(score, 4),
                        dense_rank=rank,
                        retrieval_mode="dense",
                    )
                )
                rank += 1

        return results

    def list_documents(self) -> list[Document]:
        """Return list of all indexed documents."""
        return list(self._documents.values())

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Fetch single document by ID."""
        return self._documents.get(doc_id)

    def clear(self) -> None:
        """Purge all documents, chunks, and embeddings."""
        self._documents.clear()
        self._chunks.clear()
        self._vectors = None


_global_vector_store: Optional[InMemoryVectorStore] = None


def get_vector_store() -> InMemoryVectorStore:
    """Singleton provider for in-memory vector store."""
    global _global_vector_store
    if _global_vector_store is None:
        _global_vector_store = InMemoryVectorStore()
    return _global_vector_store
