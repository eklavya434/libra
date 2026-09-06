"""
Libra RAG (Retrieval-Augmented Generation) Package

First-principles vector retrieval, document chunking, and context grounding.
"""

from packages.rag.bm25 import BM25Index, tokenize_bm25
from packages.rag.chunking import RecursiveCharacterChunker
from packages.rag.deduplication import ChunkDeduplicator, jaccard_similarity
from packages.rag.embeddings import (
    BaseEmbeddingProvider,
    EducationalDenseEmbedder,
    OllamaEmbeddingProvider,
    get_embedding_provider,
)
from packages.rag.fusion import reciprocal_rank_fusion, weighted_score_fusion
from packages.rag.hybrid import HybridRetriever, get_hybrid_retriever
from packages.rag.models import (
    Document,
    DocumentChunk,
    IngestDocumentRequest,
    RAGQueryRequest,
    RAGQueryResponse,
    SearchResult,
)
from packages.rag.reranker import BaseReRanker, HeuristicReRanker
from packages.rag.synthesizer import RAGPromptSynthesizer
from packages.rag.vector_store import InMemoryVectorStore, get_vector_store

__all__ = [
    "Document",
    "DocumentChunk",
    "SearchResult",
    "RAGQueryRequest",
    "RAGQueryResponse",
    "IngestDocumentRequest",
    "RecursiveCharacterChunker",
    "BaseEmbeddingProvider",
    "EducationalDenseEmbedder",
    "OllamaEmbeddingProvider",
    "get_embedding_provider",
    "InMemoryVectorStore",
    "get_vector_store",
    "RAGPromptSynthesizer",
    "BM25Index",
    "tokenize_bm25",
    "reciprocal_rank_fusion",
    "weighted_score_fusion",
    "BaseReRanker",
    "HeuristicReRanker",
    "ChunkDeduplicator",
    "jaccard_similarity",
    "HybridRetriever",
    "get_hybrid_retriever",
]
