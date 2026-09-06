"""
Libra RAG (Retrieval-Augmented Generation) Package

First-principles vector retrieval, document chunking, and context grounding.
"""

from packages.rag.chunking import RecursiveCharacterChunker
from packages.rag.embeddings import (
    BaseEmbeddingProvider,
    EducationalDenseEmbedder,
    OllamaEmbeddingProvider,
    get_embedding_provider,
)
from packages.rag.models import (
    Document,
    DocumentChunk,
    IngestDocumentRequest,
    RAGQueryRequest,
    RAGQueryResponse,
    SearchResult,
)
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
]
