"""
Libra API v1 - RAG & Vector Retrieval Endpoints

Provides endpoints for document ingestion, chunk management, semantic vector search,
and context grounding.
"""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, HTTPException

from packages.rag import (
    Document,
    IngestDocumentRequest,
    RAGPromptSynthesizer,
    RAGQueryRequest,
    RAGQueryResponse,
    get_hybrid_retriever,
)

router = APIRouter()


@router.get("/documents", response_model=list[Document], summary="List all indexed documents")
async def list_documents() -> list[Document]:
    """Retrieve all indexed documents with chunk statistics."""
    retriever = get_hybrid_retriever()
    return retriever.vector_store.list_documents()


@router.post("/documents", response_model=Document, summary="Ingest and index a document")
async def ingest_document(request: IngestDocumentRequest) -> Document:
    """Ingest a text document, segment into chunks, compute embeddings, and index in BM25."""
    retriever = get_hybrid_retriever()
    doc, _ = retriever.add_document(
        title=request.title,
        content=request.content,
        metadata=request.metadata,
    )
    return doc


@router.get("/documents/{doc_id}", response_model=Document, summary="Get document details")
async def get_document(doc_id: str) -> Document:
    """Fetch an indexed document by ID."""
    retriever = get_hybrid_retriever()
    doc = retriever.vector_store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return doc


@router.delete("/documents/{doc_id}", summary="Delete document and purge vectors/BM25")
async def delete_document(doc_id: str) -> dict[str, Any]:
    """Delete a document and purge all its chunks from dense and BM25 index."""
    retriever = get_hybrid_retriever()
    success = retriever.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return {"status": "deleted", "id": doc_id}


@router.post("/query", response_model=RAGQueryResponse, summary="Hybrid vector/BM25 search & prompt synthesis")
async def query_vector_store(request: RAGQueryRequest) -> RAGQueryResponse:
    """
    Execute dense, sparse (BM25), or hybrid retrieval with optional
    Reciprocal Rank Fusion, multi-factor re-ranking, and chunk deduplication.
    """
    retriever = get_hybrid_retriever()
    results = retriever.search(
        query=request.query,
        top_k=request.top_k,
        mode=request.mode,
        alpha=request.alpha,
        use_rrf=request.use_rrf,
        use_reranking=request.use_reranking,
        use_deduplication=request.use_deduplication,
        rrf_k=request.rrf_k,
        min_score=request.min_score,
    )

    synthesizer = RAGPromptSynthesizer()
    augmented_prompt = synthesizer.build_grounded_prompt(request.query, results)

    return RAGQueryResponse(
        query=request.query,
        mode=request.mode,
        results=results,
        augmented_prompt=augmented_prompt,
        total_candidates=len(results),
    )
