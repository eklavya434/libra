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
    get_vector_store,
)

router = APIRouter()


@router.get("/documents", response_model=list[Document], summary="List all indexed documents")
async def list_documents() -> list[Document]:
    """Retrieve all indexed documents with chunk statistics."""
    store = get_vector_store()
    return store.list_documents()


@router.post("/documents", response_model=Document, summary="Ingest and index a document")
async def ingest_document(request: IngestDocumentRequest) -> Document:
    """Ingest a text document, segment it into chunks, and compute vector embeddings."""
    store = get_vector_store()
    doc, _ = store.add_document(
        title=request.title,
        content=request.content,
        metadata=request.metadata,
    )
    return doc


@router.get("/documents/{doc_id}", response_model=Document, summary="Get document details")
async def get_document(doc_id: str) -> Document:
    """Fetch an indexed document by ID."""
    store = get_vector_store()
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return doc


@router.delete("/documents/{doc_id}", summary="Delete document and purge vectors")
async def delete_document(doc_id: str) -> dict[str, Any]:
    """Delete a document and purge all its chunk embeddings from vector memory."""
    store = get_vector_store()
    success = store.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return {"status": "deleted", "id": doc_id}


@router.post("/query", response_model=RAGQueryResponse, summary="Semantic vector search & prompt synthesis")
async def query_vector_store(request: RAGQueryRequest) -> RAGQueryResponse:
    """Execute cosine similarity search and return top-k chunks with augmented prompt."""
    store = get_vector_store()
    results = store.similarity_search(
        query=request.query,
        top_k=request.top_k,
        min_score=request.min_score,
    )

    synthesizer = RAGPromptSynthesizer()
    augmented_prompt = synthesizer.build_grounded_prompt(request.query, results)

    return RAGQueryResponse(
        query=request.query,
        results=results,
        augmented_prompt=augmented_prompt,
    )
