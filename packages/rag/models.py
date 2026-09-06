"""
Libra RAG Package - Data Models & Schemas

Pydantic schemas for documents, chunks, similarity search results,
and prompt synthesis payloads.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DocumentChunk(BaseModel):
    """Represents a segmented chunk of a larger document."""

    id: str = Field(..., description="Unique chunk identifier")
    doc_id: str = Field(..., description="ID of the parent document")
    doc_title: str = Field(..., description="Title of the parent document")
    text: str = Field(..., description="Text content of the chunk")
    char_start: int = Field(0, description="Starting character index in parent document")
    char_end: int = Field(0, description="Ending character index in parent document")
    token_count: int = Field(0, description="Estimated token count of the chunk")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom chunk metadata")


class Document(BaseModel):
    """Represents an ingested document in the knowledge base."""

    id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Title or filename of the document")
    content: str = Field(..., description="Raw full text content of the document")
    created_at: str = Field(default_factory=utc_now_iso, description="ISO timestamp when ingested")
    chunk_count: int = Field(0, description="Number of chunks generated from this document")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom document metadata")


class SearchResult(BaseModel):
    """Represents a matched chunk with its cosine similarity score."""

    chunk: DocumentChunk = Field(..., description="The matched document chunk")
    score: float = Field(..., description="Cosine similarity score [-1.0 to 1.0]")
    rank: int = Field(..., description="1-based ranking index")


class RAGQueryRequest(BaseModel):
    """Request payload for semantic retrieval."""

    query: str = Field(..., min_length=1, description="Search query string")
    top_k: int = Field(3, ge=1, le=20, description="Maximum number of chunks to return")
    min_score: float = Field(0.0, description="Minimum cosine similarity cutoff")


class RAGQueryResponse(BaseModel):
    """Response payload containing matching chunks and augmented prompt."""

    query: str = Field(..., description="Original search query")
    results: list[SearchResult] = Field(default_factory=list, description="Top-k matching chunks")
    augmented_prompt: str = Field(..., description="Grounding prompt with injected context")


class IngestDocumentRequest(BaseModel):
    """Request payload to ingest a new document."""

    title: str = Field(..., min_length=1, description="Title of the document")
    content: str = Field(..., min_length=1, description="Text body of the document")
    metadata: Optional[dict[str, Any]] = Field(None, description="Optional metadata tags")
