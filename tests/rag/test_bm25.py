"""
Tests for Okapi BM25 Sparse Inverted Index
"""

import pytest
from packages.rag.bm25 import BM25Index, tokenize_bm25
from packages.rag.models import DocumentChunk


def test_tokenize_bm25():
    text = "Intel Core i5-12450H CPU with SwiGLU_activation and sqlite.wal!"
    tokens = tokenize_bm25(text)
    assert "intel" in tokens
    assert "core" in tokens
    assert "i5-12450h" in tokens
    assert "swiglu_activation" in tokens
    assert "sqlite.wal" in tokens


def test_bm25_empty_index():
    index = BM25Index()
    assert index.total_chunks == 0
    assert index.search("anything") == []


def test_bm25_add_and_search():
    index = BM25Index()
    chunks = [
        DocumentChunk(
            id="c1",
            doc_id="d1",
            doc_title="Doc 1",
            text="Attention is all you need for transformer architectures.",
        ),
        DocumentChunk(
            id="c2",
            doc_id="d2",
            doc_title="Doc 2",
            text="Convolutional neural networks are commonly used for computer vision.",
        ),
        DocumentChunk(
            id="c3",
            doc_id="d1",
            doc_title="Doc 1",
            text="Transformers utilize multi-head self-attention mechanisms.",
        ),
    ]
    index.add_chunks(chunks)
    assert index.total_chunks == 3

    # Search for "transformer"
    results = index.search("transformer", top_k=2)
    assert len(results) == 2
    matched_ids = [r.chunk.id for r in results]
    assert "c1" in matched_ids
    assert "c3" in matched_ids
    assert results[0].bm25_score is not None
    assert results[0].bm25_score > 0
    assert results[0].retrieval_mode == "bm25"


def test_bm25_idf_rarity():
    index = BM25Index()
    chunks = [
        DocumentChunk(id="c1", doc_id="d1", doc_title="D1", text="common common common alpha"),
        DocumentChunk(id="c2", doc_id="d2", doc_title="D2", text="common common common beta"),
        DocumentChunk(id="c3", doc_id="d3", doc_title="D3", text="common common common gamma rare_term"),
    ]
    index.add_chunks(chunks)

    idf_common = index.idf("common")
    idf_rare = index.idf("rare_term")
    # Rare term should have higher IDF than term appearing in all documents
    assert idf_rare > idf_common


def test_bm25_remove_document():
    index = BM25Index()
    chunks = [
        DocumentChunk(id="c1", doc_id="doc_a", doc_title="DA", text="apple orange banana"),
        DocumentChunk(id="c2", doc_id="doc_b", doc_title="DB", text="car bike airplane"),
    ]
    index.add_chunks(chunks)
    assert index.total_chunks == 2

    removed_count = index.remove_document("doc_a")
    assert removed_count == 1
    assert index.total_chunks == 1
    assert index.search("apple") == []
    assert len(index.search("bike")) == 1
