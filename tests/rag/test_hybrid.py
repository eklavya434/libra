"""
Tests for HybridRetriever combining dense vector search, BM25, RRF, reranking, and deduplication
"""

from packages.rag.hybrid import HybridRetriever
from packages.rag.vector_store import InMemoryVectorStore


def test_hybrid_retriever_modes():
    store = InMemoryVectorStore()
    retriever = HybridRetriever(vector_store=store)

    # Ingest test documents
    doc1, _ = retriever.add_document(
        title="SQLite Architecture",
        content="SQLite utilizes Write-Ahead Logging (WAL) and B-Trees for zero-server relational database queries.",
    )
    doc2, _ = retriever.add_document(
        title="Transformer Attention",
        content="Transformers utilize Scaled Dot-Product Attention with queries, keys, and values to calculate attention weights.",
    )

    assert retriever.total_documents == 2
    assert retriever.total_chunks > 0

    # 1. Test BM25 exact keyword match
    bm25_res = retriever.search(query="Write-Ahead Logging", mode="bm25", top_k=2)
    assert len(bm25_res) > 0
    assert bm25_res[0].chunk.doc_title == "SQLite Architecture"
    assert bm25_res[0].retrieval_mode == "bm25"

    # 2. Test Dense semantic match
    dense_res = retriever.search(query="neural network attention mechanism", mode="dense", top_k=2)
    assert len(dense_res) > 0
    assert dense_res[0].chunk.doc_title == "Transformer Attention"
    assert dense_res[0].retrieval_mode == "dense"

    # 3. Test Hybrid search with RRF and Re-ranking
    hybrid_res = retriever.search(
        query="B-Trees database query",
        mode="hybrid",
        use_rrf=True,
        use_reranking=True,
        use_deduplication=True,
        top_k=2,
    )
    assert len(hybrid_res) > 0
    assert hybrid_res[0].chunk.doc_title == "SQLite Architecture"
    assert hybrid_res[0].rank == 1

    # 4. Test delete
    retriever.delete_document(doc1.id)
    assert retriever.total_documents == 1
    assert len(retriever.search(query="Write-Ahead Logging", mode="bm25")) == 0
