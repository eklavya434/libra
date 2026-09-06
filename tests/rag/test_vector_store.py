"""
Tests for In-Memory Vector Store (packages/rag/vector_store.py)
"""

import pytest
from packages.rag.chunking import RecursiveCharacterChunker
from packages.rag.embeddings import EducationalDenseEmbedder
from packages.rag.vector_store import InMemoryVectorStore


@pytest.fixture
def store():
    chunker = RecursiveCharacterChunker(chunk_size=120, chunk_overlap=20)
    embedder = EducationalDenseEmbedder(dimension=64)
    return InMemoryVectorStore(embedder=embedder, chunker=chunker)


def test_add_and_list_documents(store):
    doc1, chunks1 = store.add_document(
        title="Attention Paper",
        content="Attention Is All You Need introduced the Transformer architecture in 2017.",
    )
    assert doc1.id is not None
    assert doc1.chunk_count == len(chunks1)
    assert store.total_documents == 1
    assert store.total_chunks == len(chunks1)

    doc2, chunks2 = store.add_document(
        title="RoPE Paper",
        content="RoFormer introduced Rotary Positional Embeddings for transformers.",
    )
    assert store.total_documents == 2
    assert store.total_chunks == len(chunks1) + len(chunks2)


def test_similarity_search_ranking(store):
    store.add_document(
        title="Physics Notes",
        content="Newtonian mechanics describes the motion of macroscopic objects under forces.",
    )
    store.add_document(
        title="Deep Learning Notes",
        content="Backpropagation computes gradients of loss functions with respect to neural network weights.",
    )

    results = store.similarity_search("How are neural network gradients computed?", top_k=2)
    assert len(results) >= 1
    # Deep Learning notes should rank first
    assert results[0].chunk.doc_title == "Deep Learning Notes"
    assert results[0].score > 0.0
    assert results[0].rank == 1


def test_delete_document_purges_vectors(store):
    doc1, _ = store.add_document(title="Doc 1", content="Content of first document.")
    doc2, _ = store.add_document(title="Doc 2", content="Content of second document.")

    assert store.total_documents == 2
    initial_chunks = store.total_chunks

    success = store.delete_document(doc1.id)
    assert success is True
    assert store.total_documents == 1
    assert store.get_document(doc1.id) is None
    assert store.total_chunks < initial_chunks

    # Check search only finds remaining doc
    results = store.similarity_search("first document", top_k=5)
    for r in results:
        assert r.chunk.doc_id != doc1.id


def test_clear_store(store):
    store.add_document(title="Doc 1", content="Some test text.")
    assert store.total_documents == 1
    store.clear()
    assert store.total_documents == 0
    assert store.total_chunks == 0
    assert store.similarity_search("query") == []
