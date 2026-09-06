"""
Tests for RAG REST API & Chat Grounding Integration (apps/backend/api/v1/endpoints/rag.py)
"""

import pytest
from fastapi.testclient import TestClient

from apps.backend.main import app
from packages.rag import InMemoryVectorStore, HybridRetriever
import packages.rag.vector_store as rag_module
import packages.rag.hybrid as hybrid_module


@pytest.fixture(autouse=True)
def isolated_vector_store(monkeypatch):
    """Overrides global vector store and hybrid retriever with isolated instances."""
    test_store = InMemoryVectorStore()
    test_retriever = HybridRetriever(vector_store=test_store)
    monkeypatch.setattr(rag_module, "_global_vector_store", test_store)
    monkeypatch.setattr(hybrid_module, "_global_hybrid_retriever", test_retriever)
    return test_store


@pytest.fixture
def client():
    return TestClient(app)


def test_ingest_and_list_documents(client):
    res = client.post(
        "/api/v1/rag/documents",
        json={
            "title": "Quantum Computing",
            "content": "Qubits exhibit superposition and quantum entanglement, enabling parallel state exploration.",
        },
    )
    assert res.status_code == 200
    doc = res.json()
    assert doc["title"] == "Quantum Computing"
    assert doc["chunk_count"] >= 1
    doc_id = doc["id"]

    list_res = client.get("/api/v1/rag/documents")
    assert list_res.status_code == 200
    docs = list_res.json()
    assert any(d["id"] == doc_id for d in docs)


def test_query_vector_store_returns_augmented_prompt(client):
    client.post(
        "/api/v1/rag/documents",
        json={
            "title": "Educational Physics",
            "content": "Special relativity asserts that the speed of light in vacuum is universal.",
        },
    )

    query_res = client.post(
        "/api/v1/rag/query",
        json={"query": "What is the speed of light in special relativity?", "top_k": 2},
    )
    assert query_res.status_code == 200
    data = query_res.json()
    assert len(data["results"]) >= 1
    assert "REFERENCE CONTEXT" in data["augmented_prompt"]
    assert "Educational Physics" in data["augmented_prompt"]


def test_delete_document(client):
    res = client.post(
        "/api/v1/rag/documents",
        json={"title": "To Be Deleted", "content": "Sample content to purge."},
    )
    doc_id = res.json()["id"]

    del_res = client.delete(f"/api/v1/rag/documents/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    get_res = client.get(f"/api/v1/rag/documents/{doc_id}")
    assert get_res.status_code == 404


def test_chat_with_use_rag_grounding(client):
    # Ingest a unique fact into RAG store
    client.post(
        "/api/v1/rag/documents",
        json={
            "title": "Project Libra Code",
            "content": "The secret verification passphrase for Project Libra is 'Antigravity-42'.",
        },
    )

    # Chat completion with use_rag=True
    chat_res = client.post(
        "/api/v1/chat/completions",
        json={
            "model": "libra-mock-v1",
            "messages": [{"role": "user", "content": "What is the secret verification passphrase?"}],
            "use_rag": True,
            "stream": False,
        },
    )
    assert chat_res.status_code == 200
    data = chat_res.json()
    assert "choices" in data


def test_hybrid_rag_query(client):
    client.post(
        "/api/v1/rag/documents",
        json={
            "title": "Machine Learning Hardware",
            "content": "Intel Core i5-12450H CPU operates efficiently with AVX2 vector SIMD instructions.",
        },
    )

    # Test mode="hybrid" with RRF
    res = client.post(
        "/api/v1/rag/query",
        json={
            "query": "i5-12450H AVX2 SIMD instructions",
            "mode": "hybrid",
            "use_rrf": True,
            "use_reranking": True,
            "use_deduplication": True,
            "top_k": 2,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "hybrid"
    assert len(data["results"]) >= 1
    assert data["results"][0]["chunk"]["doc_title"] == "Machine Learning Hardware"
    assert data["results"][0]["rank"] == 1

