"""
Tests for RAG REST API & Chat Grounding Integration (apps/backend/api/v1/endpoints/rag.py)
"""

import pytest
from fastapi.testclient import TestClient

from apps.backend.main import app
from packages.rag import InMemoryVectorStore
import packages.rag.vector_store as rag_module


@pytest.fixture(autouse=True)
def isolated_vector_store(monkeypatch):
    """Overrides global vector store with isolated instance for each test."""
    test_store = InMemoryVectorStore()
    monkeypatch.setattr(rag_module, "_global_vector_store", test_store)
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
