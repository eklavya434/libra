"""
Tests for Document OCR & Understanding API endpoints (apps/backend/api/v1/endpoints/document_ocr.py)
"""

import pytest
from fastapi.testclient import TestClient

from apps.backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_get_document_presets(client):
    response = client.get("/api/v1/document/presets")
    assert response.status_code == 200
    data = response.json()
    assert "presets" in data
    assert len(data["presets"]) >= 3
    preset_ids = [p["id"] for p in data["presets"]]
    assert "financial_quarterly_report" in preset_ids
    assert "transformer_research_paper" in preset_ids


def test_parse_document_endpoint(client):
    payload = {
        "text": "# Test Document\nAuthor: John Doe\n\n| Col A | Col B |\n| 1 | 2 |\n",
        "title": "API Test Document",
    }
    response = client.post("/api/v1/document/parse", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    doc = data["document"]
    assert doc["title"] == "API Test Document"
    assert doc["num_pages"] == 1
    assert doc["total_elements"] >= 3


def test_chunk_document_endpoint(client):
    payload = {
        "text": "# Section Header\nParagraph text here describing the procedure.\n\n| Item | Val |\n| A | 10 |\n",
        "max_tokens_per_chunk": 200,
    }
    response = client.post("/api/v1/document/chunk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["num_chunks"] >= 1
    chunk = data["chunks"][0]
    assert "text" in chunk
    assert "breadcrumbs" in chunk
    assert "bounding_boxes" in chunk


def test_document_qa_endpoint(client):
    doc_text = "# Invoice\nDue Date: 2026-11-01\nTotal Amount: $5,000\n"
    payload = {
        "text": doc_text,
        "query": "What is the Total Amount?",
    }
    response = client.post("/api/v1/document/qa", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    result = data["result"]
    assert "$5,000" in result["answer"]
    assert len(result["citations"]) > 0
    assert result["citations"][0]["element_type"] == "key_value"
