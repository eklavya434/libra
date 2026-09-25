"""
Tests for Document OCR & Understanding API endpoints (apps/backend/api/v1/endpoints/document_ocr.py)
"""

import pytest
from fastapi.testclient import TestClient

from apps.backend.api.v1.endpoints import document_ocr as document_endpoint
from apps.backend.main import app
from packages.core.storage import MemoryObjectStore


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def memory_storage(monkeypatch):
    """Route document uploads to an explicit in-memory object store for tests."""
    store = MemoryObjectStore()
    monkeypatch.setattr(document_endpoint, "get_object_store", lambda: store)
    return store


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


def test_upload_txt_document_endpoint(client):
    content = "# Meeting Notes\nDate: 2026-09-13\n\n| Topic | Owner |\n| Sprint plan | Priya |\n"
    response = client.post(
        "/api/v1/document/upload",
        files={"file": ("meeting.md", content.encode("utf-8"), "text/markdown")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "meeting.md"
    assert data["source"] == "upload"
    doc = data["document"]
    assert doc["title"] == "meeting"
    assert doc["num_pages"] == 1
    assert doc["total_elements"] >= 2


def test_upload_persists_stored_object_and_serves_it_back(memory_storage, client):
    content = "# Stored Plan\n| Step | Time |\n| A | 1h |\n"
    response = client.post(
        "/api/v1/document/upload",
        files={"file": ("plan.md", content.encode("utf-8"), "text/markdown")},
    )
    assert response.status_code == 200
    data = response.json()
    storage = data["storage"]
    assert storage["backend"] == "memory"
    assert storage["key"] is not None
    assert storage["key"].startswith("documents/")
    assert memory_storage.exists(storage["key"])

    served = client.get(f"/api/v1/document/files/{storage['key']}")
    assert served.status_code == 200
    assert served.content == content.encode("utf-8")


def test_files_endpoint_rejects_non_document_keys(client):
    response = client.get("/api/v1/document/files/not-a-document/x.bin")
    assert response.status_code == 404


def test_files_endpoint_404_for_missing_object(memory_storage, client):
    response = client.get("/api/v1/document/files/documents/000000000000/missing.bin")
    assert response.status_code == 404


def test_upload_image_returns_501(client):
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    response = client.post(
        "/api/v1/document/upload",
        files={"file": ("scan.png", png_bytes, "image/png")},
    )
    assert response.status_code == 501
    assert "not supported" in response.json()["detail"]


def test_upload_pdf_document_endpoint(client):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    import io

    buf = io.BytesIO()
    writer.write(buf)
    buf.seek(0)

    # TestClient encodes the file through starlette/tempfile multipart handling.
    response = client.post(
        "/api/v1/document/upload",
        files={"file": ("blank.pdf", buf.getvalue(), "application/pdf")},
    )
    # A blank page has no text -> should be rejected as unparseable, not crash.
    assert response.status_code == 422
