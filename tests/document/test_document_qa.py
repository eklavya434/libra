"""
Tests for DocumentQAEngine (packages/core/document/document_qa.py)
"""

from packages.core.document.document_qa import DocumentQAEngine
from packages.core.document.layout_parser import DocumentLayoutParser


def test_document_qa_extracts_table_cell_citation():
    doc_text = """# Company Revenue Overview
| Year | Product | Revenue |
| 2024 | Cloud Hosting | $12,000,000 |
| 2025 | AI Assistant | $45,000,000 |
| 2026 | Enterprise Suite | $78,000,000 |
"""
    parser = DocumentLayoutParser()
    doc = parser.parse_document_text(doc_text)
    qa = DocumentQAEngine()

    result = qa.answer_query(doc, "What was the revenue for Enterprise Suite?")
    assert "Enterprise Suite" in result.answer
    assert "$78,000,000" in result.answer
    assert len(result.citations) > 0

    citation = result.citations[0]
    assert citation.element_type == "table"
    assert citation.cell_reference is not None
    assert "$78,000,000" in citation.cell_reference
    assert citation.bbox["x_min"] >= 0.0


def test_document_qa_extracts_key_value_citation():
    doc_text = """# Commercial Billing Invoice
Invoice Number: INV-90021
Vendor: Silicon Neural Labs
Total Due: $15,400.00
"""
    parser = DocumentLayoutParser()
    doc = parser.parse_document_text(doc_text)
    qa = DocumentQAEngine()

    result = qa.answer_query(doc, "What is the Invoice Number?")
    assert "INV-90021" in result.answer
    assert len(result.citations) > 0
    assert result.citations[0].element_type == "key_value"
