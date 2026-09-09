"""
Libra API v1 - Multi-Modal Document Understanding & OCR Pipeline (LibraOCR)
Provides endpoints for document structure parsing, layout-aware chunking,
and grounded visual question answering with table/cell citations.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from packages.core.document.document_qa import DocumentQAEngine
from packages.core.document.layout_chunker import LayoutAwareChunker
from packages.core.document.layout_parser import DocumentLayoutParser

router = APIRouter(prefix="/document", tags=["Document Understanding & OCR"])

parser = DocumentLayoutParser()
qa_engine = DocumentQAEngine()

DOCUMENT_PRESETS = [
    {
        "id": "financial_quarterly_report",
        "title": "Libra Corp - Q3 Consolidated Financial Statement",
        "description": "Enterprise financial earnings report with income statement table, profit formulas, and auditor key-values.",
        "text": """# Libra Technologies Inc. - Q3 Financial Report
Document Type: Unaudited Quarterly Statement
Fiscal Period: Q3 2026
Reporting Currency: USD ($)
Auditor Verdict: Unqualified Clean Opinion

The following financial statement summarizes revenue, operating expenses, and consolidated net income for the fiscal quarter ending September 30, 2026.

## Consolidated Statements of Operations
| Financial Metric | Q3 2025 | Q3 2026 | YoY Growth |
| Total Cloud Revenue | $45,200,000 | $62,800,000 | +38.9% |
| Research & Development | $12,400,000 | $15,100,000 | +21.8% |
| Sales & Marketing | $8,500,000 | $9,200,000 | +8.2% |
| Operating Income | $24,300,000 | $38,500,000 | +58.4% |
| Net Income | $19,800,000 | $31,200,000 | +57.6% |

Operating margin expansion was driven by architectural zero-cost efficiencies and CPU optimization across local inference pipelines.

## Financial Ratio Formulation
The operating efficiency ratio E is computed as:
$$
E = \\frac{\\text{Operating Income}}{\\text{Total Cloud Revenue}} \\times 100\\%
$$

For Q3 2026, the operating margin reached 61.3%, up from 53.8% in the prior year period.
""",
    },
    {
        "id": "transformer_research_paper",
        "title": "Scalable Autoregressive Modeling with Dynamic Binning",
        "description": "Scientific ML paper with abstract, multi-head attention equations, and hyperparameter benchmark tables.",
        "text": """# Scalable Autoregressive Modeling with Dynamic Sequence Binning
Authors: Libra Research Collective
Subject Area: Generative Foundation Models & Efficient Systems
License: Open Educational CC-BY-4.0

## Abstract
Autoregressive decoder-only language models suffer from severe quadratic attention waste when variable-length sequences are padded into uniform batch matrices. We introduce dynamic length binning and left-padding for high-throughput batching.

## Mathematical Formulation
Standard multi-head attention projects queries, keys, and values across H distinct heads:
$$
\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_H) W^O
$$

where each attention head is computed via scaled dot-product attention:
$$
\\text{head}_i = \\text{Softmax}\\left(\\frac{Q_i K_i^T}{\\sqrt{d_k}}\\right) V_i
$$

## Architectural Hyperparameters
| Model Tier | Hidden Dim (d_model) | Layers | Attention Heads | Context Window |
| Tiny | 128 | 4 | 4 | 256 |
| Small | 256 | 6 | 8 | 512 |
| Medium | 512 | 12 | 8 | 1024 |
| Large | 1024 | 24 | 16 | 2048 |

Our empirical results indicate that dynamic sequence binning reduces FLOP waste from 68.4% down to 8.2% without degrading generation quality.
""",
    },
    {
        "id": "commercial_tax_invoice",
        "title": "Commercial B2B Service Invoice #INV-8842",
        "description": "Standard business invoice with customer billing info, line-item pricing table, and tax calculations.",
        "text": """# Commercial Service Invoice
Invoice Number: INV-8842
Issue Date: 2026-09-09
Payment Due Date: 2026-10-09
Vendor Name: Libra Infrastructure Ltd
Client Name: Horizon Cognitive AI Corp

## Itemized Services
| Item # | Description | Quantity | Unit Rate | Line Total |
| 01 | PagedAttention Cache Optimization Module | 1 | $3,500.00 | $3,500.00 |
| 02 | First-Principles BPE Tokenizer Engine | 1 | $2,200.00 | $2,200.00 |
| 03 | High-Throughput Batch Job Scheduler | 1 | $4,800.00 | $4,800.00 |
| 04 | Zero-Cost CPU Hardware Tuning | 1 | $1,500.00 | $1,500.00 |

## Payment Summary
Subtotal Amount: $12,000.00
Applicable Sales Tax (8.5%): $1,020.00
Grand Total Due: $13,020.00

Payment Terms: Net 30 days via direct wire transfer.
""",
    },
]


class DocumentParseRequest(BaseModel):
    text: str = Field(
        ..., min_length=5, description="Document content in markdown, plaintext, or page breaks"
    )
    title: Optional[str] = Field(None, description="Optional document title override")
    doc_id: Optional[str] = Field(None, description="Optional custom document ID")


class DocumentChunkRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Document text to parse and chunk")
    max_tokens_per_chunk: int = Field(350, ge=50, le=2000)
    overlap_tokens: int = Field(40, ge=0, le=500)


class DocumentQARequest(BaseModel):
    text: str = Field(..., min_length=5, description="Document content to ground answers on")
    query: str = Field(..., min_length=2, description="Inquiry to answer from document elements")


@router.get("/presets")
async def get_document_presets() -> dict[str, Any]:
    """Returns benchmark document presets for testing document understanding."""
    return {"presets": DOCUMENT_PRESETS}


@router.post("/parse")
async def parse_document(req: DocumentParseRequest) -> dict[str, Any]:
    """
    Parses document text into structured AST elements (Headings, Tables, Equations, Key-Values)
    with normalized 2D spatial bounding boxes.
    """
    parsed = parser.parse_document_text(
        raw_text=req.text,
        doc_id=req.doc_id,
        title=req.title or "Parsed Document",
    )
    return {
        "status": "success",
        "document": parsed.to_dict(),
    }


@router.post("/chunk")
async def chunk_document(req: DocumentChunkRequest) -> dict[str, Any]:
    """
    Chunks visual document layout ASTs while preserving table rows, formulas, and breadcrumbs.
    """
    parsed = parser.parse_document_text(raw_text=req.text)
    chunker = LayoutAwareChunker(
        max_tokens_per_chunk=req.max_tokens_per_chunk,
        overlap_tokens=req.overlap_tokens,
    )
    chunks = chunker.chunk_document(parsed)
    return {
        "status": "success",
        "num_chunks": len(chunks),
        "chunks": [c.to_dict() for c in chunks],
    }


@router.post("/qa")
async def document_grounded_qa(req: DocumentQARequest) -> dict[str, Any]:
    """
    Answers inquiries grounded directly on document elements, providing exact table row,
    cell, and visual bounding box citations.
    """
    parsed = parser.parse_document_text(raw_text=req.text)
    result = qa_engine.answer_query(parsed, req.query)
    return {
        "status": "success",
        "result": result.to_dict(),
    }
