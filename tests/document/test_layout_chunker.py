"""
Tests for LayoutAwareChunker (packages/core/document/layout_chunker.py)
"""

from packages.core.document.layout_chunker import LayoutAwareChunker
from packages.core.document.layout_parser import DocumentLayoutParser


def test_layout_aware_chunker_preserves_table():
    doc_text = """# Machine Learning Report
## Model Evaluation
This section outlines benchmark evaluations across four distinct LLM model architectures.

| Model Tier | Layers | Heads | Benchmark Score |
| Tiny | 4 | 4 | 72.4% |
| Small | 6 | 8 | 81.2% |
| Medium | 12 | 8 | 88.5% |

Concluding remarks on architecture efficiency.
"""
    parser = DocumentLayoutParser()
    doc = parser.parse_document_text(doc_text)

    chunker = LayoutAwareChunker(max_tokens_per_chunk=300)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 1
    # Check that the table exists in a chunk with intact markdown formatting
    table_chunk = next(c for c in chunks if "table" in c.element_types)
    assert "| Model Tier | Layers | Heads | Benchmark Score |" in table_chunk.text
    assert "| Tiny | 4 | 4 | 72.4% |" in table_chunk.text
    # Breadcrumbs check
    assert any("Machine Learning Report" in b for b in table_chunk.breadcrumbs)


def test_layout_aware_chunker_splits_oversized_table_with_headers():
    rows_text = "\n".join(
        [f"| Experiment {i} | Value {i} | Result {i * 10} |" for i in range(1, 40)]
    )
    doc_text = f"""# Large Benchmark Table
| Experiment | Value | Result |
|---|---|---|
{rows_text}
"""
    parser = DocumentLayoutParser()
    doc = parser.parse_document_text(doc_text)

    # Set small chunk token ceiling to force table splitting
    chunker = LayoutAwareChunker(max_tokens_per_chunk=80)
    chunks = chunker.chunk_document(doc)

    table_chunks = [c for c in chunks if "table" in c.element_types]
    assert len(table_chunks) > 1
    # Every split table chunk must contain the table header
    for c in table_chunks:
        assert "| Experiment | Value | Result |" in c.text
