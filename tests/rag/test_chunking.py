"""
Tests for Recursive Document Chunker (packages/rag/chunking.py)
"""

import pytest

from packages.rag.chunking import RecursiveCharacterChunker


def test_chunk_short_text():
    chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=20)
    text = "Short text under the chunk size threshold."
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(text)
    assert chunks[0].token_count > 0


def test_chunk_multiline_text():
    chunker = RecursiveCharacterChunker(chunk_size=80, chunk_overlap=15)
    text = (
        "Paragraph 1 discusses transformers and self-attention mechanisms in deep learning.\n\n"
        "Paragraph 2 discusses backpropagation and gradient descent optimization algorithms."
    )
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 2
    # Check that chunks contain meaningful content
    assert any("self-attention" in c.text for c in chunks)
    assert any("gradient descent" in c.text for c in chunks)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        RecursiveCharacterChunker(chunk_size=100, chunk_overlap=120)


def test_empty_text_returns_empty():
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
    assert chunker.chunk_text("") == []
    assert chunker.chunk_text("   \n\n  ") == []
