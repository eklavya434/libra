"""
Tests for Chunk Deduplication and MMR Diversity Filtering
"""

from packages.rag.deduplication import ChunkDeduplicator, jaccard_similarity
from packages.rag.models import DocumentChunk, SearchResult


def test_jaccard_similarity():
    text1 = "Transformers use multi-head attention and rotary position embeddings."
    text2 = "Transformers use multi-head attention and rotary position embeddings!"
    # Identical words
    assert jaccard_similarity(text1, text2) == 1.0

    text3 = "A completely different sentence about cooking pasta."
    assert jaccard_similarity(text1, text3) < 0.1


def test_chunk_deduplication():
    dedup = ChunkDeduplicator(max_jaccard_threshold=0.60)
    c1 = DocumentChunk(
        id="c1",
        doc_id="d1",
        doc_title="Doc",
        text="The quick brown fox jumps over the lazy dog in the green meadow.",
    )
    # c2 is an overlapping chunk with 80% overlap
    c2 = DocumentChunk(
        id="c2",
        doc_id="d1",
        doc_title="Doc",
        text="quick brown fox jumps over the lazy dog in the green meadow today.",
    )
    # c3 is completely distinct
    c3 = DocumentChunk(
        id="c3",
        doc_id="d2",
        doc_title="Doc 2",
        text="Quantum computers use qubits to perform calculations in superposition.",
    )

    results = [
        SearchResult(chunk=c1, score=0.95, rank=1),
        SearchResult(chunk=c2, score=0.92, rank=2),
        SearchResult(chunk=c3, score=0.75, rank=3),
    ]

    deduped = dedup.deduplicate(results, top_k=3)
    # c2 should be eliminated because it's too similar to c1
    assert len(deduped) == 2
    ids = [r.chunk.id for r in deduped]
    assert "c1" in ids
    assert "c3" in ids
    assert "c2" not in ids
    assert deduped[0].rank == 1
    assert deduped[1].rank == 2


def test_mmr_selection():
    dedup = ChunkDeduplicator(mmr_lambda=0.5)
    c1 = DocumentChunk(
        id="c1",
        doc_id="d1",
        doc_title="D1",
        text="Machine learning algorithms and deep neural nets.",
    )
    c2 = DocumentChunk(
        id="c2",
        doc_id="d1",
        doc_title="D1",
        text="Deep neural nets and machine learning algorithms.",
    )
    c3 = DocumentChunk(
        id="c3", doc_id="d2", doc_title="D2", text="Database transactions use ACID properties."
    )

    results = [
        SearchResult(chunk=c1, score=0.95, rank=1),
        SearchResult(chunk=c2, score=0.90, rank=2),
        SearchResult(chunk=c3, score=0.65, rank=3),
    ]

    mmr_res = dedup.mmr(results, top_k=2)
    assert len(mmr_res) == 2
    # c1 is chosen first, then c3 should be preferred over redundant c2 due to diversity penalty
    assert mmr_res[0].chunk.id == "c1"
    assert mmr_res[1].chunk.id == "c3"
