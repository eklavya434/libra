"""
Project Libra - Phase 14 Demo: Advanced Hybrid Search & Re-Ranking
Demonstrates:
  1. Okapi BM25 Sparse Search vs Dense Vector Search
  2. Reciprocal Rank Fusion (RRF)
  3. Multi-Factor Re-Ranking (Phrase match, Keyword coverage, Span proximity)
  4. Chunk Deduplication (Jaccard similarity threshold)
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from packages.rag.hybrid import HybridRetriever
from packages.rag.synthesizer import RAGPromptSynthesizer
from packages.rag.vector_store import InMemoryVectorStore


def main() -> None:
    print("=" * 75)
    print("Project Libra - Phase 14: Advanced Hybrid RAG & Re-Ranking Laboratory")
    print("=" * 75)

    # 1. Initialize Hybrid Retriever with in-memory stores
    store = InMemoryVectorStore()
    retriever = HybridRetriever(vector_store=store)

    # 2. Ingest technical reference documents
    print("\n[Step 1] Ingesting Technical Knowledge Base...")
    docs = [
        (
            "SQLite WAL Memory Architecture",
            "SQLite operates using Write-Ahead Logging (WAL) and B-Tree indexing. "
            "WAL mode improves concurrency by allowing readers to read while writers write, "
            "preventing reader-writer lock contention in local SQLite databases.",
        ),
        (
            "Transformer Attention Mechanism",
            "In modern transformers, Scaled Dot-Product Attention calculates token relationships: "
            "Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V. "
            "Multi-Head Attention projects representations into distinct sub-spaces.",
        ),
        (
            "Rotary Positional Embeddings (RoPE)",
            "Rotary Positional Embeddings (RoPE) encode relative token distances by rotating "
            "query and key vector pairs in 2D coordinate planes using complex rotation matrices. "
            "RoPE exhibits strong length extrapolation properties.",
        ),
        (
            "CPU Hardware Optimization",
            "The Intel Core i5-12450H CPU features 8 cores and 12 threads with AVX2 SIMD vector extensions. "
            "Vectorized NumPy matrix multiplications execute sub-millisecond dot products without GPU.",
        ),
    ]

    for title, content in docs:
        doc, chunks = retriever.add_document(title=title, content=content)
        print(f"  + Indexed '{title}' -> {len(chunks)} chunk(s)")

    print(
        f"\nTotal Documents: {retriever.total_documents} | Total Chunks: {retriever.total_chunks}"
    )

    # 3. Query 1: Exact Technical Keyword Search (BM25 Strength)
    q1 = "AVX2 SIMD i5-12450H"
    print("\n" + "-" * 75)
    print(f"[Query 1: Exact Technical Acronyms] '{q1}'")
    bm25_res = retriever.search(q1, mode="bm25", top_k=2)
    dense_res = retriever.search(q1, mode="dense", top_k=2)
    hybrid_res = retriever.search(q1, mode="hybrid", top_k=2)

    print(
        "  BM25 Top Match   :",
        bm25_res[0].chunk.doc_title,
        f"(BM25 Score: {bm25_res[0].score:.3f})",
    )
    print(
        "  Dense Top Match  :",
        dense_res[0].chunk.doc_title,
        f"(Cosine Score: {dense_res[0].score:.3f})",
    )
    print(
        "  Hybrid Top Match :",
        hybrid_res[0].chunk.doc_title,
        f"(RRF Score: {hybrid_res[0].score:.4f})",
    )

    # 4. Query 2: Conceptual Semantic Search (Dense Strength)
    q2 = "rotating coordinate matrices to measure token distance"
    print("\n" + "-" * 75)
    print(f"[Query 2: Conceptual / Paraphrased Query] '{q2}'")
    bm25_res2 = retriever.search(q2, mode="bm25", top_k=2)
    dense_res2 = retriever.search(q2, mode="dense", top_k=2)
    hybrid_res2 = retriever.search(q2, mode="hybrid", top_k=2)

    bm25_title = bm25_res2[0].chunk.doc_title if bm25_res2 else "No exact lexical match"
    print("  BM25 Top Match   :", bm25_title)
    print(
        "  Dense Top Match  :",
        dense_res2[0].chunk.doc_title,
        f"(Cosine Score: {dense_res2[0].score:.3f})",
    )
    print(
        "  Hybrid Top Match :",
        hybrid_res2[0].chunk.doc_title,
        f"(RRF Score: {hybrid_res2[0].score:.4f})",
    )

    # 5. Query 3: Multi-Factor Re-Ranking & Deduplication Demonstration
    q3 = "Write-Ahead Logging concurrency in SQLite databases"
    print("\n" + "-" * 75)
    print(f"[Query 3: Hybrid + Re-Ranking + Deduplication] '{q3}'")
    results = retriever.search(
        query=q3,
        mode="hybrid",
        use_rrf=True,
        use_reranking=True,
        use_deduplication=True,
        top_k=3,
    )

    print("\nFinal Ranked Search Results:")
    for r in results:
        print(f"  Rank #{r.rank} [{r.chunk.doc_title}] Composite Score: {r.score:.4f}")
        if r.dense_score is not None:
            print(f"    - Dense Cosine Score: {r.dense_score:.3f} (Rank #{r.dense_rank})")
        if r.bm25_score is not None:
            print(f"    - BM25 Sparse Score: {r.bm25_score:.3f} (Rank #{r.bm25_rank})")
        if r.rerank_score is not None:
            print(f"    - Re-ranking Score : {r.rerank_score:.4f}")
        print(f"    - Text: {r.chunk.text[:90]}...")

    # 6. Augmented Grounded Prompt
    synthesizer = RAGPromptSynthesizer()
    prompt = synthesizer.build_grounded_prompt(q3, results)
    print("\n" + "-" * 75)
    print("[Synthesized Grounded Context Prompt Preview]")
    print(prompt[:400] + "...\n[Rest of prompt truncated for brevity]")

    print("=" * 75)
    print("Phase 14 Hybrid RAG & Re-Ranking Demo Completed Successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()
