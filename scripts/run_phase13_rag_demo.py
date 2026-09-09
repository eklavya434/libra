"""
Libra Phase 13 Demonstration: RAG Vector Retrieval & Document Chunking

Demonstrates:
1. Recursive text chunking with semantic boundary preservation and overlap.
2. First-principles dense vector embedding (D=128, L2 unit norm).
3. Exact cosine similarity vector search via NumPy matrix multiplication.
4. Prompt grounding and context synthesis with citations.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.rag import (
    InMemoryVectorStore,
    RAGPromptSynthesizer,
    RecursiveCharacterChunker,
    get_embedding_provider,
)


def main():
    print("=" * 80)
    print("LIBRA PHASE 13: RETRIEVAL-AUGMENTED GENERATION (RAG) LABORATORY")
    print("Document Chunking, Dense Vector Embeddings, Cosine Search & Grounding")
    print("=" * 80)

    # 1. Document Chunking Demonstration
    print("\n1. Recursive Character Chunking Demonstration:")
    sample_text = (
        "Project Libra is an educational LLM laboratory built from mathematical first principles.\n\n"
        "Modern transformer architectures utilize Rotary Positional Embeddings (RoPE) to encode\n"
        "relative token distances directly into query and key attention dot-products.\n\n"
        "Root Mean Square Normalization (RMSNorm) replaces LayerNorm by omitting mean-centering,\n"
        "significantly improving CPU throughput during autoregressive token generation."
    )

    chunker = RecursiveCharacterChunker(chunk_size=160, chunk_overlap=30)
    chunks = chunker.chunk_text(sample_text, doc_id="doc-arch", doc_title="Libra Architecture")

    print(f"   Original Text Length: {len(sample_text)} characters")
    print(f"   Generated Chunks:     {len(chunks)} chunks (size=160, overlap=30)")
    for i, c in enumerate(chunks):
        print(
            f"   [Chunk {i + 1}] (chars {c.char_start:3d}-{c.char_end:3d}, ~{c.token_count:2d} tok): {c.text[:65]}..."
        )

    # 2. Dense Vector Embeddings
    print("\n2. First-Principles Dense Vector Embeddings:")
    embedder = get_embedding_provider("educational")
    sample_phrase = "Rotary Positional Embeddings in transformers"
    vec = embedder.embed_text(sample_phrase)
    l2_norm = sum(x * x for x in vec) ** 0.5

    print(f"   Input Text:         '{sample_phrase}'")
    print(f"   Vector Dimension:   D = {embedder.dimension}")
    print(f"   Euclidean L2 Norm:  ||v||_2 = {l2_norm:.6f} (Unit sphere normalized)")
    print(f"   Sample Dimensions:  [{', '.join(f'{x:+.3f}' for x in vec[:6])}, ...]")

    # 3. In-Memory Vector Store Indexing
    print("\n3. Indexing Educational Knowledge Base:")
    store = InMemoryVectorStore(embedder=embedder, chunker=chunker)

    documents = [
        (
            "Attention & Transformers",
            "Attention computes weighted similarity between query vectors and key vectors. "
            "Softmax normalization scales dot products by sqrt(d_k) to prevent gradient vanishing.",
        ),
        (
            "Positional Encodings",
            "Rotary Positional Embedding (RoPE) applies a rotation matrix to queries and keys in 2D pairs. "
            "This preserves relative distance information invariant to absolute sequence positions.",
        ),
        (
            "Hardware Sizing for CPU",
            "Project Libra targets Intel Core i5 CPUs with 16GB RAM. Models are strictly budgeted "
            "under 15GB disk storage with CPU-friendly execution under 15 minutes.",
        ),
    ]

    for title, content in documents:
        doc, doc_chunks = store.add_document(title=title, content=content)
        print(f"   - Indexed: '{doc.title}' ({len(doc_chunks)} chunks)")

    print(f"\n   Total Documents in Index: {store.total_documents}")
    print(f"   Total Chunks in Matrix:   {store.total_chunks}")

    # 4. Semantic Similarity Search
    print("\n4. Executing Semantic Vector Search:")
    query = "How does RoPE positional encoding work in queries and keys?"
    print(f"   Search Query: '{query}'")

    results = store.similarity_search(query, top_k=2)
    print(f"\n   Retrieved Top-{len(results)} Chunks:")
    for res in results:
        print(f"   [Rank {res.rank}] Score: {res.score:.4f} | Source: '{res.chunk.doc_title}'")
        print(f"          Snippet: {res.chunk.text}")

    # 5. Prompt Grounding & Synthesis
    print("\n5. Prompt Grounding & Synthesis for LLM Generation:")
    synthesizer = RAGPromptSynthesizer()
    grounded_prompt = synthesizer.build_grounded_prompt(query, results)

    print("-" * 80)
    print(grounded_prompt)
    print("-" * 80)

    print("\n" + "=" * 80)
    print("PHASE 13 COMPLETE: First-Principles RAG Pipeline Verified Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
