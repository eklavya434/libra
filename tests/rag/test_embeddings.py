"""
Tests for Dense Embedding Providers (packages/rag/embeddings.py)
"""

import math

import numpy as np

from packages.rag.embeddings import EducationalDenseEmbedder, OllamaEmbeddingProvider


def test_embedder_dimension_and_norm():
    embedder = EducationalDenseEmbedder(dimension=128)
    vec = embedder.embed_text("Self-attention enables contextual word representations.")

    assert len(vec) == 128
    # Verify vector has unit Euclidean norm
    norm = math.sqrt(sum(x * x for x in vec))
    assert math.isclose(norm, 1.0, rel_tol=1e-5)


def test_embedder_determinism():
    embedder = EducationalDenseEmbedder(dimension=64)
    text = "Machine learning models minimize empirical risk."

    v1 = embedder.embed_text(text)
    v2 = embedder.embed_text(text)
    assert v1 == v2


def test_semantic_similarity_behavior():
    embedder = EducationalDenseEmbedder(dimension=128)
    v_base = np.array(embedder.embed_text("Transformer neural network architecture"))
    v_sim = np.array(embedder.embed_text("Transformer neural networks and deep architecture"))
    v_diff = np.array(embedder.embed_text("Making homemade pasta with flour and eggs"))

    score_sim = float(np.dot(v_base, v_sim))
    score_diff = float(np.dot(v_base, v_diff))

    assert score_sim > score_diff


def test_empty_string_embedding():
    embedder = EducationalDenseEmbedder(dimension=64)
    v = embedder.embed_text("")
    assert len(v) == 64
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-5)


def test_ollama_fallback():
    # Target nonexistent port to trigger fallback
    provider = OllamaEmbeddingProvider(base_url="http://127.0.0.1:99999", fallback_dim=64)
    vec = provider.embed_text("Fallback embedding test")
    assert len(vec) == 64
