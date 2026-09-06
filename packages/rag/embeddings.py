"""
Libra RAG Package - Dense Vector Embeddings

Provides embedding providers for semantic vector representation:
1. EducationalDenseEmbedder: First-principles deterministic n-gram hash projection (D=128, L2-normalized).
2. OllamaEmbeddingProvider: Adapter for local Ollama embedding models with graceful fallback.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Optional
import urllib.request
import json


class BaseEmbeddingProvider(ABC):
    """Abstract interface for text embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate a dense embedding vector for a single text."""
        pass

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        return [self.embed_text(t) for t in texts]


class EducationalDenseEmbedder(BaseEmbeddingProvider):
    """
    First-principles dense embedder.

    Computes semantic vector representations using tokenization, sliding character
    n-grams, deterministic feature hashing, and Euclidean L2-normalization.
    """

    def __init__(self, dimension: int = 128) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _tokenize_ngrams(self, text: str) -> list[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        words = cleaned.split()
        features = list(words)

        # Generate character n-grams (3-grams and 4-grams) to capture subword morphology
        for word in words:
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    features.append(word[i : i + 3])
            if len(word) >= 4:
                for i in range(len(word) - 3):
                    features.append(word[i : i + 4])

        return features

    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            # Return zero vector with unit norm fallback
            v = [0.0] * self._dim
            v[0] = 1.0
            return v

        vec = [0.0] * self._dim
        features = self._tokenize_ngrams(text)

        for feat in features:
            # Deterministic hash to dimension index
            h_bytes = hashlib.md5(feat.encode("utf-8")).digest()
            idx = int.from_bytes(h_bytes[:4], byteorder="little") % self._dim
            # Sign hash (-1 or +1)
            sign = 1.0 if (h_bytes[4] % 2 == 0) else -1.0
            vec[idx] += sign

        # Compute Euclidean L2 norm
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            vec = [x / norm for x in vec]
        else:
            vec[0] = 1.0

        return vec


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Connects to local Ollama daemon for dense embeddings with fallback."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://127.0.0.1:11434",
        fallback_dim: int = 128,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.fallback = EducationalDenseEmbedder(dimension=fallback_dim)
        self._dim = fallback_dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> list[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    emb = data.get("embedding")
                    if emb and isinstance(emb, list):
                        self._dim = len(emb)
                        return emb
        except Exception:
            pass

        # Graceful fallback to zero-cost offline embedder
        return self.fallback.embed_text(text)


def get_embedding_provider(name: str = "educational") -> BaseEmbeddingProvider:
    """Factory function for embedding providers."""
    if name == "ollama":
        return OllamaEmbeddingProvider()
    return EducationalDenseEmbedder(dimension=128)
