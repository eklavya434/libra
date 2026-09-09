"""
Libra RAG Package - Document Chunking

Recursive hierarchical chunking strategies to split documents along natural
semantic boundaries (paragraphs, sentences, words) while respecting size and overlap constraints.
"""

from __future__ import annotations

import math
import uuid
from typing import Optional

from packages.rag.models import Document, DocumentChunk


class RecursiveCharacterChunker:
    """Splits text recursively using a hierarchy of separators."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
        separators: Optional[list[str]] = None,
        chars_per_token: float = 3.8,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.chars_per_token = chars_per_token

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / self.chars_per_token))

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text by the first matching separator."""
        final_chunks: list[str] = []

        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Find first matching separator
        chosen_sep = ""
        for sep in separators:
            if sep == "":
                chosen_sep = ""
                break
            if sep in text:
                chosen_sep = sep
                break

        if chosen_sep:
            splits = text.split(chosen_sep)
        else:
            # Character-level fallback split
            step = max(1, self.chunk_size - self.chunk_overlap)
            return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]

        # Merge splits up to chunk_size
        current_chunk: list[str] = []
        current_len = 0
        remaining_separators = (
            separators[separators.index(chosen_sep) + 1 :] if chosen_sep in separators else []
        )

        for piece in splits:
            piece_len = len(piece) + (len(chosen_sep) if current_chunk else 0)
            if current_len + piece_len <= self.chunk_size:
                current_chunk.append(piece)
                current_len += piece_len
            else:
                if current_chunk:
                    merged = chosen_sep.join(current_chunk)
                    if len(merged) > self.chunk_size and remaining_separators:
                        final_chunks.extend(self._split_text(merged, remaining_separators))
                    elif merged.strip():
                        final_chunks.append(merged)
                current_chunk = [piece]
                current_len = len(piece)

        if current_chunk:
            merged = chosen_sep.join(current_chunk)
            if len(merged) > self.chunk_size and remaining_separators:
                final_chunks.extend(self._split_text(merged, remaining_separators))
            elif merged.strip():
                final_chunks.append(merged)

        return final_chunks

    def chunk_text(
        self,
        text: str,
        doc_id: str = "doc-0",
        doc_title: str = "Document",
    ) -> list[DocumentChunk]:
        """Splits raw text into DocumentChunk instances with character offsets."""
        raw_splits = self._split_text(text, self.separators)
        if not raw_splits:
            return []

        chunks: list[DocumentChunk] = []
        char_search_idx = 0

        for idx, split_str in enumerate(raw_splits):
            clean_text = split_str.strip()
            if not clean_text:
                continue

            # Locate starting index in parent text
            pos = text.find(clean_text, char_search_idx)
            if pos == -1:
                pos = char_search_idx
            char_end = pos + len(clean_text)
            char_search_idx = max(0, char_end - self.chunk_overlap)

            chunk_id = f"chk-{uuid.uuid4().hex[:10]}"
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    text=clean_text,
                    char_start=pos,
                    char_end=char_end,
                    token_count=self._estimate_tokens(clean_text),
                    metadata={"chunk_index": idx, "chunk_count_total": len(raw_splits)},
                )
            )

        return chunks

    def chunk_document(self, doc: Document) -> list[DocumentChunk]:
        """Convenience helper to chunk a Document model."""
        return self.chunk_text(text=doc.content, doc_id=doc.id, doc_title=doc.title)
