"""
Libra Core - Layout-Aware Semantic Document Chunker
Preserves tabular matrices, LaTeX formulas, and heading breadcrumbs during RAG chunking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from packages.core.document.layout_parser import (
    DocumentElement,
    ElementType,
    ParsedDocument,
)


@dataclass
class DocumentChunk:
    """A semantic chunk preserving layout hierarchy, metadata, and visual coordinates."""

    chunk_id: str
    text: str
    page_number: int
    breadcrumbs: list[str]
    element_types: list[str]
    element_ids: list[str]
    bounding_boxes: list[dict[str, float]]
    token_estimate: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "page_number": self.page_number,
            "breadcrumbs": self.breadcrumbs,
            "element_types": self.element_types,
            "element_ids": self.element_ids,
            "bounding_boxes": self.bounding_boxes,
            "token_estimate": self.token_estimate,
            "metadata": self.metadata,
        }


class LayoutAwareChunker:
    """
    Chunks visual document layout ASTs into retrieval-optimized segments:
    1. Preserves table rows without splitting across arbitrary character limits.
    2. Injects table headers onto overflow split tables.
    3. Retains heading breadcrumbs for hierarchical RAG retrieval.
    4. Attaches page numbers and spatial bounding boxes for visual citation.
    """

    def __init__(self, max_tokens_per_chunk: int = 400, overlap_tokens: int = 50) -> None:
        self.max_tokens_per_chunk = max(100, max_tokens_per_chunk)
        self.overlap_tokens = max(0, overlap_tokens)

    def _estimate_tokens(self, text: str) -> int:
        """Heuristic token estimator (average 1 token ≈ 4 characters or word count * 1.3)."""
        words = len(text.split())
        return max(1, int(words * 1.3))

    def chunk_document(self, doc: ParsedDocument) -> list[DocumentChunk]:
        """Processes all pages and elements into structured chunks."""
        chunks: list[DocumentChunk] = []
        active_breadcrumbs: list[str] = [doc.title] if doc.title else []
        chunk_idx = 0

        for page in doc.pages:
            page_num = page.page_number
            curr_elements: list[DocumentElement] = []
            curr_text_parts: list[str] = []
            curr_token_count = 0

            for elem in page.elements:
                # Update heading breadcrumbs
                if elem.type == ElementType.HEADING:
                    # Clean heading text
                    clean_h = elem.content.lstrip("#").strip()
                    level = elem.metadata.get("level", 1)
                    if level <= len(active_breadcrumbs):
                        active_breadcrumbs = active_breadcrumbs[:level]
                        active_breadcrumbs.append(clean_h)
                    else:
                        active_breadcrumbs.append(clean_h)

                elem_text = elem.content
                elem_tokens = self._estimate_tokens(elem_text)

                # Special Handling for Large Tables
                if elem.type == ElementType.TABLE and elem.table_data:
                    # If current buffer has text, flush it first
                    if curr_elements:
                        chunk_idx += 1
                        chunks.append(
                            self._create_chunk(
                                chunk_idx=chunk_idx,
                                page_num=page_num,
                                breadcrumbs=list(active_breadcrumbs),
                                elements=curr_elements,
                                text_parts=curr_text_parts,
                            )
                        )
                        curr_elements = []
                        curr_text_parts = []
                        curr_token_count = 0

                    # Check if table fits in a single chunk or needs row-level splitting
                    if elem_tokens <= self.max_tokens_per_chunk:
                        chunk_idx += 1
                        chunks.append(
                            self._create_chunk(
                                chunk_idx=chunk_idx,
                                page_num=page_num,
                                breadcrumbs=list(active_breadcrumbs),
                                elements=[elem],
                                text_parts=[elem_text],
                            )
                        )
                    else:
                        # Split table rows while re-injecting headers
                        table_chunks = self._split_large_table(elem, active_breadcrumbs, chunk_idx)
                        chunks.extend(table_chunks)
                        chunk_idx += len(table_chunks)
                    continue

                # Check if adding this element exceeds chunk token ceiling
                if curr_token_count + elem_tokens > self.max_tokens_per_chunk and curr_elements:
                    chunk_idx += 1
                    chunks.append(
                        self._create_chunk(
                            chunk_idx=chunk_idx,
                            page_num=page_num,
                            breadcrumbs=list(active_breadcrumbs),
                            elements=curr_elements,
                            text_parts=curr_text_parts,
                        )
                    )
                    curr_elements = []
                    curr_text_parts = []
                    curr_token_count = 0

                curr_elements.append(elem)
                curr_text_parts.append(elem_text)
                curr_token_count += elem_tokens

            # Flush remaining elements on the page
            if curr_elements:
                chunk_idx += 1
                chunks.append(
                    self._create_chunk(
                        chunk_idx=chunk_idx,
                        page_num=page_num,
                        breadcrumbs=list(active_breadcrumbs),
                        elements=curr_elements,
                        text_parts=curr_text_parts,
                    )
                )

        return chunks

    def _create_chunk(
        self,
        chunk_idx: int,
        page_num: int,
        breadcrumbs: list[str],
        elements: list[DocumentElement],
        text_parts: list[str],
    ) -> DocumentChunk:
        """Constructs a DocumentChunk with full spatial metadata."""
        # Prepend breadcrumbs context header if available
        breadcrumb_header = f"[{' > '.join(breadcrumbs)}]\n\n" if breadcrumbs else ""
        full_text = breadcrumb_header + "\n\n".join(text_parts)

        return DocumentChunk(
            chunk_id=f"chunk_{chunk_idx:03d}",
            text=full_text,
            page_number=page_num,
            breadcrumbs=breadcrumbs,
            element_types=[e.type.value for e in elements],
            element_ids=[e.element_id for e in elements],
            bounding_boxes=[e.bbox.to_dict() for e in elements],
            token_estimate=self._estimate_tokens(full_text),
            metadata={"num_elements": len(elements)},
        )

    def _split_large_table(
        self,
        elem: DocumentElement,
        breadcrumbs: list[str],
        start_idx: int,
    ) -> list[DocumentChunk]:
        """Splits an oversized table row-by-row, ensuring headers are repeated on every chunk."""
        grid = elem.table_data
        if not grid or not grid.rows:
            return []

        chunks: list[DocumentChunk] = []
        headers_md = (
            "| "
            + " | ".join(grid.headers)
            + " |\n| "
            + " | ".join(["---"] * len(grid.headers))
            + " |"
        )
        header_tokens = self._estimate_tokens(headers_md)

        curr_rows: list[list[str]] = []
        curr_tokens = header_tokens
        split_idx = start_idx

        for row in grid.rows:
            row_md = "| " + " | ".join(row) + " |"
            row_tokens = self._estimate_tokens(row_md)

            if curr_tokens + row_tokens > self.max_tokens_per_chunk and curr_rows:
                split_idx += 1
                table_text = (
                    headers_md + "\n" + "\n".join(["| " + " | ".join(r) + " |" for r in curr_rows])
                )
                breadcrumb_prefix = f"[{' > '.join(breadcrumbs)} (Table Part)]\n\n"
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"chunk_{split_idx:03d}",
                        text=breadcrumb_prefix + table_text,
                        page_number=elem.page_number,
                        breadcrumbs=breadcrumbs,
                        element_types=[ElementType.TABLE.value],
                        element_ids=[elem.element_id],
                        bounding_boxes=[elem.bbox.to_dict()],
                        token_estimate=self._estimate_tokens(table_text),
                        metadata={"is_split_table": True, "num_rows": len(curr_rows)},
                    )
                )
                curr_rows = []
                curr_tokens = header_tokens

            curr_rows.append(row)
            curr_tokens += row_tokens

        if curr_rows:
            split_idx += 1
            table_text = (
                headers_md + "\n" + "\n".join(["| " + " | ".join(r) + " |" for r in curr_rows])
            )
            breadcrumb_prefix = f"[{' > '.join(breadcrumbs)} (Table Part)]\n\n"
            chunks.append(
                DocumentChunk(
                    chunk_id=f"chunk_{split_idx:03d}",
                    text=breadcrumb_prefix + table_text,
                    page_number=elem.page_number,
                    breadcrumbs=breadcrumbs,
                    element_types=[ElementType.TABLE.value],
                    element_ids=[elem.element_id],
                    bounding_boxes=[elem.bbox.to_dict()],
                    token_estimate=self._estimate_tokens(table_text),
                    metadata={"is_split_table": True, "num_rows": len(curr_rows)},
                )
            )

        return chunks
