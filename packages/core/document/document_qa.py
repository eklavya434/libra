"""
Libra Core - Grounded Document QA Engine
Performs grounded answering with spatial bounding box and tabular cell citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from packages.core.document.layout_parser import (
    DocumentElement,
    ElementType,
    ParsedDocument,
)


@dataclass
class DocumentCitation:
    """Exact spatial and tabular reference backing an extracted answer."""

    page_number: int
    element_id: str
    element_type: str
    snippet: str
    bbox: dict[str, float]
    cell_reference: str | None = None
    confidence: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "element_id": self.element_id,
            "element_type": self.element_type,
            "snippet": self.snippet,
            "bbox": self.bbox,
            "cell_reference": self.cell_reference,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class GroundedQAResult:
    """The synthesized answer accompanied by visual and tabular citations."""

    query: str
    answer: str
    citations: list[DocumentCitation] = field(default_factory=list)
    relevant_element_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "relevant_element_ids": self.relevant_element_ids,
        }


class DocumentQAEngine:
    """
    Grounded visual question-answering engine for structured documents.
    Maps user inquiries to tables, key-values, equations, and paragraphs,
    extracting precise cell and bounding box citations.
    """

    def answer_query(self, document: ParsedDocument, query: str) -> GroundedQAResult:
        """Analyzes document AST to answer the question with visual citations."""
        q_lower = query.lower()
        q_tokens = set(re.findall(r"\w+", q_lower))

        scored_elements: list[tuple[float, DocumentElement, str | None]] = []

        for elem in document.all_elements:
            score = 0.0
            cell_ref = None

            # 1. Key-Value Match
            if elem.type == ElementType.KEY_VALUE and elem.metadata:
                key = elem.metadata.get("key", "").lower()
                val = elem.metadata.get("value", "")
                k_tokens = set(re.findall(r"\w+", key))
                overlap = len(q_tokens.intersection(k_tokens))
                if overlap > 0:
                    score = 2.0 * overlap + 1.0
                    cell_ref = f"Key '{elem.metadata.get('key')}': {val}"

            # 2. Table Match
            elif elem.type == ElementType.TABLE and elem.table_data:
                grid = elem.table_data
                # Check header overlaps
                h_overlap = sum(1 for h in grid.headers if any(t in h.lower() for t in q_tokens))
                if h_overlap > 0:
                    score += 1.5 * h_overlap

                # Search rows for query token matches
                for r_i, row in enumerate(grid.rows):
                    row_text = " ".join(row).lower()
                    row_matches = sum(1 for t in q_tokens if t in row_text)
                    if row_matches > 0:
                        score += 2.0 * row_matches
                        # Identify specific matching column
                        col_names = [
                            grid.headers[c_i]
                            for c_i, c in enumerate(row)
                            if any(t in c.lower() for t in q_tokens) and c_i < len(grid.headers)
                        ]
                        col_label = f" (columns: {', '.join(col_names)})" if col_names else ""
                        cell_ref = f"Table Row {r_i + 1}{col_label}: {' | '.join(row)}"
                        break

            # 3. LaTeX Equation Match
            elif elem.type == ElementType.EQUATION:
                if any(
                    w in q_lower
                    for w in [
                        "formula",
                        "equation",
                        "math",
                        "calculate",
                        "derivative",
                        "attention",
                        "loss",
                        "entropy",
                    ]
                ):
                    score += 2.5
                    cell_ref = "Formula block"

            # 4. Heading or Paragraph Match
            else:
                elem_tokens = set(re.findall(r"\w+", elem.content.lower()))
                overlap = len(q_tokens.intersection(elem_tokens))
                score += overlap * 0.8

            if score > 0:
                scored_elements.append((score, elem, cell_ref))

        # Sort elements by relevance score
        scored_elements.sort(key=lambda x: x[0], reverse=True)

        if not scored_elements:
            return GroundedQAResult(
                query=query,
                answer="The document does not appear to contain relevant information answering this query.",
                citations=[],
                relevant_element_ids=[],
            )

        top_elements = scored_elements[:3]
        citations: list[DocumentCitation] = []
        relevant_ids: list[str] = []

        answer_points: list[str] = []

        for score, elem, cell_ref in top_elements:
            relevant_ids.append(elem.element_id)
            snippet = elem.content[:200]

            citations.append(
                DocumentCitation(
                    page_number=elem.page_number,
                    element_id=elem.element_id,
                    element_type=elem.type.value,
                    snippet=snippet,
                    bbox=elem.bbox.to_dict(),
                    cell_reference=cell_ref,
                    confidence=min(0.99, 0.70 + score * 0.05),
                )
            )

            if elem.type == ElementType.KEY_VALUE:
                answer_points.append(f"Based on **{elem.content}** (Page {elem.page_number}).")
            elif elem.type == ElementType.TABLE and cell_ref:
                answer_points.append(
                    f"According to the table on Page {elem.page_number} ({cell_ref})."
                )
            elif elem.type == ElementType.EQUATION and elem.latex_formula:
                answer_points.append(
                    f"Referencing equation on Page {elem.page_number}: $${elem.latex_formula}$$."
                )
            else:
                answer_points.append(
                    f'From Page {elem.page_number}: "{elem.content.strip()[:180]}..."'
                )

        answer = " ".join(answer_points)

        return GroundedQAResult(
            query=query,
            answer=answer,
            citations=citations,
            relevant_element_ids=relevant_ids,
        )
