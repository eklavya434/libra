"""
Libra Core - First-Principles Document Layout Parser (LibraOCR)
Reference: "LayoutLM: Pre-training of Text and Layout for Document Image Understanding"

Deconstructs visual document layouts into structured semantic AST elements:
Headings, Paragraphs, Tables, LaTeX Equations, and Key-Value pairs with
normalized spatial bounding boxes [x_min, y_min, x_max, y_max] in [0, 1]^4.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ElementType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    EQUATION = "equation"
    KEY_VALUE = "key_value"
    LIST_ITEM = "list_item"


@dataclass
class BoundingBox:
    """Normalized spatial 2D coordinates [x_min, y_min, x_max, y_max] in [0, 1]^4."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        self.x_min = max(0.0, min(1.0, float(self.x_min)))
        self.y_min = max(0.0, min(1.0, float(self.y_min)))
        self.x_max = max(self.x_min, min(1.0, float(self.x_max)))
        self.y_max = max(self.y_min, min(1.0, float(self.y_max)))

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains_point(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def intersection_over_union(self, other: BoundingBox) -> float:
        """Calculates 2D IoU overlap metric between two bounding boxes."""
        inter_xmin = max(self.x_min, other.x_min)
        inter_ymin = max(self.y_min, other.y_min)
        inter_xmax = min(self.x_max, other.x_max)
        inter_ymax = min(self.y_max, other.y_max)

        inter_w = max(0.0, inter_xmax - inter_xmin)
        inter_h = max(0.0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        return (inter_area / union_area) if union_area > 0 else 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "x_min": round(self.x_min, 4),
            "y_min": round(self.y_min, 4),
            "x_max": round(self.x_max, 4),
            "y_max": round(self.y_max, 4),
        }


@dataclass
class TableGrid:
    """Structured matrix representation of tabular data."""

    headers: list[str]
    rows: list[list[str]]

    @property
    def num_rows(self) -> int:
        return len(self.rows)

    @property
    def num_cols(self) -> int:
        return len(self.headers) if self.headers else (len(self.rows[0]) if self.rows else 0)

    def to_markdown(self) -> str:
        if not self.headers and not self.rows:
            return ""
        lines: list[str] = []
        if self.headers:
            lines.append("| " + " | ".join(self.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")
        for row in self.rows:
            # Pad row if fewer cols than headers
            padded_row = row + [""] * (len(self.headers) - len(row))
            lines.append("| " + " | ".join(padded_row[: len(self.headers)]) + " |")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "headers": self.headers,
            "rows": self.rows,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
        }


@dataclass
class DocumentElement:
    """A semantic document layout element with visual bounding coordinates."""

    element_id: str
    type: ElementType
    content: str
    bbox: BoundingBox
    page_number: int
    reading_order: int
    table_data: TableGrid | None = None
    latex_formula: str | None = None
    confidence: float = 0.98
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {
            "element_id": self.element_id,
            "type": self.type.value,
            "content": self.content,
            "bbox": self.bbox.to_dict(),
            "page_number": self.page_number,
            "reading_order": self.reading_order,
            "confidence": round(self.confidence, 3),
            "metadata": self.metadata,
        }
        if self.table_data:
            res["table_data"] = self.table_data.to_dict()
        if self.latex_formula:
            res["latex_formula"] = self.latex_formula
        return res


@dataclass
class DocumentPage:
    """Represents a single page within a parsed visual document."""

    page_number: int
    elements: list[DocumentElement] = field(default_factory=list)
    width: int = 800
    height: int = 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "num_elements": len(self.elements),
            "elements": [e.to_dict() for e in self.elements],
        }


@dataclass
class ParsedDocument:
    """The complete multi-page document layout AST."""

    doc_id: str
    title: str
    pages: list[DocumentPage] = field(default_factory=list)

    @property
    def all_elements(self) -> list[DocumentElement]:
        elems: list[DocumentElement] = []
        for p in self.pages:
            elems.extend(p.elements)
        return elems

    def to_markdown(self) -> str:
        lines: list[str] = [f"# {self.title}\n"]
        for p in self.pages:
            lines.append(f"\n<!-- Page {p.page_number} -->\n")
            for elem in p.elements:
                if elem.type == ElementType.HEADING:
                    lines.append(f"\n{elem.content}\n")
                elif elem.type == ElementType.TABLE and elem.table_data:
                    lines.append(f"\n{elem.table_data.to_markdown()}\n")
                elif elem.type == ElementType.EQUATION and elem.latex_formula:
                    lines.append(f"\n$$\n{elem.latex_formula}\n$$\n")
                elif elem.type == ElementType.KEY_VALUE:
                    lines.append(f"- **{elem.content}**")
                else:
                    lines.append(f"\n{elem.content}\n")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "num_pages": len(self.pages),
            "total_elements": len(self.all_elements),
            "pages": [p.to_dict() for p in self.pages],
        }


class DocumentLayoutParser:
    """
    Parses documents into structured visual layout blocks:
    Headings, Tables, LaTeX formulas, Key-Value pairs, and Paragraphs.
    Assigns realistic normalized 2D bounding boxes for visual verification.
    """

    def parse_document_text(
        self,
        raw_text: str,
        doc_id: str | None = None,
        title: str = "Parsed Document",
    ) -> ParsedDocument:
        """Parses multi-page or single-page raw text into structured DocumentPages."""
        doc_id = doc_id or f"doc_{uuid.uuid4().hex[:6]}"
        raw_pages = raw_text.split("---PAGE---") if "---PAGE---" in raw_text else [raw_text]

        parsed_pages: list[DocumentPage] = []

        for p_idx, page_str in enumerate(raw_pages):
            page_num = p_idx + 1
            elements = self._parse_page_elements(page_str.strip(), page_num)
            parsed_pages.append(
                DocumentPage(
                    page_number=page_num,
                    elements=elements,
                    width=800,
                    height=1000,
                )
            )

        # Detect document title from first heading only if not explicitly provided
        if title in (None, "Parsed Document"):
            first_heading = next(
                (e.content for e in parsed_pages[0].elements if e.type == ElementType.HEADING),
                None,
            )
            if first_heading:
                title = first_heading.lstrip("#").strip()

        return ParsedDocument(
            doc_id=doc_id,
            title=title,
            pages=parsed_pages,
        )

    def _parse_page_elements(self, page_str: str, page_num: int) -> list[DocumentElement]:
        """Scans lines and blocks of a page, categorizing each into ElementType."""
        lines = page_str.split("\n")
        elements: list[DocumentElement] = []
        reading_order = 0

        # State tracking for multi-line elements (tables and equations)
        i = 0
        total_lines = len(lines)
        curr_y = 0.05  # Start with 5% top margin

        while i < total_lines:
            line = lines[i].strip()
            if not line:
                i += 1
                curr_y += 0.015
                continue

            reading_order += 1
            elem_id = f"p{page_num}_e{reading_order}"

            # 1. LaTeX Display Equation ($$...$$)
            if line.startswith("$$"):
                eq_lines: list[str] = []
                if line.endswith("$$") and len(line) > 2:
                    eq_content = line[2:-2].strip()
                    i += 1
                else:
                    i += 1
                    while i < total_lines and not lines[i].strip().endswith("$$"):
                        eq_lines.append(lines[i].strip())
                        i += 1
                    if i < total_lines:
                        # Add any text before trailing $$
                        trailing = lines[i].strip()[:-2].strip()
                        if trailing:
                            eq_lines.append(trailing)
                        i += 1
                    eq_content = "\n".join(eq_lines)

                elem_h = 0.08
                bbox = BoundingBox(
                    x_min=0.15, y_min=curr_y, x_max=0.85, y_max=min(0.95, curr_y + elem_h)
                )
                curr_y += elem_h + 0.02

                elements.append(
                    DocumentElement(
                        element_id=elem_id,
                        type=ElementType.EQUATION,
                        content=f"$${eq_content}$$",
                        bbox=bbox,
                        page_number=page_num,
                        reading_order=reading_order,
                        latex_formula=eq_content,
                    )
                )
                continue

            # 2. Markdown Table Detection (| col1 | col2 |)
            if line.startswith("|") and line.endswith("|"):
                table_lines: list[str] = []
                while (
                    i < total_lines
                    and lines[i].strip().startswith("|")
                    and lines[i].strip().endswith("|")
                ):
                    table_lines.append(lines[i].strip())
                    i += 1

                grid = self._parse_markdown_table(table_lines)
                elem_h = min(0.35, 0.04 * (len(table_lines) + 1))
                bbox = BoundingBox(
                    x_min=0.08, y_min=curr_y, x_max=0.92, y_max=min(0.95, curr_y + elem_h)
                )
                curr_y += elem_h + 0.02

                elements.append(
                    DocumentElement(
                        element_id=elem_id,
                        type=ElementType.TABLE,
                        content=grid.to_markdown(),
                        bbox=bbox,
                        page_number=page_num,
                        reading_order=reading_order,
                        table_data=grid,
                        metadata={"num_rows": grid.num_rows, "num_cols": grid.num_cols},
                    )
                )
                continue

            # 3. Headings (# Title, ## Section)
            if line.startswith("#"):
                elem_h = 0.05 if line.startswith("##") else 0.07
                bbox = BoundingBox(
                    x_min=0.08, y_min=curr_y, x_max=0.92, y_max=min(0.95, curr_y + elem_h)
                )
                curr_y += elem_h + 0.015

                elements.append(
                    DocumentElement(
                        element_id=elem_id,
                        type=ElementType.HEADING,
                        content=line,
                        bbox=bbox,
                        page_number=page_num,
                        reading_order=reading_order,
                        metadata={"level": line.count("#", 0, line.find(" "))},
                    )
                )
                i += 1
                continue

            # 4. Key-Value Pairs (e.g. "Invoice Date: 2026-09-09", "Total Amount: $1,250.00")
            kv_match = re.match(r"^([A-Za-z0-9 _\-\(\)]+)\s*:\s*(.+)$", line)
            if kv_match and len(line) < 100 and not line.startswith("http"):
                key, val = kv_match.group(1).strip(), kv_match.group(2).strip()
                elem_h = 0.035
                bbox = BoundingBox(
                    x_min=0.08, y_min=curr_y, x_max=0.75, y_max=min(0.95, curr_y + elem_h)
                )
                curr_y += elem_h + 0.01

                elements.append(
                    DocumentElement(
                        element_id=elem_id,
                        type=ElementType.KEY_VALUE,
                        content=line,
                        bbox=bbox,
                        page_number=page_num,
                        reading_order=reading_order,
                        metadata={"key": key, "value": val},
                    )
                )
                i += 1
                continue

            # 5. Standard Paragraph / Text
            para_lines = [line]
            i += 1
            while (
                i < total_lines
                and lines[i].strip()
                and not lines[i].strip().startswith(("#", "|", "$$"))
                and not re.match(r"^[A-Za-z0-9 _\-\(\)]+\s*:\s*.+$", lines[i].strip())
            ):
                para_lines.append(lines[i].strip())
                i += 1

            para_text = " ".join(para_lines)
            elem_h = min(0.20, 0.03 * max(1, len(para_lines)))
            bbox = BoundingBox(
                x_min=0.08, y_min=curr_y, x_max=0.92, y_max=min(0.95, curr_y + elem_h)
            )
            curr_y += elem_h + 0.015

            elements.append(
                DocumentElement(
                    element_id=elem_id,
                    type=ElementType.PARAGRAPH,
                    content=para_text,
                    bbox=bbox,
                    page_number=page_num,
                    reading_order=reading_order,
                )
            )

        return elements

    def _parse_markdown_table(self, table_lines: list[str]) -> TableGrid:
        """Parses rows of a pipe table into TableGrid."""
        parsed_rows: list[list[str]] = []
        for line in table_lines:
            cells = [c.strip() for c in line.strip("|").split("|")]
            # Filter out divider rows like |---|---|
            if cells and all(re.match(r"^:?-+:?$", c) for c in cells):
                continue
            parsed_rows.append(cells)

        if not parsed_rows:
            return TableGrid(headers=[], rows=[])

        headers = parsed_rows[0]
        rows = parsed_rows[1:] if len(parsed_rows) > 1 else []
        return TableGrid(headers=headers, rows=rows)
