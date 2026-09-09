"""
Tests for DocumentLayoutParser & BoundingBox (packages/core/document/layout_parser.py)
"""

from packages.core.document.layout_parser import (
    BoundingBox,
    DocumentLayoutParser,
    ElementType,
    TableGrid,
)


def test_bounding_box_geometry():
    b1 = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
    assert b1.width == 0.4
    assert b1.height == 0.4
    assert round(b1.area, 4) == 0.16
    assert b1.contains_point(0.2, 0.3)
    assert not b1.contains_point(0.6, 0.6)

    # Intersection over union (IoU) with identical box
    assert b1.intersection_over_union(b1) == 1.0

    # IoU with disjoint box
    b2 = BoundingBox(x_min=0.6, y_min=0.6, x_max=0.9, y_max=0.9)
    assert b1.intersection_over_union(b2) == 0.0


def test_table_grid_markdown_roundtrip():
    headers = ["Metric", "2025", "2026"]
    rows = [
        ["Revenue", "$10M", "$15M"],
        ["Net Income", "$2M", "$3.5M"],
    ]
    grid = TableGrid(headers=headers, rows=rows)
    assert grid.num_rows == 2
    assert grid.num_cols == 3
    md = grid.to_markdown()
    assert "| Revenue | $10M | $15M |" in md
    assert "| Net Income | $2M | $3.5M |" in md


def test_document_layout_parser_extracts_all_element_types():
    doc_text = """# Libra Financial Audit 2026
Document ID: DOC-994
Reporting Currency: USD

This is an introductory paragraph describing the operational review.

| Service | Q1 | Q2 |
| Engine | $100 | $150 |
| Cache | $80 | $90 |

The loss function used for alignment is defined as:
$$
L_{DPO} = -\\log \\sigma(r_w - r_l)
$$
"""
    parser = DocumentLayoutParser()
    doc = parser.parse_document_text(doc_text)

    assert doc.title == "Libra Financial Audit 2026"
    assert len(doc.pages) == 1

    elems = doc.all_elements
    types = [e.type for e in elems]

    assert ElementType.HEADING in types
    assert ElementType.KEY_VALUE in types
    assert ElementType.PARAGRAPH in types
    assert ElementType.TABLE in types
    assert ElementType.EQUATION in types

    # Verify table data
    table_elem = next(e for e in elems if e.type == ElementType.TABLE)
    assert table_elem.table_data is not None
    assert table_elem.table_data.headers == ["Service", "Q1", "Q2"]
    assert len(table_elem.table_data.rows) == 2

    # Verify equation data
    eq_elem = next(e for e in elems if e.type == ElementType.EQUATION)
    assert eq_elem.latex_formula is not None
    assert "L_{DPO}" in eq_elem.latex_formula

    # Verify bounding box validity
    for e in elems:
        assert 0.0 <= e.bbox.x_min <= e.bbox.x_max <= 1.0
        assert 0.0 <= e.bbox.y_min <= e.bbox.y_max <= 1.0
