# Phase 40: Multi-Modal Document Understanding & OCR Pipeline (LibraOCR)

## 1. WHAT Was Built
Phase 40 introduces **LibraOCR & Structured Document Understanding**, a visual layout parser, semantic chunker, and grounded question-answering pipeline engineered to process complex visual documents (invoices, financial reports, scientific research papers) on CPU hardware.

Key deliverables:
1. **First-Principles Document Layout Parser (`DocumentLayoutParser` / `LibraOCR`)**:
   - Parses document pages into structured semantic AST elements: `HEADING`, `PARAGRAPH`, `TABLE`, `EQUATION`, and `KEY_VALUE`.
   - Associates every element with normalized 2D spatial bounding boxes:
     $$\text{BoundingBox} = [x_{\min}, y_{\min}, x_{\max}, y_{\max}] \in [0, 1]^4$$
   - Calculates 2D spatial metrics including bounding box Area and Intersection-over-Union (IoU).
   - Reconstructs tables into structured `TableGrid` matrices with row/column headers and Markdown serialization.
   - Extracts mathematical expressions into LaTeX formulas.
2. **Layout-Aware Semantic Chunker (`LayoutAwareChunker`)**:
   - Chunks documents by semantic visual elements rather than arbitrary character offsets.
   - Preserves table rows without splitting them across boundaries; re-injects table headers when an oversized table spans multiple chunks.
   - Propagates hierarchical heading breadcrumbs (e.g. `[Document Title > Section 2: Financial Operations]`) to anchor chunks for RAG retrieval.
3. **Grounded Document QA Engine (`DocumentQAEngine`)**:
   - Maps user questions to document elements with semantic relevance scoring.
   - Returns answers accompanied by verified **Visual & Tabular Citations** (page number, element ID, table row/column cell reference, and normalized bounding box coordinates).
4. **FastAPI Endpoints (`apps/backend/api/v1/endpoints/document_ocr.py`)**:
   - `GET /api/v1/document/presets`: Preloaded benchmark documents (Financial Statement, Transformer Paper, Commercial Invoice).
   - `POST /api/v1/document/parse`: Generates structured layout AST with bounding boxes.
   - `POST /api/v1/document/chunk`: Layout-aware semantic chunking.
   - `POST /api/v1/document/qa`: Grounded question answering with visual citations.
5. **Interactive Document Lab UI (`DocumentOCRView.tsx`)**:
   - 2D Document Page Canvas rendering elements with color-coded bounding box overlays.
   - AST node metadata inspector.
   - Grounded Q&A Assistant highlighting cited table rows and bounding boxes in real time.

---

## 2. WHY It Exists (The Problem Solved)

### The Failure of Raw Text Extraction
Standard plain-text extractors (`pdf2text`, OCR without layout awareness) flatten 2D visual documents into 1D character streams. This leads to severe errors:
1. **Table Scrambling**: Columns merge into unreadable sentences (`"Revenue $45M Q3 $62M"`), making numerical analysis impossible.
2. **Formula Corruption**: LaTeX superscripts, subscripts, and fraction bars lose spatial meaning.
3. **Hierarchy Destruction**: Headings and key-value pairs (e.g. `Invoice Number: INV-8842`) are disconnected from their context.
4. **RAG Hallucinations**: Standard character-chunkers split table rows in half mid-number, leaving retrieval engines with fragmented, meaningless snippets.

---

## 3. HOW It Works (The Math & Mechanics)

### A. Normalized Spatial Bounding Boxes & 2D IoU Overlap
Every visual element on a page of width $W$ and height $H$ is normalized into unit coordinate space $[0, 1]^2$:
$$x_{\text{norm}} = \frac{x_{\text{pixel}}}{W}, \quad y_{\text{norm}} = \frac{y_{\text{pixel}}}{H}$$

For two bounding boxes $B_1$ and $B_2$, the Intersection-over-Union (IoU) overlap is computed as:
$$\text{IoU}(B_1, B_2) = \frac{\text{Area}(B_1 \cap B_2)}{\text{Area}(B_1 \cup B_2)} = \frac{\text{Area}(B_1 \cap B_2)}{\text{Area}(B_1) + \text{Area}(B_2) - \text{Area}(B_1 \cap B_2)}$$

### B. Table Grid Matrix Serialization
A tabular element $T$ is parsed into headers $H = [h_1, \dots, h_C]$ and data rows $R = [r_1, \dots, r_M]$, where each row $r_i = [c_{i,1}, \dots, c_{i,C}]$.
When an oversized table exceeds the maximum token ceiling $T_{\max}$, it is partitioned into sub-tables $T_1, T_2, \dots$ such that the header row $H$ is prepended to every split chunk:
$$T_k = H \parallel [r_j, \dots, r_{j+K}]$$
This guarantees that each vector chunk maintains full column header context for vector embedding models.

### C. Grounded Citations
Rather than generating an unanchored response, every answer point $A_k$ is linked to a citation tuple:
$$C_k = \langle \text{Page}, \text{ElementID}, \text{ElementType}, \text{CellRef}, \text{BBox} \rangle$$
This enables the UI to render an interactive highlight ring directly on the exact visual element on the page.

---

## 4. TEST Verification
The document understanding subsystem is verified across 4 test suites:
- `tests/document/test_layout_parser.py`:
  - Validates bounding box math (area, containment, IoU).
  - Tests markdown table grid roundtripping.
  - Tests multi-element layout extraction (headings, tables, equations, key-values).
- `tests/document/test_layout_chunker.py`:
  - Validates table preservation within chunks.
  - Tests oversized table row-level splitting with re-injected headers.
  - Verifies hierarchical breadcrumb propagation.
- `tests/document/test_document_qa.py`:
  - Tests grounded table cell extraction.
  - Tests key-value pair extraction.
  - Verifies visual citation metadata.
- `tests/api/test_document_endpoint.py`:
  - Validates `/presets`, `/parse`, `/chunk`, and `/qa` REST endpoints.

---

## 5. NEXT (Phase 41)
With structured document understanding in place, **Phase 41** focuses on **Multi-Turn Agentic Code Interpreter & Data Analysis Sandbox (LibraNotebook & Python Analytics Loop)**:
- Multi-cell stateful notebook session memory.
- Execution sandbox with data visualization outputs (SVG/PNG charts).
- Auto-analyzing tables extracted by Phase 40 directly in Python.
