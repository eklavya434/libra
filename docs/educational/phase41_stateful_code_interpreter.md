# Phase 41: Stateful Code Interpreter & Data Analytics Sandbox (LibraNotebook)

> **Project Libra — First-Principles Educational LLM Laboratory**
> *Dual-Track Architecture: Interactive AI Product & First-Principles Foundation Lab*

---

## 1. WHAT: System Architecture & Deliverables

Phase 41 engineers **LibraNotebook**, a first-principles stateful code interpreter and interactive data analytics laboratory running 100% locally on CPU at zero cost ($0 / ₹0).

### Core Components Delivered:
1. **Stateful REPL Kernel (`NotebookKernel` & `NotebookSessionManager`)**:
   - Maintains execution memory (`_globals`) across cells, mimicking interactive Jupyter REPL behavior.
   - AST node partitioning: separates leading multi-line statements (`ast.Module`) from trailing expressions (`ast.Expression`) to automatically evaluate and display the cell's final value.
   - Dual stream redirection: captures standard output (`sys.stdout`) and standard error (`sys.stderr`) into formatted outputs without polluting the host terminal.
   - Execution counter tracking (`In [1]`, `In [2]`) with microsecond execution latency metrics.

2. **First-Principles In-Memory Tabular Engine (`LibraTable`)**:
   - Zero-dependency data structure supporting schema projection (`select`), predicate filtering (`filter`), multi-column sorting (`sort_by`), and row slicing (`head`, `tail`).
   - Group-by aggregation engine (`sum`, `mean`, `count`, `min`, `max`, `median`) with support for targeted column aggregation specs.
   - Descriptive summary statistics (`describe()`) computing count, mean, sample standard deviation, and five-number percentile summaries (min, 25%, 50%, 75%, max).
   - Multi-format serialization: GitHub-flavored Markdown tables, CSS-styled responsive HTML tables, and RFC 4180 CSV strings.

3. **Pure-Python SVG Vector Chart Generator (`LibraChart`)**:
   - Pure mathematical SVG rendering engine creating crisp vector graphics (`image/svg+xml`) without heavy external GUI or binary dependencies (e.g. Qt, matplotlib, libcairo).
   - Generates responsive `<svg>` viewports with coordinate transformations, dark-theme styling, linear gradients, gridlines, axis ticks, and data labels.
   - Supported chart types:
     - **Bar Chart** (`LibraChart.bar`): Discrete categorical bars with value badges.
     - **Line Plot** (`LibraChart.line`): Continuous polyline trends with smooth area fills and data point markers.
     - **Scatter Plot** (`LibraChart.scatter`): 2D continuous Cartesian coordinate distributions.
     - **Histogram** (`LibraChart.histogram`): Automatic frequency binning and interval distribution visualization.
   - Native Jupyter / MIME hook support via `_repr_svg_()` and `_repr_html_()`.

4. **Multi-Layer AST Security Boundary (`NotebookSecurityPolicy`)**:
   - Static AST validation prohibiting shell execution (`os`, `subprocess`), reflection escapes (`__subclasses__`, `__mro__`, `__globals__`), socket networking (`socket`), and dynamic code evaluation (`eval`, `exec`, `compile`).
   - Allowlist restricted to standard mathematical and data analysis libraries (`math`, `statistics`, `random`, `json`, `csv`, `re`, `datetime`, `collections`, `itertools`, `LibraTable`, `LibraChart`).

5. **FastAPI Endpoints (`apps/backend/api/v1/endpoints/notebook.py`)**:
   - `POST /api/v1/notebook/sessions`: Create or get stateful kernel session.
   - `GET /api/v1/notebook/sessions`: List active sessions and execution metadata.
   - `GET /api/v1/notebook/sessions/{session_id}`: Inspect session state and namespace.
   - `DELETE /api/v1/notebook/sessions/{session_id}`: Terminate session.
   - `POST /api/v1/notebook/sessions/{session_id}/execute`: Execute code cell in stateful session.
   - `GET /api/v1/notebook/sessions/{session_id}/variables`: Namespace variable inspection.
   - `POST /api/v1/notebook/sessions/{session_id}/reset`: Reset namespace to initial state.
   - `GET /api/v1/notebook/presets`: Fetch educational analytics templates.

6. **Interactive Jupyter-Style Notebook UI (`apps/frontend/src/components/NotebookView.tsx`)**:
   - Multi-cell canvas with syntax-accented code cells and markdown cells.
   - Execution controls: individual cell run (`Shift+Enter`), "Run All", restart kernel, cell reordering (`Up`/`Down`), and delete.
   - Output renderer supporting stdout, stderr, rich HTML data tables, and embedded SVG charts.
   - Live Variable Explorer sidebar inspecting active variable types, shapes, and value representations.

---

## 2. WHY: Problem Space & Theoretical Motivation

Modern LLM-assisted data analysis (such as ChatGPT Advanced Data Analysis) requires **stateful execution**. Stateless code runners suffer from significant flaws:

1. **Loss of Intermediate State**: If every code snippet executes in a fresh process, users cannot build exploratory workflows step-by-step (e.g. load dataset in Cell 1, filter outliers in Cell 2, render chart in Cell 3).
2. **Heavy Dependencies & Overhead**: Standard Python analytics environments often depend on hundreds of megabytes of C-extensions (numpy, pandas, matplotlib) that strain disk quotas and require compilation toolchains. Building `LibraTable` and `LibraChart` from first principles guarantees lightweight, instant, and reproducible execution on consumer CPUs without external binary bloat.
3. **Security in Stateful Scenarios**: Unlike one-shot sandbox processes that can be killed immediately after running, stateful kernels keep memory alive. They must enforce AST analysis before execution to prevent namespace pollution and sandbox escapes.

---

## 3. HOW: Mathematical & Algorithmic Implementation

### A. AST Separation: Statements vs. Trailing Expression
When executing arbitrary multi-line Python code, standard `exec()` discards the value of the final expression:
```python
# In standard exec():
exec("x = 10\nx + 5")  # Returns None!
```
To replicate IPython / Jupyter behavior, `NotebookKernel` parses the code into an Abstract Syntax Tree (`ast.parse`):
```python
parsed = ast.parse(code)
if parsed.body and isinstance(parsed.body[-1], ast.Expr):
    # 1. Compile statements [0 : -1] in 'exec' mode
    stmt_mod = ast.Module(body=parsed.body[:-1], type_ignores=[])
    exec(compile(stmt_mod, filename="<cell>", mode="exec"), self._globals)

    # 2. Compile trailing expression in 'eval' mode
    expr_mod = ast.Expression(body=parsed.body[-1].value)
    eval_result = eval(compile(expr_mod, filename="<cell>", mode="eval"), self._globals)
```
If `eval_result` defines `_repr_svg_()` or `_repr_html_()`, the kernel routes the markup into `mime_outputs` for rich browser display.

### B. First-Principles Percentile Calculation
In `LibraTable.describe()`, quantiles are computed using linear interpolation between order statistics without external libraries:
For sorted sample array $X = (x_{(0)}, x_{(1)}, \dots, x_{(n-1)})$ and quantile $p \in [0, 1]$:
$$k = (n - 1) \cdot p$$
$$i = \lfloor k \rfloor, \quad f = k - i$$
$$Q(p) = x_{(i)} + f \cdot (x_{(i+1)} - x_{(i)})$$

Sample standard deviation uses Bessel's correction:
$$s = \sqrt{\frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2}$$

### C. SVG Vector Coordinate Projection
In `LibraChart`, continuous data values are mapped into pixel viewport coordinates $(x_{\text{pixel}}, y_{\text{pixel}})$ using linear affine transformations:
For a plot region spanning $x \in [x_{\min}, x_{\max}]$ with viewport padding $P_{\text{left}}$ and plot width $W_{\text{plot}}$:
$$x_{\text{pixel}} = P_{\text{left}} + \frac{x - x_{\min}}{x_{\max} - x_{\min}} \cdot W_{\text{plot}}$$
For $y \in [y_{\min}, y_{\max}]$ (where SVG $y$ coordinates grow downward):
$$y_{\text{pixel}} = P_{\text{top}} + W_{\text{plot\_height}} - \frac{y - y_{\min}}{y_{\max} - y_{\min}} \cdot W_{\text{plot\_height}}$$

---

## 4. TEST: Verification & Regression Results

1. **Phase 41 Unit Tests**:
   - `tests/notebook/test_session_kernel.py`: Verified basic execution, variable persistence across cells, stdout/stderr capture, kernel reset, and AST security rejections (7 tests).
   - `tests/notebook/test_analytics_and_charts.py`: Verified `LibraTable` (filtering, group by aggregations, descriptive stats, CSV) and `LibraChart` SVG generation (bar, line, scatter, histogram, rich MIME display) (4 tests).
   - `tests/api/test_notebook_endpoint.py`: Verified session lifecycle, cell execution, variable inspection, presets, and security rejection responses (4 tests).
   - **All 15 Phase 41 tests passed in 6.18s**.

2. **Full Regression Test Suite**:
   - **469 passed, 0 failed** across all 41 phases.

3. **Frontend Production Build**:
   - Next.js 14 App Router compiled cleanly with zero TypeScript or lint errors.

---

## 5. NEXT: Upcoming Roadmap

With Phase 41 complete, Project Libra possesses a full interactive code interpreter and visual data analytics loop.
The curriculum transitions to:
- **Phase 42: Long-Context Needle-in-a-Haystack (NIAH) & Dynamic Attention Compaction**:
  - Stress testing context windows up to 32k tokens.
  - Needle retrieval evaluation, precision-recall curve across context depths.
  - Dynamic token importance sparsification and KV eviction for CPU efficiency.
