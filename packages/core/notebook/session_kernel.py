"""
Libra Notebook Package - Stateful REPL Kernel & Session Manager

Maintains persistent execution memory, stateful variable scoping, execution order
tracking (In [x] / Out [x]), standard stream capture, and rich multi-MIME display.
"""

from __future__ import annotations

import ast
import contextlib
import io
import math
import random
import re
import statistics
import time
import uuid
from typing import Any, Dict, List, Optional

from packages.core.notebook.analytics import LibraTable
from packages.core.notebook.charts import ChartResult, LibraChart
from packages.core.notebook.security import NotebookSecurityError, NotebookSecurityPolicy


class ExecutionOutput:
    """Encapsulates the full execution result of a single notebook code cell."""

    def __init__(
        self,
        status: str,
        execution_count: int,
        stdout: str = "",
        stderr: str = "",
        result: Optional[str] = None,
        mime_outputs: Optional[Dict[str, str]] = None,
        duration_ms: float = 0.0,
        variables: Optional[List[Dict[str, Any]]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        self.status = status  # "ok" or "error"
        self.execution_count = execution_count
        self.stdout = stdout
        self.stderr = stderr
        self.result = result
        self.mime_outputs = mime_outputs or {}
        self.duration_ms = duration_ms
        self.variables = variables or []
        self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "execution_count": self.execution_count,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "result": self.result,
            "mime_outputs": self.mime_outputs,
            "duration_ms": round(self.duration_ms, 2),
            "variables": self.variables,
            "error_message": self.error_message,
        }


class NotebookKernel:
    """
    Isolated stateful REPL kernel for a single user/session.
    Variables and definitions created in earlier cells remain active in subsequent cells.
    """

    DEFAULT_IMPORTS = {
        "math": math,
        "random": random,
        "statistics": statistics,
        "re": re,
        "LibraTable": LibraTable,
        "LibraChart": LibraChart,
    }

    def __init__(self, session_id: str, timeout_sec: float = 5.0) -> None:
        self.session_id = session_id
        self.timeout_sec = timeout_sec
        self.execution_count = 0
        self.created_at = time.time()
        self.last_accessed = time.time()
        self._globals: Dict[str, Any] = {}
        self._initial_keys: set[str] = set()
        self.reset()

    def reset(self) -> None:
        """Flushes user state and restores default built-in analytics utilities."""
        self._globals = {
            "__name__": "__main__",
            "__doc__": None,
            "__builtins__": __builtins__,
        }
        self._globals.update(self.DEFAULT_IMPORTS)
        self._initial_keys = set(self._globals.keys())
        self.execution_count = 0
        self.last_accessed = time.time()

    def get_variables(self) -> List[Dict[str, Any]]:
        """Introspects user-defined variables in the active namespace."""
        vars_summary: List[Dict[str, Any]] = []
        for key, val in self._globals.items():
            if key.startswith("_") or key in self.DEFAULT_IMPORTS:
                continue

            val_type = type(val).__name__
            val_repr = repr(val)
            if len(val_repr) > 120:
                val_repr = val_repr[:117] + "..."

            size_info = None
            if hasattr(val, "shape"):
                size_info = f"shape={val.shape}"
            elif hasattr(val, "__len__"):
                size_info = f"len={len(val)}"

            vars_summary.append(
                {
                    "name": key,
                    "type": val_type,
                    "value_repr": val_repr,
                    "size": size_info,
                }
            )

        return sorted(vars_summary, key=lambda x: x["name"])

    def execute(self, code: str) -> ExecutionOutput:
        """
        Executes code within the kernel's persistent namespace.
        Splits execution into statements and an optional final evaluated expression.
        """
        self.last_accessed = time.time()
        start_time = time.perf_counter()

        # 1. AST Security Validation
        try:
            NotebookSecurityPolicy.validate(code)
        except (NotebookSecurityError, SyntaxError) as sec_err:
            duration = (time.perf_counter() - start_time) * 1000.0
            self.execution_count += 1
            return ExecutionOutput(
                status="error",
                execution_count=self.execution_count,
                stderr=str(sec_err),
                error_message=str(sec_err),
                duration_ms=duration,
                variables=self.get_variables(),
            )

        # 2. Parse into AST to separate trailing expression from statements
        try:
            parsed = ast.parse(code)
        except SyntaxError as syn_err:
            duration = (time.perf_counter() - start_time) * 1000.0
            self.execution_count += 1
            return ExecutionOutput(
                status="error",
                execution_count=self.execution_count,
                stderr=f"SyntaxError: {syn_err}",
                error_message=f"SyntaxError: {syn_err}",
                duration_ms=duration,
                variables=self.get_variables(),
            )

        # Intercept standard output and standard error
        stdout_io = io.StringIO()
        stderr_io = io.StringIO()
        eval_result: Any = None
        has_expr = False

        self.execution_count += 1

        try:
            with contextlib.redirect_stdout(stdout_io), contextlib.redirect_stderr(stderr_io):
                if parsed.body and isinstance(parsed.body[-1], ast.Expr):
                    # Last statement is an expression; execute statements prior to it
                    has_expr = True
                    stmt_nodes = parsed.body[:-1]
                    expr_node = parsed.body[-1]

                    if stmt_nodes:
                        stmt_mod = ast.Module(body=stmt_nodes, type_ignores=[])
                        stmt_code = compile(stmt_mod, filename="<cell>", mode="exec")
                        exec(stmt_code, self._globals)  # noqa: S102

                    # Evaluate final expression
                    expr_mod = ast.Expression(body=expr_node.value)
                    expr_code = compile(expr_mod, filename="<cell>", mode="eval")
                    eval_result = eval(expr_code, self._globals)  # noqa: S102
                else:
                    # Pure statements
                    full_code = compile(parsed, filename="<cell>", mode="exec")
                    exec(full_code, self._globals)  # noqa: S102

            duration = (time.perf_counter() - start_time) * 1000.0
            mime_outputs: Dict[str, str] = {}
            res_str: Optional[str] = None

            # 3. Format MIME outputs
            if has_expr and eval_result is not None:
                res_str = repr(eval_result)
                mime_outputs["text/plain"] = res_str

                # Rich SVG Display Hook
                if hasattr(eval_result, "_repr_svg_"):
                    mime_outputs["image/svg+xml"] = eval_result._repr_svg_()
                elif isinstance(eval_result, ChartResult):
                    mime_outputs["image/svg+xml"] = eval_result.to_svg()

                # Rich HTML Display Hook
                if hasattr(eval_result, "_repr_html_"):
                    mime_outputs["text/html"] = eval_result._repr_html_()

            return ExecutionOutput(
                status="ok",
                execution_count=self.execution_count,
                stdout=stdout_io.getvalue(),
                stderr=stderr_io.getvalue(),
                result=res_str,
                mime_outputs=mime_outputs,
                duration_ms=duration,
                variables=self.get_variables(),
            )

        except Exception as e:
            duration = (time.perf_counter() - start_time) * 1000.0
            err_msg = f"{type(e).__name__}: {e}"
            return ExecutionOutput(
                status="error",
                execution_count=self.execution_count,
                stdout=stdout_io.getvalue(),
                stderr=stderr_io.getvalue() + f"\n{err_msg}".strip(),
                error_message=err_msg,
                duration_ms=duration,
                variables=self.get_variables(),
            )


class NotebookSessionManager:
    """Manages active interactive notebook sessions and presets."""

    def __init__(self) -> None:
        self._sessions: Dict[str, NotebookKernel] = {}

    def get_or_create(self, session_id: Optional[str] = None) -> NotebookKernel:
        """Retrieves existing kernel or creates a new stateful session."""
        sid = session_id or str(uuid.uuid4())[:8]
        if sid not in self._sessions:
            self._sessions[sid] = NotebookKernel(session_id=sid)
        return self._sessions[sid]

    def get(self, session_id: str) -> Optional[NotebookKernel]:
        """Gets kernel by session ID if it exists."""
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Lists active sessions with summary metadata."""
        return [
            {
                "session_id": sid,
                "execution_count": kernel.execution_count,
                "created_at": kernel.created_at,
                "last_accessed": kernel.last_accessed,
                "variables_count": len(kernel.get_variables()),
            }
            for sid, kernel in self._sessions.items()
        ]

    def delete(self, session_id: str) -> bool:
        """Deletes session and flushes state."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def get_presets(self) -> List[Dict[str, Any]]:
        """Returns ready-to-run educational notebook templates."""
        return [
            {
                "id": "financial_sales_analytics",
                "title": "Quarterly Sales & Revenue Analysis",
                "description": "Ingest tabular transaction data, group by product category, calculate summary statistics, and generate bar chart.",
                "cells": [
                    {
                        "type": "markdown",
                        "content": "### 📊 Quarterly Sales & Revenue Analysis\nIn this notebook, we load sales transaction data into a `LibraTable`, run aggregations, and render an interactive vector SVG chart using `LibraChart`.",
                    },
                    {
                        "type": "code",
                        "content": (
                            "# Step 1: Create transaction dataset\n"
                            "raw_sales = [\n"
                            '    {"quarter": "Q1", "region": "North", "sales": 12500, "profit": 3200},\n'
                            '    {"quarter": "Q1", "region": "South", "sales": 9800,  "profit": 2100},\n'
                            '    {"quarter": "Q2", "region": "North", "sales": 14200, "profit": 3900},\n'
                            '    {"quarter": "Q2", "region": "South", "sales": 11500, "profit": 2800},\n'
                            '    {"quarter": "Q3", "region": "North", "sales": 16800, "profit": 4600},\n'
                            '    {"quarter": "Q3", "region": "South", "sales": 13100, "profit": 3400},\n'
                            '    {"quarter": "Q4", "region": "North", "sales": 19500, "profit": 5500},\n'
                            '    {"quarter": "Q4", "region": "South", "sales": 15400, "profit": 4100},\n'
                            "]\n\n"
                            "table = LibraTable.from_records(raw_sales)\n"
                            'print(f"Loaded table with shape: {table.shape}")\n'
                            "table"
                        ),
                    },
                    {
                        "type": "code",
                        "content": (
                            "# Step 2: Group by Quarter and calculate total sales & profit\n"
                            'quarterly = table.group_by("quarter", {"total_sales": "sales:sum", "total_profit": "profit:sum"})\n'
                            "quarterly"
                        ),
                    },
                    {
                        "type": "code",
                        "content": (
                            "# Step 3: Visualize Quarterly Revenue with LibraChart\n"
                            'quarters = quarterly.column_values("quarter")\n'
                            'revenue = quarterly.column_values("total_sales")\n\n'
                            "LibraChart.bar(\n"
                            "    categories=quarters,\n"
                            "    values=revenue,\n"
                            '    title="2025 Quarterly Total Revenue ($)",\n'
                            '    x_label="Quarter",\n'
                            '    y_label="Revenue ($)",\n'
                            '    color="#6366f1"\n'
                            ")"
                        ),
                    },
                ],
            },
            {
                "id": "model_loss_benchmarking",
                "title": "LLM Training Loss & Convergence Curves",
                "description": "Simulate training vs validation loss trajectories over training steps and render dual-trend curves.",
                "cells": [
                    {
                        "type": "markdown",
                        "content": "### 📉 Transformer Training & Validation Loss\nSimulate learning rate decay and exponential loss convergence curve.",
                    },
                    {
                        "type": "code",
                        "content": (
                            "# Generate synthetic training loss trajectory\n"
                            "steps = list(range(1, 21))\n"
                            "train_loss = [round(3.8 * math.exp(-0.12 * s) + 0.35 + random.uniform(-0.03, 0.03), 3) for s in steps]\n\n"
                            'print("Final step loss:", train_loss[-1])\n'
                            "LibraChart.line(\n"
                            "    x_values=steps,\n"
                            "    y_values=train_loss,\n"
                            '    title="Pretraining Loss Convergence (20 Steps)",\n'
                            '    x_label="Training Step",\n'
                            '    y_label="Cross-Entropy Loss",\n'
                            '    color="#06b6d4"\n'
                            ")"
                        ),
                    },
                ],
            },
            {
                "id": "statistical_distribution",
                "title": "Statistical Distribution & Histogram",
                "description": "Generate normal-like sample distributions, evaluate mean and variance, and render frequency histogram.",
                "cells": [
                    {
                        "type": "markdown",
                        "content": "### 🎲 Gaussian Sampling & Distribution Histogram\nCompute descriptive percentiles and plot distribution bins.",
                    },
                    {
                        "type": "code",
                        "content": (
                            "# Sample 100 values using Central Limit Theorem simulation\n"
                            "samples = [sum(random.uniform(0, 10) for _ in range(6)) for _ in range(120)]\n"
                            'stats_tbl = LibraTable.from_records([{"val": v} for v in samples])\n'
                            'print("Summary Statistics:")\n'
                            'for k, v in stats_tbl.describe()["val"].items():\n'
                            '    print(f"  {k}: {v}")\n\n'
                            "LibraChart.histogram(\n"
                            "    values=samples,\n"
                            "    bins=8,\n"
                            '    title="Sample Frequency Distribution",\n'
                            '    x_label="Binned Value",\n'
                            '    y_label="Count",\n'
                            '    color="#10b981"\n'
                            ")"
                        ),
                    },
                ],
            },
        ]


# Global session manager singleton
notebook_session_manager = NotebookSessionManager()
