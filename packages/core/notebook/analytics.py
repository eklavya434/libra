"""
Libra Notebook Package - First-Principles In-Memory Tabular Data Analytics Engine

Implements `LibraTable`, a lightweight zero-dependency DataFrame engineered from
first principles for statistical exploration, filtering, aggregation, and display.
"""

from __future__ import annotations

import csv
import io
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


class LibraTable:
    """
    In-memory columnar and tabular data structure providing SQL/pandas-like
    operations without external C-extensions.
    """

    def __init__(
        self,
        data: Sequence[Dict[str, Any]],
        columns: Optional[Sequence[str]] = None,
    ) -> None:
        self._data: List[Dict[str, Any]] = [dict(row) for row in data]
        if columns is not None:
            self._columns: List[str] = list(columns)
        elif self._data:
            self._columns = list(self._data[0].keys())
        else:
            self._columns = []

    @property
    def columns(self) -> List[str]:
        return list(self._columns)

    @property
    def shape(self) -> Tuple[int, int]:
        """Returns (num_rows, num_columns)."""
        return len(self._data), len(self._columns)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: Union[int, str]) -> Any:
        if isinstance(item, int):
            return self._data[item]
        elif isinstance(item, str):
            return [row.get(item) for row in self._data]
        raise TypeError(f"Invalid index type: {type(item)}")

    @classmethod
    def from_records(cls, records: Sequence[Dict[str, Any]]) -> LibraTable:
        """Create table from list of dictionary records."""
        return cls(records)

    @classmethod
    def from_rows(cls, rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> LibraTable:
        """Create table from 2D row data and column headers."""
        headers_list = list(headers)
        data = []
        for row in rows:
            data.append(
                {headers_list[i]: val for i, val in enumerate(row) if i < len(headers_list)}
            )
        return cls(data, columns=headers_list)

    @classmethod
    def from_csv(cls, csv_text: str) -> LibraTable:
        """Parse raw CSV string into a typed LibraTable."""
        f = io.StringIO(csv_text.strip())
        reader = csv.DictReader(f)
        data: List[Dict[str, Any]] = []
        columns = list(reader.fieldnames or [])

        for row in reader:
            parsed_row: Dict[str, Any] = {}
            for k, v in row.items():
                if v is None:
                    parsed_row[k] = None
                    continue
                v_str = str(v).strip()
                # Try integer
                try:
                    parsed_row[k] = int(v_str)
                    continue
                except ValueError:
                    pass
                # Try float
                try:
                    parsed_row[k] = float(v_str)
                    continue
                except ValueError:
                    pass
                # Try boolean
                if v_str.lower() in ("true", "false"):
                    parsed_row[k] = v_str.lower() == "true"
                    continue
                # Keep as string
                parsed_row[k] = v_str
            data.append(parsed_row)
        return cls(data, columns=columns)

    def select(self, *columns: str) -> LibraTable:
        """Project table down to the specified column subset."""
        target_cols = [c for c in columns if c in self._columns]
        new_data = [{c: row.get(c) for c in target_cols} for row in self._data]
        return LibraTable(new_data, columns=target_cols)

    def filter(self, predicate: Callable[[Dict[str, Any]], bool]) -> LibraTable:
        """Filter rows matching the boolean predicate callable."""
        new_data = [row for row in self._data if predicate(row)]
        return LibraTable(new_data, columns=self._columns)

    def sort_by(self, column: str, reverse: bool = False) -> LibraTable:
        """Sort rows by the specified column values."""
        if column not in self._columns:
            raise KeyError(f"Column '{column}' not in table columns {self._columns}")

        def _sort_key(row: Dict[str, Any]) -> Any:
            val = row.get(column)
            if val is None:
                # Place None at end
                return (1, 0)
            return (0, val)

        sorted_data = sorted(self._data, key=_sort_key, reverse=reverse)
        return LibraTable(sorted_data, columns=self._columns)

    def head(self, n: int = 5) -> LibraTable:
        """Return the first n rows."""
        return LibraTable(self._data[:n], columns=self._columns)

    def tail(self, n: int = 5) -> LibraTable:
        """Return the last n rows."""
        return LibraTable(self._data[-n:], columns=self._columns)

    def column_values(self, column: str) -> List[Any]:
        """Extract column as list of values."""
        if column not in self._columns:
            raise KeyError(f"Column '{column}' not in table columns")
        return [row.get(column) for row in self._data]

    def add_column(
        self,
        name: str,
        fn_or_values: Union[Callable[[Dict[str, Any]], Any], Sequence[Any]],
    ) -> LibraTable:
        """Return a new table with a new computed or provided column."""
        new_columns = list(self._columns)
        if name not in new_columns:
            new_columns.append(name)

        new_data = []
        if callable(fn_or_values):
            for row in self._data:
                r = dict(row)
                r[name] = fn_or_values(r)
                new_data.append(r)
        else:
            values_list = list(fn_or_values)
            for idx, row in enumerate(self._data):
                r = dict(row)
                r[name] = values_list[idx] if idx < len(values_list) else None
                new_data.append(r)

        return LibraTable(new_data, columns=new_columns)

    def group_by(
        self,
        by_column: str,
        aggregations: Dict[str, str],
    ) -> LibraTable:
        """
        Group rows by a key column and compute aggregate metrics.
        Supported aggregations: "sum", "mean", "avg", "count", "min", "max", "median".
        """
        if by_column not in self._columns:
            raise KeyError(f"Group-by column '{by_column}' not found")

        groups: Dict[Any, List[Dict[str, Any]]] = {}
        for row in self._data:
            key = row.get(by_column)
            if key not in groups:
                groups[key] = []
            groups[key].append(row)

        output_rows: List[Dict[str, Any]] = []
        output_cols = [by_column] + list(aggregations.keys())

        for key, group_rows in groups.items():
            result_row: Dict[str, Any] = {by_column: key}
            for out_col, agg_fn in aggregations.items():
                agg_type = agg_fn.lower().strip()
                # If aggregation is like "col:sum", parse target column, else default to out_col
                target_col = out_col
                if ":" in agg_type:
                    target_col, agg_type = agg_type.split(":", 1)

                vals = [r.get(target_col) for r in group_rows if r.get(target_col) is not None]
                num_vals = [float(v) for v in vals if isinstance(v, (int, float))]

                if agg_type == "count":
                    result_row[out_col] = len(group_rows)
                elif not num_vals:
                    result_row[out_col] = None
                elif agg_type in ("sum", "total"):
                    result_row[out_col] = round(sum(num_vals), 4)
                elif agg_type in ("mean", "avg"):
                    result_row[out_col] = round(sum(num_vals) / len(num_vals), 4)
                elif agg_type == "min":
                    result_row[out_col] = min(num_vals)
                elif agg_type == "max":
                    result_row[out_col] = max(num_vals)
                elif agg_type == "median":
                    sorted_v = sorted(num_vals)
                    mid = len(sorted_v) // 2
                    if len(sorted_v) % 2 == 0:
                        med = (sorted_v[mid - 1] + sorted_v[mid]) / 2.0
                    else:
                        med = sorted_v[mid]
                    result_row[out_col] = round(med, 4)
                else:
                    raise ValueError(f"Unsupported aggregation: '{agg_type}'")

            output_rows.append(result_row)

        return LibraTable(output_rows, columns=output_cols)

    def describe(self) -> Dict[str, Dict[str, float]]:
        """
        Calculates summary statistics for all numeric columns.
        Returns: {col_name: {count, mean, std, min, 25%, 50%, 75%, max}}
        """
        stats: Dict[str, Dict[str, float]] = {}
        for col in self._columns:
            vals = [float(r[col]) for r in self._data if isinstance(r.get(col), (int, float))]
            if not vals:
                continue
            n = len(vals)
            mean_val = sum(vals) / n
            variance = sum((x - mean_val) ** 2 for x in vals) / (n - 1) if n > 1 else 0.0
            std_val = math.sqrt(variance)
            sorted_v = sorted(vals)

            def _percentile(p: float) -> float:
                k = (n - 1) * p
                f = math.floor(k)
                c = math.ceil(k)
                if f == c:
                    return sorted_v[int(k)]
                return sorted_v[f] * (c - k) + sorted_v[c] * (k - f)

            stats[col] = {
                "count": float(n),
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "min": round(sorted_v[0], 4),
                "25%": round(_percentile(0.25), 4),
                "50%": round(_percentile(0.50), 4),
                "75%": round(_percentile(0.75), 4),
                "max": round(sorted_v[-1], 4),
            }
        return stats

    def to_markdown(self) -> str:
        """Render table as a GitHub-flavored markdown table."""
        if not self._columns:
            return "_Empty LibraTable_"

        lines = [
            "| " + " | ".join(self._columns) + " |",
            "| " + " | ".join(["---"] * len(self._columns)) + " |",
        ]
        for row in self._data:
            lines.append("| " + " | ".join(str(row.get(c, "")) for c in self._columns) + " |")
        return "\n".join(lines)

    def to_html(self) -> str:
        """Render stylized HTML table with dark theme CSS classes."""
        if not self._columns:
            return "<div class='text-xs text-slate-500 italic'>Empty LibraTable</div>"

        headers_html = "".join(
            f"<th class='px-3 py-2 text-left text-xs font-semibold text-slate-300 uppercase tracking-wider bg-slate-800/80 border-b border-slate-700'>{col}</th>"
            for col in self._columns
        )
        rows_html = []
        for i, row in enumerate(self._data):
            bg = "bg-slate-900/40" if i % 2 == 0 else "bg-slate-900/80"
            cells = "".join(
                f"<td class='px-3 py-1.5 text-xs text-slate-200 border-b border-slate-800/60 font-mono'>{row.get(c, '')}</td>"
                for c in self._columns
            )
            rows_html.append(
                f"<tr class='{bg} hover:bg-indigo-950/30 transition-colors'>{cells}</tr>"
            )

        return (
            "<div class='overflow-x-auto rounded-lg border border-slate-800 shadow-md my-2'>"
            "<table class='min-w-full divide-y divide-slate-800 text-left'>"
            f"<thead><tr>{headers_html}</tr></thead>"
            f"<tbody class='divide-y divide-slate-800/40'>{''.join(rows_html)}</tbody>"
            "</table>"
            f"<div class='px-3 py-1 bg-slate-950/80 text-[10px] text-slate-400 font-mono border-t border-slate-800/60'>"
            f"{len(self._data)} rows × {len(self._columns)} columns"
            "</div></div>"
        )

    def _repr_html_(self) -> str:
        """Jupyter / rich display protocol hook."""
        return self.to_html()

    def to_csv(self) -> str:
        """Serialize table to CSV format."""
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=self._columns)
        writer.writeheader()
        writer.writerows(self._data)
        return out.getvalue()

    def to_dict(self) -> List[Dict[str, Any]]:
        """Return table data as a list of dictionaries."""
        return [dict(r) for r in self._data]

    def __repr__(self) -> str:
        return f"LibraTable({len(self._data)} rows, {len(self._columns)} cols: {', '.join(self._columns[:4])}{'...' if len(self._columns) > 4 else ''})"
