"""
Libra Notebook Package - First-Principles Pure-Python SVG Vector Chart Engine

Generates standards-compliant SVG graphics (bar charts, line plots, scatter plots,
and histograms) with responsive viewports, clean dark-theme styling, gridlines,
and axes without relying on heavyweight GUI dependencies (e.g. matplotlib, Qt).
"""

from __future__ import annotations

import base64
import html
from typing import Any, Sequence


class ChartResult:
    """Encapsulates generated SVG markup with rich display hooks."""

    def __init__(self, svg_code: str, chart_type: str = "chart") -> None:
        self.svg_code = svg_code.strip()
        self.chart_type = chart_type

    def _repr_svg_(self) -> str:
        """IPython / Jupyter display hook for SVG MIME type."""
        return self.svg_code

    def to_svg(self) -> str:
        """Returns the raw SVG XML string."""
        return self.svg_code

    def to_data_uri(self) -> str:
        """Returns base64 data URI for direct embedding in <img> tags."""
        encoded = base64.b64encode(self.svg_code.encode("utf-8")).decode("ascii")
        return f"data:image/svg+xml;base64,{encoded}"

    def __repr__(self) -> str:
        return f"<LibraChart:{self.chart_type} length={len(self.svg_code)} bytes>"


class LibraChart:
    """Pure-Python SVG chart generator for scientific and data analytics visualization."""

    VIEW_WIDTH = 640
    VIEW_HEIGHT = 360
    PADDING_LEFT = 65
    PADDING_RIGHT = 30
    PADDING_TOP = 45
    PADDING_BOTTOM = 55

    @classmethod
    def _create_canvas(
        cls,
        title: str,
        x_label: str,
        y_label: str,
        y_min: float,
        y_max: float,
        y_ticks: int = 5,
    ) -> tuple[list[str], float, float, float, float]:
        """Builds background, title, gridlines, axes, and y-axis tick labels."""
        w = cls.VIEW_WIDTH
        h = cls.VIEW_HEIGHT
        pl = cls.PADDING_LEFT
        pr = cls.PADDING_RIGHT
        pt = cls.PADDING_TOP
        pb = cls.PADDING_BOTTOM

        plot_w = w - pl - pr
        plot_h = h - pt - pb

        elements: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'class="w-full h-auto max-w-[640px] rounded-lg border border-slate-800 bg-slate-950 font-sans shadow-lg">'
        ]

        # Definitions (gradients/glow)
        elements.append(
            "<defs>"
            '<linearGradient id="barGlow" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0%" stop-color="#6366f1" stop-opacity="0.9"/>'
            '<stop offset="100%" stop-color="#4338ca" stop-opacity="0.5"/>'
            "</linearGradient>"
            "</defs>"
        )

        # Title
        if title:
            safe_title = html.escape(title)
            elements.append(
                f'<text x="{w / 2}" y="{pt - 18}" text-anchor="middle" '
                f'fill="#f1f5f9" font-size="14" font-weight="600">{safe_title}</text>'
            )

        # Y Axis label
        if y_label:
            safe_ylabel = html.escape(y_label)
            elements.append(
                f'<text x="{-(pt + plot_h / 2)}" y="20" text-anchor="middle" '
                f'transform="rotate(-90)" fill="#94a3b8" font-size="11">{safe_ylabel}</text>'
            )

        # X Axis label
        if x_label:
            safe_xlabel = html.escape(x_label)
            elements.append(
                f'<text x="{pl + plot_w / 2}" y="{h - 12}" text-anchor="middle" '
                f'fill="#94a3b8" font-size="11">{safe_xlabel}</text>'
            )

        # Grid lines and Y ticks
        y_span = y_max - y_min if y_max != y_min else 1.0
        for i in range(y_ticks + 1):
            val = y_min + (y_span * i / y_ticks)
            y_pos = pt + plot_h - (i * plot_h / y_ticks)

            # Gridline
            elements.append(
                f'<line x1="{pl}" y1="{y_pos:.1f}" x2="{w - pr}" y2="{y_pos:.1f}" '
                f'stroke="#1e293b" stroke-width="1" stroke-dasharray="3,3"/>'
            )
            # Tick label
            elements.append(
                f'<text x="{pl - 8}" y="{y_pos + 4:.1f}" text-anchor="end" '
                f'fill="#64748b" font-size="10" font-family="monospace">{val:.1f}</text>'
            )

        # Axis Lines
        elements.append(
            f'<line x1="{pl}" y1="{pt}" x2="{pl}" y2="{pt + plot_h}" stroke="#334155" stroke-width="1.5"/>'
        )
        elements.append(
            f'<line x1="{pl}" y1="{pt + plot_h}" x2="{w - pr}" y2="{pt + plot_h}" stroke="#334155" stroke-width="1.5"/>'
        )

        return elements, pl, pt, plot_w, plot_h

    @classmethod
    def bar(
        cls,
        categories: Sequence[str],
        values: Sequence[float],
        title: str = "Bar Chart",
        x_label: str = "",
        y_label: str = "",
        color: str = "#6366f1",
    ) -> ChartResult:
        """Render a vertical bar chart."""
        cats = [str(c) for c in categories]
        vals = [float(v) for v in values]
        if not vals:
            return ChartResult(
                "<svg viewBox='0 0 400 100'><text x='20' y='50' fill='#94a3b8'>No data</text></svg>",
                "bar",
            )

        y_min = min(0.0, min(vals))
        y_max = max(vals) * 1.1 if max(vals) > 0 else 1.0
        if y_max == y_min:
            y_max += 1.0

        elements, pl, pt, plot_w, plot_h = cls._create_canvas(
            title=title, x_label=x_label, y_label=y_label, y_min=y_min, y_max=y_max
        )

        n = len(vals)
        bar_group_w = plot_w / n
        bar_w = min(bar_group_w * 0.7, 45.0)

        y_span = y_max - y_min
        zero_y = pt + plot_h - ((0.0 - y_min) / y_span * plot_h)

        for i, (cat, val) in enumerate(zip(cats, vals)):
            cx = pl + (i + 0.5) * bar_group_w
            bx = cx - bar_w / 2
            val_h = (val / y_span) * plot_h

            if val >= 0:
                by = zero_y - val_h
                bh = val_h
            else:
                by = zero_y
                bh = abs(val_h)

            elements.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" '
                f'rx="3" fill="{color}" opacity="0.88"/>'
            )
            # Value label
            elements.append(
                f'<text x="{cx:.1f}" y="{by - 4 if val >= 0 else by + bh + 12:.1f}" '
                f'text-anchor="middle" fill="#e2e8f0" font-size="9" font-family="monospace">{val:.1f}</text>'
            )
            # X label
            short_cat = cat if len(cat) <= 8 else cat[:7] + "…"
            elements.append(
                f'<text x="{cx:.1f}" y="{pt + plot_h + 16:.1f}" text-anchor="middle" '
                f'fill="#94a3b8" font-size="10">{html.escape(short_cat)}</text>'
            )

        elements.append("</svg>")
        return ChartResult("".join(elements), "bar")

    @classmethod
    def line(
        cls,
        x_values: Sequence[Any],
        y_values: Sequence[float],
        title: str = "Line Plot",
        x_label: str = "",
        y_label: str = "",
        color: str = "#06b6d4",
    ) -> ChartResult:
        """Render a continuous 2D line plot with data points."""
        xs = list(x_values)
        ys = [float(v) for v in y_values]
        if not ys or len(ys) < 2:
            return ChartResult(
                "<svg viewBox='0 0 400 100'><text x='20' y='50' fill='#94a3b8'>Insufficient data</text></svg>",
                "line",
            )

        y_min = min(ys) * 0.95 if min(ys) > 0 else min(ys) * 1.05
        y_max = max(ys) * 1.05 if max(ys) > 0 else max(ys) * 0.95
        if y_max == y_min:
            y_max += 1.0
            y_min -= 1.0

        elements, pl, pt, plot_w, plot_h = cls._create_canvas(
            title=title, x_label=x_label, y_label=y_label, y_min=y_min, y_max=y_max
        )

        n = len(ys)
        points: list[tuple[float, float]] = []
        y_span = y_max - y_min

        for i, val in enumerate(ys):
            px = pl + (i / (n - 1)) * plot_w
            py = pt + plot_h - ((val - y_min) / y_span * plot_h)
            points.append((px, py))

        # Polyline path
        path_data = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        elements.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" '
            f'stroke-linecap="round" stroke-linejoin="round" points="{path_data}"/>'
        )

        # Area gradient fill
        area_pts = (
            [f"{pl:.1f},{pt + plot_h:.1f}"]
            + [f"{x:.1f},{y:.1f}" for x, y in points]
            + [f"{pl + plot_w:.1f},{pt + plot_h:.1f}"]
        )
        elements.append(f'<polygon fill="{color}" opacity="0.12" points="{" ".join(area_pts)}"/>')

        # Draw markers & x-labels
        step = max(1, n // 7)
        for i, (px, py) in enumerate(points):
            elements.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="{color}" stroke="#0f172a" stroke-width="1.5"/>'
            )
            if i % step == 0 or i == n - 1:
                x_str = str(xs[i])
                short_x = x_str if len(x_str) <= 7 else x_str[:6] + "…"
                elements.append(
                    f'<text x="{px:.1f}" y="{pt + plot_h + 16:.1f}" text-anchor="middle" '
                    f'fill="#94a3b8" font-size="10">{html.escape(short_x)}</text>'
                )

        elements.append("</svg>")
        return ChartResult("".join(elements), "line")

    @classmethod
    def scatter(
        cls,
        x_values: Sequence[float],
        y_values: Sequence[float],
        title: str = "Scatter Plot",
        x_label: str = "X",
        y_label: str = "Y",
        color: str = "#ec4899",
    ) -> ChartResult:
        """Render a 2D scatter plot with continuous X and Y scales."""
        xs = [float(x) for x in x_values]
        ys = [float(y) for y in y_values]
        if not xs or not ys or len(xs) != len(ys):
            return ChartResult(
                "<svg viewBox='0 0 400 100'><text x='20' y='50' fill='#94a3b8'>Invalid scatter data</text></svg>",
                "scatter",
            )

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if x_max == x_min:
            x_max += 1.0
        if y_max == y_min:
            y_max += 1.0

        elements, pl, pt, plot_w, plot_h = cls._create_canvas(
            title=title, x_label=x_label, y_label=y_label, y_min=y_min, y_max=y_max
        )

        x_span = x_max - x_min
        y_span = y_max - y_min

        # X-axis ticks
        for i in range(6):
            tick_x_val = x_min + (x_span * i / 5)
            px = pl + (i * plot_w / 5)
            elements.append(
                f'<text x="{px:.1f}" y="{pt + plot_h + 16:.1f}" text-anchor="middle" '
                f'fill="#64748b" font-size="10" font-family="monospace">{tick_x_val:.1f}</text>'
            )

        # Points
        for x, y in zip(xs, ys):
            px = pl + ((x - x_min) / x_span * plot_w)
            py = pt + plot_h - ((y - y_min) / y_span * plot_h)
            elements.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}" opacity="0.8" '
                f'stroke="#ffffff" stroke-width="0.8"/>'
            )

        elements.append("</svg>")
        return ChartResult("".join(elements), "scatter")

    @classmethod
    def histogram(
        cls,
        values: Sequence[float],
        bins: int = 10,
        title: str = "Histogram",
        x_label: str = "Value Range",
        y_label: str = "Frequency",
        color: str = "#10b981",
    ) -> ChartResult:
        """Compute frequency bins and render an SVG histogram."""
        vals = [float(v) for v in values]
        if not vals:
            return ChartResult(
                "<svg viewBox='0 0 400 100'><text x='20' y='50' fill='#94a3b8'>No data</text></svg>",
                "histogram",
            )

        v_min, v_max = min(vals), max(vals)
        if v_min == v_max:
            v_max += 1.0

        bin_width = (v_max - v_min) / bins
        counts = [0] * bins

        for v in vals:
            idx = int((v - v_min) / bin_width)
            if idx >= bins:
                idx = bins - 1
            counts[idx] += 1

        categories = [f"{v_min + i * bin_width:.1f}" for i in range(bins)]
        return cls.bar(
            categories=categories,
            values=counts,
            title=title,
            x_label=x_label,
            y_label=y_label,
            color=color,
        )
