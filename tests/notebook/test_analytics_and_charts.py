"""
Unit tests for LibraTable analytics engine and LibraChart SVG generator.
"""

from packages.core.notebook.analytics import LibraTable
from packages.core.notebook.charts import LibraChart
from packages.core.notebook.session_kernel import NotebookKernel


def test_libra_table_creation_and_filtering():
    raw_data = [
        {"name": "Alice", "dept": "Engineering", "salary": 120000},
        {"name": "Bob", "dept": "Sales", "salary": 95000},
        {"name": "Charlie", "dept": "Engineering", "salary": 110000},
        {"name": "Diana", "dept": "Design", "salary": 105000},
    ]
    table = LibraTable.from_records(raw_data)
    assert table.shape == (4, 3)
    assert "salary" in table.columns

    # Filter
    eng = table.filter(lambda r: r["dept"] == "Engineering")
    assert len(eng) == 2
    assert eng.column_values("name") == ["Alice", "Charlie"]

    # Select
    projected = table.select("name", "salary")
    assert projected.columns == ["name", "salary"]

    # Sort
    sorted_tbl = table.sort_by("salary", reverse=True)
    assert sorted_tbl.column_values("name")[0] == "Alice"
    assert sorted_tbl.column_values("name")[-1] == "Bob"


def test_libra_table_csv_and_aggregations():
    csv_data = """product,category,revenue,units
Widget A,Hardware,250.5,10
Widget B,Software,400.0,5
Widget C,Hardware,150.0,8
Widget D,Software,600.0,12
"""
    table = LibraTable.from_csv(csv_data)
    assert len(table) == 4

    # Group by
    grouped = table.group_by("category", {"total_rev": "revenue:sum", "avg_units": "units:mean"})
    assert len(grouped) == 2

    rev_map = {r["category"]: r["total_rev"] for r in grouped.to_dict()}
    assert rev_map["Hardware"] == 400.5
    assert rev_map["Software"] == 1000.0

    # Describe
    stats = table.describe()
    assert "revenue" in stats
    assert stats["revenue"]["count"] == 4.0
    assert stats["revenue"]["min"] == 150.0
    assert stats["revenue"]["max"] == 600.0

    # Serialization
    md = table.to_markdown()
    assert "| Widget A |" in md

    html_out = table.to_html()
    assert "<table" in html_out
    assert "Widget B" in html_out


def test_libra_chart_svg_generation():
    # Bar Chart
    bar_chart = LibraChart.bar(
        categories=["Jan", "Feb", "Mar"],
        values=[10.5, 24.2, 18.0],
        title="Quarterly Trend",
    )
    svg_bar = bar_chart.to_svg()
    assert "<svg" in svg_bar
    assert "</svg>" in svg_bar
    assert "<rect" in svg_bar
    assert "Quarterly Trend" in svg_bar

    # Line Plot
    line_chart = LibraChart.line(
        x_values=[1, 2, 3, 4],
        y_values=[5.0, 8.2, 12.1, 15.4],
        title="Growth Curve",
    )
    svg_line = line_chart.to_svg()
    assert "<polyline" in svg_line
    assert "Growth Curve" in svg_line

    # Scatter Plot
    scatter_chart = LibraChart.scatter(
        x_values=[1.0, 2.5, 3.8],
        y_values=[10.0, 15.0, 25.0],
        title="Correlation",
    )
    svg_scatter = scatter_chart.to_svg()
    assert "<circle" in svg_scatter
    assert "Correlation" in svg_scatter

    # Histogram
    hist_chart = LibraChart.histogram(
        values=[1.2, 1.5, 2.1, 2.8, 3.1, 3.3, 3.9, 4.0],
        bins=4,
        title="Distribution",
    )
    svg_hist = hist_chart.to_svg()
    assert "<rect" in svg_hist


def test_kernel_rich_mime_display():
    kernel = NotebookKernel(session_id="test_mime")

    # Rich HTML display from LibraTable
    out1 = kernel.execute("tbl = LibraTable.from_records([{'x': 1, 'y': 2}])\ntbl")
    assert out1.status == "ok"
    assert "text/html" in out1.mime_outputs
    assert "<table" in out1.mime_outputs["text/html"]

    # Rich SVG display from LibraChart
    out2 = kernel.execute("chart = LibraChart.bar(['A', 'B'], [10, 20])\nchart")
    assert out2.status == "ok"
    assert "image/svg+xml" in out2.mime_outputs
    assert "<svg" in out2.mime_outputs["image/svg+xml"]
