"""
Tests for Builtin Tools (packages/tools/builtin.py)

Validates CalculatorTool, PythonInterpreterTool, WebSearchTool, and KnowledgeBaseTool.
"""

import packages.rag.web_search as web_search_module
from packages.rag.hybrid import get_hybrid_retriever
from packages.rag.web_search import MockSearchProvider
from packages.tools.builtin import (
    CalculatorTool,
    KnowledgeBaseTool,
    PythonInterpreterTool,
    WebSearchTool,
)


def test_calculator_tool_arithmetic():
    calc = CalculatorTool()
    res = calc.execute(expression="144 / 12")
    assert res.success is True
    assert res.output["result"] == 12

    res = calc.execute(expression="2**8 + 10")
    assert res.success is True
    assert res.output["result"] == 266


def test_calculator_tool_math_functions():
    calc = CalculatorTool()
    res = calc.execute(expression="sqrt(256)")
    assert res.success is True
    assert res.output["result"] == 16

    res = calc.execute(expression="log2(32)")
    assert res.success is True
    assert res.output["result"] == 5

    res = calc.execute(expression="factorial(5)")
    assert res.success is True
    assert res.output["result"] == 120


def test_calculator_tool_security_rejections():
    calc = CalculatorTool()
    # Code injection attempts should fail safely
    res = calc.execute(expression="__import__('os').system('dir')")
    assert res.success is False
    assert res.error is not None

    res = calc.execute(expression="open('test.txt')")
    assert res.success is False


def test_python_interpreter_tool_execution():
    tool = PythonInterpreterTool(timeout_sec=2.0)
    res = tool.execute(
        code="""
data = [1, 2, 3, 4, 5]
squared = [x**2 for x in data]
print("Sum:", sum(squared))
sum(squared)
"""
    )
    assert res.success is True
    assert res.output["success"] is True
    assert res.output["result"] == 55
    assert "Sum: 55" in res.output["stdout"]


def test_python_interpreter_tool_blocks_prohibited_actions():
    tool = PythonInterpreterTool()
    res = tool.execute(code="import os\nos.system('echo hacked')")
    assert res.success is False
    assert "Security policy violation" in res.error


def test_web_search_tool(monkeypatch):
    # Configure mock search provider
    mock_provider = MockSearchProvider()
    monkeypatch.setattr(web_search_module, "_global_web_search_provider", mock_provider)

    tool = WebSearchTool()
    res = tool.execute(query="transformer", max_results=2)
    assert res.success is True
    assert isinstance(res.output, list)
    assert len(res.output) >= 1
    assert "title" in res.output[0]
    assert "url" in res.output[0]


def test_knowledge_base_tool():
    retriever = get_hybrid_retriever()
    retriever.add_document(
        title="Tool Use Manual",
        content="Tool use enables language models to interact with external APIs, calculators, and sandboxes.",
        metadata={"category": "ai"},
    )

    kb_tool = KnowledgeBaseTool()
    res = kb_tool.execute(query="external APIs calculators sandboxes", top_k=2)
    assert res.success is True
    assert isinstance(res.output, list)
    assert len(res.output) >= 1
    assert "Tool Use Manual" in res.output[0]["doc_title"]
