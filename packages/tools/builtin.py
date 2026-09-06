"""
Libra Tools Package - Built-in Concrete Tools

Provides:
1. CalculatorTool: Safe AST arithmetic and mathematical evaluator.
2. PythonInterpreterTool: Sandboxed Python code executor.
3. WebSearchTool: Live web information retriever.
4. KnowledgeBaseTool: Hybrid RAG knowledge base searcher.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable
from pydantic import BaseModel, Field

from packages.tools.base import BaseTool
from packages.tools.sandbox import SafePythonSandbox
from packages.rag.hybrid import get_hybrid_retriever
from packages.rag.web_search import get_web_search_provider


# -----------------------------------------------------------------------------
# 1. Calculator Tool (AST Based Arithmetic)
# -----------------------------------------------------------------------------

class CalculatorInput(BaseModel):
    expression: str = Field(..., description="The mathematical expression to evaluate (e.g., '144 * 12', 'math.sqrt(256)', '2**16')")


SAFE_OPERATORS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_MATH_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "abs": abs,
    "round": round,
}

SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _eval_ast_math(node: ast.AST) -> float | int:
    """Recursively evaluates safe mathematical AST nodes."""
    if isinstance(node, ast.Expression):
        return _eval_ast_math(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left = _eval_ast_math(node.left)
        right = _eval_ast_math(node.right)
        return SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _eval_ast_math(node.operand)
        return SAFE_OPERATORS[op_type](operand)
    elif isinstance(node, ast.Call):
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "math":
                func_name = node.func.attr

        if func_name in SAFE_MATH_FUNCS:
            args = [_eval_ast_math(arg) for arg in node.args]
            return SAFE_MATH_FUNCS[func_name](*args)
        raise ValueError(f"Function call not permitted: '{func_name}'")
    elif isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "math":
            if node.attr in SAFE_CONSTANTS:
                return SAFE_CONSTANTS[node.attr]
        raise ValueError(f"Constant access not permitted: {node.attr}")
    elif isinstance(node, ast.Name):
        if node.id in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[node.id]
        raise ValueError(f"Variable not recognized: '{node.id}'")
    else:
        raise ValueError(f"Unsupported mathematical syntax node: {type(node).__name__}")


class CalculatorTool(BaseTool):
    """Accurately and safely evaluates mathematical expressions using AST parsing."""

    name = "calculator"
    description = "Safely evaluate mathematical and arithmetic expressions without eval vulnerabilities. Supports standard arithmetic, powers, and math functions (sqrt, sin, cos, log, etc.)."
    args_schema = CalculatorInput

    def _run(self, expression: str) -> dict[str, Any]:
        expr_clean = expression.strip()
        tree = ast.parse(expr_clean, mode="eval")
        result = _eval_ast_math(tree)
        # Format round floats if exact int
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return {"expression": expr_clean, "result": result}


# -----------------------------------------------------------------------------
# 2. Python Interpreter Tool
# -----------------------------------------------------------------------------

class PythonInterpreterInput(BaseModel):
    code: str = Field(..., description="The Python code snippet to execute in the secure sandbox")


class PythonInterpreterTool(BaseTool):
    """Executes Python code in an isolated security sandbox."""

    name = "python_interpreter"
    description = "Execute multi-line Python code in a safe, isolated sandbox with math, collections, random, and json. Returns printed stdout and the value of the final expression."
    args_schema = PythonInterpreterInput

    def __init__(self, timeout_sec: float = 5.0) -> None:
        self.sandbox = SafePythonSandbox(default_timeout=timeout_sec)

    def _run(self, code: str) -> dict[str, Any]:
        return self.sandbox.execute(code)


# -----------------------------------------------------------------------------
# 3. Web Search Tool
# -----------------------------------------------------------------------------

class WebSearchToolInput(BaseModel):
    query: str = Field(..., description="The search query to look up on the web")
    max_results: int = Field(5, ge=1, le=10, description="Maximum number of web search results to return")


class WebSearchTool(BaseTool):
    """Searches the live web using the active WebSearchProvider."""

    name = "web_search"
    description = "Search the web for real-time information, documentation, news, or technical references using DuckDuckGo."
    args_schema = WebSearchToolInput

    def _run(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        provider = get_web_search_provider()
        results = provider.search_sync(query, max_results=max_results)
        return [r.model_dump() for r in results]

    async def _arun(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        provider = get_web_search_provider()
        results = await provider.search(query, max_results=max_results)
        return [r.model_dump() for r in results]


# -----------------------------------------------------------------------------
# 4. Knowledge Base Tool
# -----------------------------------------------------------------------------

class KnowledgeBaseToolInput(BaseModel):
    query: str = Field(..., description="The query to search within the local indexed knowledge base")
    top_k: int = Field(3, ge=1, le=10, description="Number of document chunks to retrieve")
    mode: str = Field("hybrid", description="Retrieval mode: 'hybrid', 'dense', or 'bm25'")


class KnowledgeBaseTool(BaseTool):
    """Retrieves grounded document passages from the local Libra Knowledge Base."""

    name = "knowledge_base_search"
    description = "Search local indexed knowledge base documents using hybrid Okapi BM25 and dense vector search with Reciprocal Rank Fusion."
    args_schema = KnowledgeBaseToolInput

    def _run(self, query: str, top_k: int = 3, mode: str = "hybrid") -> list[dict[str, Any]]:
        retriever = get_hybrid_retriever()
        results = retriever.search(query=query, top_k=top_k, mode=mode)
        return [
            {
                "doc_title": r.chunk.doc_title,
                "text": r.chunk.text,
                "score": r.score,
                "rank": r.rank,
                "doc_id": r.chunk.doc_id,
            }
            for r in results
        ]
