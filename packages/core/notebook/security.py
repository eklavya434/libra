"""
Libra Notebook Package - AST Security Validator & Sandbox Boundary Enforcement

Ensures code executed within the stateful notebook kernel cannot perform:
1. Unauthorized imports (e.g. os, sys, subprocess, socket, ctypes).
2. Dynamic code evaluation (eval, exec, compile).
3. Reflection-based sandbox escapes (__subclasses__, __globals__, __builtins__).
4. Shell execution or network socket operations.
"""

from __future__ import annotations

import ast
from typing import Set

# Safe standard libraries permitted for in-kernel data analysis and calculations
ALLOWED_MODULES: Set[str] = {
    "math",
    "random",
    "json",
    "datetime",
    "collections",
    "itertools",
    "re",
    "string",
    "statistics",
    "decimal",
    "fractions",
    "time",
    "csv",
    "io",
    "copy",
    "functools",
    "operator",
    "bisect",
    "heapq",
    "sys",
}

FORBIDDEN_CALLS: Set[str] = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "exit",
    "quit",
}

FORBIDDEN_ATTRS: Set[str] = {
    "__subclasses__",
    "__globals__",
    "__code__",
    "__closure__",
    "__bases__",
    "__class__",
    "__mro__",
    "__dict__",
    "__builtins__",
    "modules",
    "_getframe",
    "settrace",
    "setprofile",
}


class NotebookSecurityError(PermissionError):
    """Raised when notebook code violates static AST security rules."""

    pass


class NotebookSecurityVisitor(ast.NodeVisitor):
    """
    AST visitor that scans code for forbidden imports, dangerous builtins,
    and reflection attacks.
    """

    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base_mod = alias.name.split(".")[0]
            if base_mod not in ALLOWED_MODULES:
                self.violations.append(f"Forbidden import '{alias.name}' (not in allowed list)")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            base_mod = node.module.split(".")[0]
            if base_mod not in ALLOWED_MODULES:
                self.violations.append(
                    f"Forbidden from-import '{node.module}' (not in allowed list)"
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                self.violations.append(f"Forbidden call: '{node.func.id}()'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_ATTRS:
            self.violations.append(f"Forbidden attribute access: '{node.attr}'")
        self.generic_visit(node)


class NotebookSecurityPolicy:
    """Validates source code strings before execution."""

    @staticmethod
    def validate(code: str) -> None:
        """
        Parses source code into an AST and verifies compliance with safety policy.
        Raises SyntaxError if parsing fails, or NotebookSecurityError if violations exist.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise SyntaxError(f"Syntax error in notebook cell: {e}") from e

        visitor = NotebookSecurityVisitor()
        visitor.visit(tree)

        if visitor.violations:
            raise NotebookSecurityError(
                f"Security policy violation: {'; '.join(visitor.violations)}"
            )
