"""
Libra Tools Package - Isolated Child Process Runner Script

Executed as an independent OS process to run sandboxed Python snippets.
Communicates via standard I/O (JSON config in stdin, JSON result in stdout).
"""

from __future__ import annotations

import ast
import collections
import datetime
import decimal
import fractions
import io
import itertools
import json
import math
import os
import random
import re
import statistics
import string
import sys
import time
import traceback


def main() -> None:
    try:
        raw_config = sys.stdin.read()
        config = json.loads(raw_config)
    except Exception as e:
        sys.stdout.write(
            json.dumps(
                {
                    "stdout": "",
                    "result": None,
                    "success": False,
                    "error": f"Failed to initialize sandbox runner: {e}",
                }
            )
        )
        sys.stdout.flush()
        return

    code = config.get("code", "")
    sandbox_dir = os.path.realpath(config.get("sandbox_dir", os.getcwd()))

    # 1. Scoped safe_open function that strictly prohibits filesystem escapes
    def safe_open(file: str | os.PathLike, mode: str = "r", *args, **kwargs):
        target_path = os.path.realpath(os.path.join(sandbox_dir, str(file)))
        try:
            common = os.path.commonpath([sandbox_dir, target_path])
        except ValueError:
            raise PermissionError(
                f"Access denied: path '{file}' is outside the sandbox drive"
            )
        if common != sandbox_dir:
            raise PermissionError(
                f"Access denied: path '{file}' attempts to escape scoped sandbox directory"
            )
        return open(target_path, mode, *args, **kwargs)

    # 2. Module allowlist mapper
    allowed_modules = {
        "math": math,
        "random": random,
        "json": json,
        "datetime": datetime,
        "collections": collections,
        "itertools": itertools,
        "re": re,
        "string": string,
        "statistics": statistics,
        "decimal": decimal,
        "fractions": fractions,
        "time": time,
    }

    def safe_import(name: str, *args, **kwargs):
        base_name = name.split(".")[0]
        if base_name in allowed_modules:
            return allowed_modules[base_name]
        raise ImportError(f"Importing '{name}' is forbidden in sandbox environment")

    # 3. Restricted builtins
    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "round": round,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "print": print,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "hasattr": hasattr,
        "open": safe_open,
        "__import__": safe_import,
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "IndexError": IndexError,
        "KeyError": KeyError,
        "ZeroDivisionError": ZeroDivisionError,
        "AssertionError": AssertionError,
        "AttributeError": AttributeError,
        "RuntimeError": RuntimeError,
        "StopIteration": StopIteration,
    }

    safe_globals = {
        "__builtins__": safe_builtins,
        **allowed_modules,
    }

    # 4. Redirect stdout
    stdout_buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_buf

    try:
        tree = ast.parse(code)
        last_expr_val = None

        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = tree.body.pop()
            if tree.body:
                exec(compile(tree, "<sandbox>", "exec"), safe_globals)
            expr_code = compile(ast.Expression(last_expr.value), "<sandbox>", "eval")
            last_expr_val = eval(expr_code, safe_globals)
        else:
            exec(compile(tree, "<sandbox>", "exec"), safe_globals)

        out_str = stdout_buf.getvalue()
        sys.stdout = old_stdout

        # Normalize last_expr_val if not directly JSON serializable
        try:
            json.dumps(last_expr_val)
            normalized_result = last_expr_val
        except (TypeError, OverflowError):
            normalized_result = str(last_expr_val)

        sys.stdout.write(
            json.dumps(
                {
                    "stdout": out_str,
                    "result": normalized_result if normalized_result is not None else out_str.strip(),
                    "success": True,
                    "error": None,
                }
            )
        )
        sys.stdout.flush()

    except MemoryError:
        out_str = stdout_buf.getvalue()
        sys.stdout = old_stdout
        sys.stdout.write(
            json.dumps(
                {
                    "stdout": out_str,
                    "result": None,
                    "success": False,
                    "error": "MemoryError: Execution exceeded allocated memory limit",
                    "traceback": traceback.format_exc(),
                }
            )
        )
        sys.stdout.flush()
    except Exception as e:
        out_str = stdout_buf.getvalue()
        sys.stdout = old_stdout
        sys.stdout.write(
            json.dumps(
                {
                    "stdout": out_str,
                    "result": None,
                    "success": False,
                    "error": f"{type(e).__name__}: {str(e)}",
                    "traceback": traceback.format_exc(),
                }
            )
        )
        sys.stdout.flush()


if __name__ == "__main__":
    main()
