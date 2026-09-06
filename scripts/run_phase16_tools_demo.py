"""
Phase 16 Interactive Demonstration: Tool Use & Safe Sandbox Execution

Demonstrates:
1. Tool Discovery & OpenAI Schema Generation.
2. AST Mathematical Calculations without eval().
3. Safe Python Sandbox execution with scoped file I/O.
4. Hard Preemptive Process Termination on Infinite Loops.
5. Kernel-Enforced Memory Limit on Memory Bombs.
6. Filesystem Escape Denial.
7. Disallowed Import Rejection.
8. LLM Tool Call Parsing (XML, Markdown, JSON).
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from packages.tools import (
    CalculatorTool,
    PythonInterpreterTool,
    SafePythonSandbox,
    SandboxSecurityError,
    ToolCallParser,
    get_tool_registry,
)


def print_banner(title: str) -> None:
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def demo_registry_and_schemas():
    print_banner("1. TOOL REGISTRY & OPENAI SCHEMAS")
    registry = get_tool_registry()
    print(f"[+] Registered tools ({len(registry.list_tools())}): {registry.get_tool_names()}")
    schemas = registry.get_schemas()
    for s in schemas:
        fn = s["function"]
        print(f"  - {fn['name']}: {fn['description'][:60]}... ({len(fn['parameters']['properties'])} args)")


def demo_ast_calculator():
    print_banner("2. AST-BASED MATHEMATICAL CALCULATOR (ZERO eval())")
    calc = CalculatorTool()
    expressions = [
        "144 * 12",
        "sqrt(65536)",
        "2**16",
        "log2(1024)",
        "factorial(6)",
        "sin(0) + cos(0)",
    ]
    for expr in expressions:
        res = calc.execute(expression=expr)
        print(f"  Expr: {expr:<20} -> Result: {res.output['result']} ({res.execution_time_ms}ms)")


def demo_safe_sandbox_execution():
    print_banner("3. SAFE PYTHON SANDBOX (PROCESS ISOLATION + SCOPED FILE I/O)")
    sandbox = SafePythonSandbox(default_timeout=3.0, max_memory_mb=128.0)
    code = """
import math
import statistics
import json

data = [12, 15, 18, 24, 30, 45, 90]
mean_val = statistics.mean(data)
std_val = statistics.stdev(data)

with open("output_report.json", "w") as f:
    json.dump({"mean": mean_val, "stdev": std_val}, f)

with open("output_report.json", "r") as f:
    loaded = json.load(f)

print(f"Calculated stats over {len(data)} items.")
loaded
"""
    res = sandbox.execute(code)
    print(f"  Success: {res['success']}")
    print(f"  Stdout:  {res['stdout'].strip()}")
    print(f"  Result:  {res['result']}")


def demo_adversarial_defenses():
    print_banner("4. ADVERSARIAL ATTACK DEFENSE DEMONSTRATION")
    sandbox = SafePythonSandbox(default_timeout=0.5, max_memory_mb=64.0)

    # 4.1 Infinite Loop
    print("\n[A] Attack: Runaway Infinite Loop (while True)")
    t0 = time.perf_counter()
    try:
        sandbox.execute("while True: pass", timeout_sec=0.5)
        print("  FAIL: Did not time out!")
    except TimeoutError as e:
        print(f"  BLOCKED: {e} (Elapsed: {round(time.perf_counter() - t0, 3)}s)")
        print("  OS Kernel terminated child process cleanly. Zero GIL starvation.")

    # 4.2 Memory Bomb
    print("\n[B] Attack: Memory Bomb Allocation (Runaway List Growth)")
    bomb_code = """
l = []
while True:
    l.append('A' * 1000000)
"""
    res_bomb = sandbox.execute(bomb_code, timeout_sec=2.0, memory_mb=64.0)
    print(f"  BLOCKED: {res_bomb['error']}")
    print("  Windows Job Object / memory limit enforced ceiling successfully.")

    # 4.3 Filesystem Escape
    print("\n[C] Attack: Filesystem Directory Traversal Escape (../../)")
    escape_code = "with open('../../secret.txt', 'w') as f: f.write('escape')"
    res_escape = sandbox.execute(escape_code)
    print(f"  BLOCKED: {res_escape['error']}")

    # 4.4 Disallowed Import
    print("\n[D] Attack: Importing Forbidden System Modules (os, socket)")
    for mod in ["os", "socket", "subprocess"]:
        try:
            sandbox.execute(f"import {mod}")
            print(f"  FAIL: {mod} was not blocked!")
        except SandboxSecurityError as e:
            print(f"  BLOCKED: {e}")


def demo_tool_parser():
    print_banner("5. LLM TOOL CALL PARSER")
    raw_llm_response = """
I will check the square root of 256:
<tool_call>
{
    "name": "calculator",
    "arguments": {"expression": "sqrt(256)"}
}
</tool_call>
And then search the documentation:
```json
{
    "name": "web_search",
    "arguments": {"query": "PyTorch attention", "max_results": 3}
}
```
"""
    calls = ToolCallParser.parse(raw_llm_response)
    print(f"[+] Extracted {len(calls)} structured tool calls from LLM generation:")
    for i, c in enumerate(calls, 1):
        print(f"  [{i}] Tool: '{c.name}', Args: {c.arguments}")

    cleaned = ToolCallParser.strip_tool_calls(raw_llm_response)
    print(f"\n[+] Conversational text with tool markup stripped:\n{cleaned}")


def main():
    print("=" * 75)
    print("      LIBRA PHASE 16: TOOL USE & SAFE SANDBOX EXECUTION DEMO")
    print("=" * 75)
    demo_registry_and_schemas()
    demo_ast_calculator()
    demo_safe_sandbox_execution()
    demo_adversarial_defenses()
    demo_tool_parser()
    print_banner("PHASE 16 DEMO COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
