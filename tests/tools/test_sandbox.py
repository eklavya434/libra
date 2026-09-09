"""
Adversarial Security & Isolation Test Suite for SafePythonSandbox (packages/tools/sandbox.py)

Directly tests:
1. Infinite loop execution & preemptive OS-level process termination.
2. Memory bomb allocations & kernel memory limit enforcement.
3. Filesystem escape attempts (relative traversal & absolute system paths) vs scoped I/O.
4. Disallowed imports (network, OS, system, introspection).
5. Subprocess spawning & dunder reflection breakout attempts.
"""

import sys
import time

import pytest

from packages.tools.sandbox import SafePythonSandbox, SandboxSecurityError


@pytest.fixture
def sandbox():
    return SafePythonSandbox(default_timeout=3.0, max_memory_mb=128.0)


# =============================================================================
# 1. ADVERSARIAL TEST: Infinite Loop & Hard Timeout Termination
# =============================================================================


def test_adversarial_infinite_loop(sandbox):
    """Verifies infinite loops are killed preemptively without thread abandonment or GIL stalling."""
    infinite_loop_code = """
count = 0
while True:
    count += 1
"""
    t0 = time.perf_counter()
    with pytest.raises(TimeoutError) as exc_info:
        sandbox.execute(infinite_loop_code, timeout_sec=0.5)

    duration = time.perf_counter() - t0
    assert "timed out after 0.5s" in str(exc_info.value)
    # Ensure termination happened promptly (within 1.5s total including process launch)
    assert duration < 2.0


# =============================================================================
# 2. ADVERSARIAL TEST: Memory Bomb Allocation Guard
# =============================================================================


def test_adversarial_memory_bomb(sandbox):
    """Verifies that attempts to consume excessive memory are caught and blocked."""
    memory_bomb_code = """
chunks = []
while True:
    chunks.append("M" * (1024 * 1024))
"""
    # Strict 64 MB memory limit
    try:
        res = sandbox.execute(memory_bomb_code, timeout_sec=3.0, memory_mb=64.0)
        assert res["success"] is False
        assert (
            "MemoryError" in res["error"]
            or "memory limit" in res["error"].lower()
            or "terminated" in res["error"].lower()
        )
    except TimeoutError:
        pass


# =============================================================================
# 3. ADVERSARIAL TEST: Filesystem Isolation & Escape Attempts
# =============================================================================


def test_adversarial_filesystem_escape_relative(sandbox):
    """Verifies that path traversal attempts (../..) outside the sandbox are denied."""
    escape_code = """
with open("../../escape_attempt.txt", "w") as f:
    f.write("hacked")
"""
    res = sandbox.execute(escape_code)
    assert res["success"] is False
    assert "PermissionError" in res["error"]
    assert "attempts to escape scoped sandbox directory" in res["error"]


def test_adversarial_filesystem_escape_absolute(sandbox):
    """Verifies that accessing absolute paths outside the scoped directory is denied."""
    target_path = "C:/Windows/win.ini" if sys.platform == "win32" else "/etc/passwd"
    escape_code = f"""
with open("{target_path}", "r") as f:
    content = f.read()
"""
    res = sandbox.execute(escape_code)
    assert res["success"] is False
    assert "PermissionError" in res["error"]
    assert "escape" in res["error"].lower() or "outside" in res["error"].lower()


def test_legitimate_scoped_filesystem_io(sandbox):
    """Verifies that safe file reading and writing within the scoped sandbox directory succeeds."""
    scoped_code = """
with open("workspace_file.txt", "w") as f:
    f.write("Libra isolated storage 2026")

with open("workspace_file.txt", "r") as f:
    recovered = f.read()

recovered
"""
    res = sandbox.execute(scoped_code)
    assert res["success"] is True
    assert res["result"] == "Libra isolated storage 2026"


# =============================================================================
# 4. ADVERSARIAL TEST: Disallowed & Dangerous Imports
# =============================================================================


@pytest.mark.parametrize(
    "disallowed_module",
    [
        "socket",
        "requests",
        "urllib",
        "http",
        "os",
        "sys",
        "subprocess",
        "shutil",
        "ctypes",
        "threading",
        "multiprocessing",
        "importlib",
    ],
)
def test_adversarial_disallowed_imports(sandbox, disallowed_module):
    """Verifies that non-allowlisted modules are rejected statically at AST parse time."""
    code = f"import {disallowed_module}\nprint('Should not run')"
    with pytest.raises(SandboxSecurityError) as exc_info:
        sandbox.execute(code)
    assert f"Forbidden import: '{disallowed_module}'" in str(exc_info.value)


def test_adversarial_disallowed_from_import(sandbox):
    """Verifies that from-imports of disallowed modules are rejected."""
    code = "from subprocess import Popen\nprint('Failed')"
    with pytest.raises(SandboxSecurityError) as exc_info:
        sandbox.execute(code)
    assert "Forbidden from-import: 'subprocess'" in str(exc_info.value)


# =============================================================================
# 5. ADVERSARIAL TEST: Subprocess & Reflection Breakout Attempts
# =============================================================================


def test_adversarial_reflection_breakout(sandbox):
    """Verifies that dunder reflection attacks are blocked."""
    breakout_code = "().__class__.__subclasses__()"
    with pytest.raises(SandboxSecurityError) as exc_info:
        sandbox.execute(breakout_code)
    assert "Forbidden attribute access: '__class__'" in str(exc_info.value)


def test_adversarial_forbidden_eval_call(sandbox):
    """Verifies that dynamic evaluation via eval/exec/compile/__import__ is blocked."""
    code = "eval('1 + 1')"
    with pytest.raises(SandboxSecurityError) as exc_info:
        sandbox.execute(code)
    assert "Forbidden function call: 'eval()'" in str(exc_info.value)


def test_adversarial_forbidden_dunder_import_call(sandbox):
    """Verifies that direct __import__ call is blocked."""
    code = "__import__('os')"
    with pytest.raises(SandboxSecurityError) as exc_info:
        sandbox.execute(code)
    assert "Forbidden function call: '__import__()'" in str(exc_info.value)


# =============================================================================
# 6. FUNCTIONAL TEST: Allowed Standard Libraries & Computations
# =============================================================================


def test_safe_calculations_and_stdlib(sandbox):
    """Verifies that safe mathematical, statistical, and formatting tools execute seamlessly."""
    code = """
import math
import statistics
import json
import collections

values = [10, 20, 30, 40, 50]
mean_val = statistics.mean(values)
sqrt_val = math.sqrt(mean_val)
counter = collections.Counter("project_libra")

payload = {
    "mean": mean_val,
    "sqrt": sqrt_val,
    "top_char": counter.most_common(1)[0][0]
}
print("Calculation complete!")
payload
"""
    res = sandbox.execute(code)
    assert res["success"] is True
    assert res["result"]["mean"] == 30.0
    assert round(res["result"]["sqrt"], 2) == 5.48
    assert "Calculation complete!" in res["stdout"]
