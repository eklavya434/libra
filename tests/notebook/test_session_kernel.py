"""
Unit tests for Libra NotebookKernel, variable persistence, and AST security.
"""

import pytest

from packages.core.notebook.security import NotebookSecurityError, NotebookSecurityPolicy
from packages.core.notebook.session_kernel import NotebookKernel, NotebookSessionManager


def test_kernel_basic_execution():
    kernel = NotebookKernel(session_id="test_basic")
    out = kernel.execute("x = 10\ny = 20\nx + y")

    assert out.status == "ok"
    assert out.execution_count == 1
    assert out.result == "30"
    assert out.duration_ms > 0
    assert any(v["name"] == "x" and v["value_repr"] == "10" for v in out.variables)
    assert any(v["name"] == "y" and v["value_repr"] == "20" for v in out.variables)


def test_kernel_state_persistence_across_cells():
    kernel = NotebookKernel(session_id="test_persist")

    # Cell 1: define variables and function
    c1 = kernel.execute("multiplier = 5\ndef compute(v):\n    return v * multiplier")
    assert c1.status == "ok"
    assert c1.execution_count == 1

    # Cell 2: use variables and function defined in Cell 1
    c2 = kernel.execute("res = compute(12)\nres")
    assert c2.status == "ok"
    assert c2.execution_count == 2
    assert c2.result == "60"

    # Cell 3: inspect variables
    vars_list = kernel.get_variables()
    var_dict = {v["name"]: v["value_repr"] for v in vars_list}
    assert var_dict.get("multiplier") == "5"
    assert var_dict.get("res") == "60"


def test_kernel_stdout_stderr_capture():
    kernel = NotebookKernel(session_id="test_io")
    out = kernel.execute(
        "import sys\nprint('standard output line')\nprint('standard error line', file=sys.stderr)"
    )

    assert out.status == "ok"
    assert "standard output line" in out.stdout
    assert "standard error line" in out.stderr


def test_kernel_reset():
    kernel = NotebookKernel(session_id="test_reset")
    kernel.execute("custom_var = 999")
    assert any(v["name"] == "custom_var" for v in kernel.get_variables())

    kernel.reset()
    assert kernel.execution_count == 0
    assert not any(v["name"] == "custom_var" for v in kernel.get_variables())


def test_security_policy_rejection():
    # Forbidden os import
    with pytest.raises(NotebookSecurityError):
        NotebookSecurityPolicy.validate("import os\nos.system('whoami')")

    # Forbidden subprocess import
    with pytest.raises(NotebookSecurityError):
        NotebookSecurityPolicy.validate("import subprocess\nsubprocess.run(['dir'])")

    # Forbidden eval call
    with pytest.raises(NotebookSecurityError):
        NotebookSecurityPolicy.validate("x = eval('2 + 2')")

    # Forbidden reflection attribute
    with pytest.raises(NotebookSecurityError):
        NotebookSecurityPolicy.validate("cls = ().__class__.__bases__[0].__subclasses__()")


def test_kernel_executes_security_error_gracefully():
    kernel = NotebookKernel(session_id="test_sec_exec")
    out = kernel.execute("import socket\ns = socket.socket()")

    assert out.status == "error"
    assert "Security policy violation" in out.error_message
    assert "socket" in out.stderr


def test_session_manager():
    manager = NotebookSessionManager()
    k1 = manager.get_or_create("sess_1")
    assert k1.session_id == "sess_1"

    k1.execute("score = 100")
    sessions = manager.list_sessions()
    assert any(s["session_id"] == "sess_1" for s in sessions)

    presets = manager.get_presets()
    assert len(presets) >= 3
    assert any("sales" in p["id"] for p in presets)

    deleted = manager.delete("sess_1")
    assert deleted is True
    assert manager.get("sess_1") is None
