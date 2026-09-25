"""
Unit tests for the subprocess jail executor (packages/core/notebook/jail.py).

These spawn real worker subprocesses, so they are slightly slower than the
in-process kernel tests but verify genuine process-boundary behavior: state
persistence, security-policy enforcement, output truncation, wall-clock
timeout termination, and worker-crash surfacing.
"""

import contextlib

import pytest

from packages.core.notebook.jail import NotebookJail
from packages.core.notebook.session_kernel import NotebookKernel


@pytest.fixture
def jail_kernel_factory():
    created: list[NotebookJail] = []

    def _make(timeout_sec: float = 8.0) -> NotebookKernel:
        kernel = NotebookKernel(session_id=f"jail_test_{len(created)}")
        jail = NotebookJail(session_id=kernel.session_id, timeout_sec=timeout_sec)
        kernel.set_executor(jail)
        created.append(jail)
        return kernel

    yield _make

    for jail in created:
        with contextlib.suppress(Exception):
            jail.close()


def test_jail_basic_execution_and_state_persistence(jail_kernel_factory):
    kernel = jail_kernel_factory()

    out1 = kernel.execute("x = 40\nx")
    assert out1.status == "ok"
    assert out1.result == "40"
    assert out1.execution_count == 1

    out2 = kernel.execute("y = x + 2\ny")
    assert out2.status == "ok"
    assert out2.result == "42"
    assert out2.execution_count == 2

    var_map = {v["name"]: v["value_repr"] for v in kernel.get_variables()}
    assert var_map["x"] == "40"
    assert var_map["y"] == "42"


def test_jail_enforces_security_policy(jail_kernel_factory):
    kernel = jail_kernel_factory()
    out = kernel.execute("import subprocess\nsubprocess.call(['whoami'])")
    assert out.status == "error"
    assert "Security policy violation" in out.error_message


def test_jail_output_truncation(jail_kernel_factory):
    kernel = jail_kernel_factory()
    out = kernel.execute("print('A' * 200_000)")
    assert out.status == "ok"
    assert len(out.stdout) <= 65_536 + 64
    assert isinstance(kernel.get_variables(), list)


def test_jail_kills_hung_cell_and_reports_timeout(jail_kernel_factory):
    kernel = jail_kernel_factory(timeout_sec=0.6)
    out = kernel.execute("while True:\n    pass")
    assert out.status == "error"
    assert "jail timeout" in out.error_message
    # A fresh (non-hung) cell must work afterwards via a respawned worker.
    out2 = kernel.execute("2 + 2")
    assert out2.status == "ok"
    assert out2.result == "4"


def test_jail_reset_wipes_state(jail_kernel_factory):
    kernel = jail_kernel_factory()
    kernel.execute("secret = 1234")
    assert any(v["name"] == "secret" for v in kernel.get_variables())

    kernel.reset()
    assert kernel.execution_count == 0
    names = [v["name"] for v in kernel.get_variables()]
    assert "secret" not in names


def test_worker_crash_is_reported_and_session_recovers(jail_kernel_factory):
    kernel = jail_kernel_factory(timeout_sec=2.0)
    out = kernel.execute("raise SystemExit('boom')")
    # The worker dies loudly; the API must surface a clear error, not a hang.
    assert out.status == "error"

    out2 = kernel.execute("1 + 1")
    assert out2.status == "ok"
    assert out2.result == "2"
