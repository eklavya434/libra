"""
Libra Tools Package - Secure Process-Isolated Python Execution Sandbox

Implements multi-layer defense-in-depth:
1. Static Abstract Syntax Tree (AST) Validation & Strict Import Allowlisting.
2. Out-of-Process OS Isolation (Child Process via `subprocess.Popen`).
3. Kernel-Enforced Memory Limits (Windows Job Objects / POSIX rlimit).
4. Hard Preemptive Process Termination on Timeout (TerminateProcess / SIGKILL).
5. Scoped Ephemeral Filesystem Isolation (Chrooted path validation).
6. Kernel Subprocess Blocking (ActiveProcessLimit = 1).
7. Network Isolation (Blackhole proxy environment & socket prohibitions).
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Optional

RUNNER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "runner.py"))

# Strict Module Allowlist (Only pure standard calculation & parsing utilities)
ALLOWED_MODULES = {
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
}

FORBIDDEN_CALLS = {
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

FORBIDDEN_ATTRS = {
    "__subclasses__",
    "__globals__",
    "__code__",
    "__closure__",
    "__bases__",
    "__class__",
    "__mro__",
    "__dict__",
    "__builtins__",
}


class SandboxSecurityError(PermissionError):
    """Raised when code violates static AST security analysis."""

    pass


class SecurityVisitor(ast.NodeVisitor):
    """
    Statically analyzes the Python AST to reject unauthorized imports,
    dynamic code evaluation, and reflection-based sandbox breakouts.
    """

    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base_mod = alias.name.split(".")[0]
            if base_mod not in ALLOWED_MODULES:
                self.violations.append(f"Forbidden import: '{alias.name}' (not in allowlist)")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            base_mod = node.module.split(".")[0]
            if base_mod not in ALLOWED_MODULES:
                self.violations.append(f"Forbidden from-import: '{node.module}' (not in allowlist)")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                self.violations.append(f"Forbidden function call: '{node.func.id}()'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_ATTRS:
            self.violations.append(f"Forbidden attribute access: '{node.attr}'")
        self.generic_visit(node)


class SafePythonSandbox:
    """
    High-security, process-isolated execution environment for user- or agent-generated Python code.
    Enforces time, memory, filesystem, subprocess, and network boundaries.
    """

    def __init__(
        self,
        default_timeout: float = 5.0,
        max_memory_mb: float = 128.0,
    ) -> None:
        self.default_timeout = default_timeout
        self.max_memory_mb = max_memory_mb

    def validate_code(self, code: str) -> None:
        """Parses AST and validates against the strict security policy."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise SyntaxError(f"Syntax error in script: {e}")

        visitor = SecurityVisitor()
        visitor.visit(tree)

        if visitor.violations:
            raise SandboxSecurityError(
                f"Security policy violation: {'; '.join(visitor.violations)}"
            )

    def execute(
        self,
        code: str,
        timeout_sec: Optional[float] = None,
        memory_mb: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Executes Python code in a dedicated OS child process with hard resource limits.
        """
        # 1. Static AST Analysis
        self.validate_code(code)

        timeout = timeout_sec or self.default_timeout
        mem_limit = memory_mb or self.max_memory_mb

        # 2. Ephemeral Scoped Working Directory
        with tempfile.TemporaryDirectory(prefix="libra_sandbox_") as temp_dir:
            payload = json.dumps(
                {
                    "code": code,
                    "sandbox_dir": temp_dir,
                }
            )

            # 3. Prepare blackholed network environment
            child_env = {
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
                "PATH": os.environ.get("PATH", ""),
                "PYTHONIOENCODING": "utf-8",
                "HTTP_PROXY": "http://127.0.0.1:0",
                "HTTPS_PROXY": "http://127.0.0.1:0",
                "ALL_PROXY": "http://127.0.0.1:0",
                "NO_PROXY": "",
            }

            # 4. Spawn child process executing runner.py in isolated mode (-I: no user site, no PYTHONPATH)
            python_bin = getattr(sys, "_base_executable", sys.executable)
            preexec = None
            if sys.platform != "win32":
                try:
                    import resource

                    def _set_posix_limits(limit_mb=mem_limit):
                        bytes_limit = int(limit_mb * 1024 * 1024)
                        resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))

                    preexec = _set_posix_limits
                except (ImportError, AttributeError):
                    pass

            proc = subprocess.Popen(
                [python_bin, "-I", "-s", RUNNER_PATH],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=temp_dir,
                env=child_env,
                text=True,
                preexec_fn=preexec,
            )

            # 5. Apply OS Kernel Resource Limits (Memory & Subprocess Blocking)
            job_handle = self._apply_kernel_limits(proc, mem_limit)

            # 6. Execute with Hard Preemptive Timeout
            try:
                stdout_data, stderr_data = proc.communicate(input=payload, timeout=timeout)
            except subprocess.TimeoutExpired:
                # Preemptively terminate process
                proc.kill()
                proc.communicate()
                raise TimeoutError(
                    f"Execution timed out after {timeout:.1f}s (infinite loop guard)"
                )
            finally:
                self._cleanup_job_handle(job_handle)

            # 7. Decode Result
            if proc.returncode != 0 and not stdout_data.strip():
                # Check for Windows Job Object memory ceiling or crash exit codes
                # 0xC0000017 = STATUS_NO_MEMORY, 0xC0000005 = ACCESS_VIOLATION
                err_msg = (
                    stderr_data.strip() or f"Process terminated with exit code {proc.returncode}"
                )
                if proc.returncode in (3221225495, -1073741797) or "MemoryError" in err_msg:
                    err_msg = "MemoryError: Execution exceeded allocated memory limit"
                return {
                    "stdout": "",
                    "result": None,
                    "success": False,
                    "error": err_msg,
                    "traceback": None,
                }

            try:
                result_obj = json.loads(stdout_data.strip())
                result_obj.setdefault("traceback", None)
                return result_obj
            except json.JSONDecodeError:
                return {
                    "stdout": stdout_data,
                    "result": None,
                    "success": False,
                    "error": stderr_data.strip() or "Failed to decode child process output",
                    "traceback": None,
                }

    def _apply_kernel_limits(self, proc: subprocess.Popen, mem_limit_mb: float) -> Any:
        """Applies Windows Job Object limits (Memory limit and ActiveProcessLimit=1)."""
        if sys.platform != "win32":
            return None

        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_uint64),
                    ("WriteOperationCount", ctypes.c_uint64),
                    ("OtherOperationCount", ctypes.c_uint64),
                    ("ReadTransferCount", ctypes.c_uint64),
                    ("TransferCount", ctypes.c_uint64),
                    ("OtherTransferCount", ctypes.c_uint64),
                ]

            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryLimit", ctypes.c_size_t),
                    ("PeakJobMemoryLimit", ctypes.c_size_t),
                ]

            job = kernel32.CreateJobObjectW(None, None)
            if not job:
                return None

            JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
            JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
            JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

            limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            limits.BasicLimitInformation.LimitFlags = (
                JOB_OBJECT_LIMIT_PROCESS_MEMORY
                | JOB_OBJECT_LIMIT_JOB_MEMORY
                | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
                | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            )
            limits.BasicLimitInformation.ActiveProcessLimit = 1

            limit_bytes = int(mem_limit_mb * 1024 * 1024)
            limits.ProcessMemoryLimit = limit_bytes
            limits.JobMemoryLimit = limit_bytes

            JobObjectExtendedLimitInformation = 9
            kernel32.SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                ctypes.byref(limits),
                ctypes.sizeof(limits),
            )

            kernel32.AssignProcessToJobObject(job, int(proc._handle))
            return job
        except Exception:
            return None

    def _cleanup_job_handle(self, job_handle: Any) -> None:
        """Closes the Win32 Job Object handle."""
        if job_handle and sys.platform == "win32":
            try:
                import ctypes

                ctypes.windll.kernel32.CloseHandle(job_handle)
            except Exception:
                pass
