"""
Libra Notebook - Subprocess Jail Worker

Executes notebook cells inside an isolated OS process. This is the sandbox
backend behind ``LIBRA_CODE_SANDBOX=jail``.

Hardening applied here:
1. Real process boundary: user code runs in a separate process from the API,
   so a crash, hang, or malformed ``exec`` cannot corrupt the API process.
2. Wall-clock timeout enforcement is the parent's job (``packages/core/notebook/jail.py``).
3. POSIX resource limits (where ``resource`` exists): address-space cap,
   CPU-seconds cap, and a file-size cap so cells cannot exhaust memory, spin
   forever inside the worker, or write unbounded data to disk. On Windows the
   ``resource`` module is unavailable, so only the parent wall-clock kill
   applies (documented limitation).
4. Every cell still passes ``NotebookSecurityPolicy.validate`` (AST allow/deny
   list) before execution.

State (variables, imports, definitions) persists inside this single worker for
the life of one notebook session; ``reset`` wipes it.

Wire protocol: one JSON object per line on stdin, one JSON response line on
stdout. Actions: ``execute``, ``list_vars``, ``reset``, ``shutdown``.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

from packages.core.notebook.session_kernel import NotebookKernel

# Output caps so a single cell cannot flood the API with unbounded text.
MAX_STDOUT_CHARS = 65_536
MAX_STDERR_CHARS = 16_384
MAX_MIME_CHARS = 1_048_576

# POSIX resource limits (best-effort; skipped on platforms without `resource`).
JAIL_MEM_MB = int(os.environ.get("LIBRA_JAIL_MEM_MB", "2048"))
JAIL_FSIZE_BYTES = int(os.environ.get("LIBRA_JAIL_FSIZE_BYTES", str(1024 * 1024)))
JAIL_CPU_SEC = int(os.environ.get("LIBRA_JAIL_CPU_SEC", "30"))


def _apply_posix_limits(cpu_seconds: int) -> None:
    """Best-effort OS resource limits; silently skipped on non-POSIX hosts."""
    try:
        import resource  # POSIX only
    except ImportError:  # pragma: no cover - Windows execution path
        print(
            "jail: `resource` module unavailable on this platform; only the "
            "parent wall-clock timeout applies.",
            file=sys.stderr,
            flush=True,
        )
        return

    try:
        resource.setrlimit(resource.RLIMIT_AS, (JAIL_MEM_MB * 1024 * 1024,) * 2)
    except (ValueError, OSError):  # pragma: no cover - environment dependent
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    except (ValueError, OSError):  # pragma: no cover - environment dependent
        pass
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (JAIL_FSIZE_BYTES,) * 2)
    except (ValueError, OSError):  # pragma: no cover - environment dependent
        pass


def _truncate_output(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload["stdout"] = payload.get("stdout", "")[:MAX_STDOUT_CHARS]
    stderr = payload.get("stderr", "")
    if len(stderr) > MAX_STDERR_CHARS:
        tail = stderr[-MAX_STDERR_CHARS:]
        payload["stderr"] = f"[prev output truncated]\n{tail}"
    mime = payload.get("mime_outputs") or {}
    for mime_type, content in list(mime.items()):
        if len(content) > MAX_MIME_CHARS:
            mime[mime_type] = content[:MAX_MIME_CHARS] + "\n...[truncated]"
    payload["mime_outputs"] = mime
    return payload


def _handle_request(kernel: NotebookKernel, request: Dict[str, Any]) -> Dict[str, Any]:
    action = request.get("action")
    if action == "execute":
        code = request.get("code", "")
        output = kernel.execute(code)
        payload = output.to_dict()
        payload["action"] = "execute"
        return _truncate_output(payload)
    if action == "list_vars":
        return {"action": "list_vars", "status": "ok", "variables": kernel.get_variables()}
    if action == "reset":
        kernel.reset()
        return {"action": "reset", "status": "ok", "execution_count": kernel.execution_count}
    if action == "shutdown":
        return {"action": "shutdown", "status": "ok"}
    return {"action": action, "status": "error", "error_message": "Unknown action"}


def main() -> int:
    session_id = sys.argv[1] if len(sys.argv) > 1 else "jail-worker"

    # Confine write-prone operations to the jail work directory when provided.
    work_dir = os.environ.get("LIBRA_WORK_DIR")
    if work_dir:
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)

    kernel = NotebookKernel(session_id=session_id)
    _apply_posix_limits(cpu_seconds=JAIL_CPU_SEC)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = _handle_request(kernel, request)
        except (ValueError, TypeError) as parse_err:
            response = {
                "action": "execute",
                "status": "error",
                "error_message": f"Malformed worker request: {parse_err}",
            }
        try:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except BrokenPipeError:  # pragma: no cover - parent went away
            return 0
        if request.get("action") == "shutdown":
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
