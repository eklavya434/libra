"""
Libra Notebook - Parent-Side Jail Executor

Manages the isolated worker subprocess (see ``worker.py``) for one notebook
session. Responsibilities:

* Spawning/restarting the worker (one per session).
* Request/response JSON line protocol over stdin/stdout pipes.
* Hard wall-clock timeout per cell: the worker is killed if it does not answer
  in time (works on every OS; POSIX ``resource`` limits are an extra layer).
* Honest error surfacing when the worker dies, times out, or crashes.

The in-kernel namespace lives ONLY inside the worker process. The parent keeps
a lightweight mirror of ``execution_count`` and forwards variable inspection
to the worker.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from packages.core.notebook.session_kernel import ExecutionOutput


class NotebookJailError(RuntimeError):
    """Raised when the jail worker infrastructure itself fails."""

    pass


class NotebookJail:
    """Owns one worker subprocess and exposes a kernel-compatible execution API."""

    # Keep streaming stderr (discarded per line) in a bounded buffer for diagnostics.
    STDERRLINES_CAP = 60

    def __init__(
        self,
        session_id: str,
        timeout_sec: float = 8.0,
        work_root: Optional[Path] = None,
        restart_grace_sec: float = 20.0,
    ) -> None:
        self.session_id = session_id
        self.timeout_sec = timeout_sec
        self.restart_grace_sec = restart_grace_sec
        self._work_dir = (
            Path(work_root or Path(tempfile.gettempdir()) / "libra-jail-work") / session_id
        )
        self._proc: Optional[subprocess.Popen] = None
        self.execution_count = 0
        self._lock = threading.Lock()
        self._stderr_tail: list[str] = []
        self._stderr_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ lifecycle

    def _spawn(self) -> subprocess.Popen:
        env = dict(os.environ)
        env["LIBRA_WORK_DIR"] = str(self._work_dir)
        self._work_dir.mkdir(parents=True, exist_ok=True)
        rep_root = str(Path(__file__).resolve().parents[3])
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "packages.core.notebook.worker", self.session_id],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=rep_root,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._stderr_tail = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, args=(proc,), daemon=True)
        self._stderr_thread.start()
        return proc

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        """Continuously drain worker stderr so pipes never fill (deadlock)."""
        assert proc.stderr is not None
        for line in proc.stderr:
            self._stderr_tail.append(line.rstrip())
            if len(self._stderr_tail) > self.STDERRLINES_CAP:
                self._stderr_tail.pop(0)

    def _ensure_proc(self) -> subprocess.Popen:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._proc = self._spawn()
            return self._proc

    def _force_kill(self, proc: subprocess.Popen) -> None:
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - edge case
            pass

    # ------------------------------------------------------------------ protocol

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send one JSON request and read exactly one JSON response line.

        Enforces the wall-clock timeout by killing the worker if the response
        does not arrive in time. The worker process is single-threaded so the
        response we read always matches the request we sent.
        """
        proc = self._ensure_proc()
        assert proc.stdin is not None and proc.stdout is not None

        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            self._mark_dead()
            raise NotebookJailError(
                f"Worker for session '{self.session_id}' is not responding (pipe closed)."
            ) from exc

        timed_out = False

        def _kill_on_timeout() -> None:
            nonlocal timed_out
            timed_out = True
            self._force_kill(proc)

        killer = threading.Timer(self.timeout_sec, _kill_on_timeout)
        killer.daemon = True
        killer.start()
        try:
            line = proc.stdout.readline()
        finally:
            killer.cancel()

        if timed_out:
            self._mark_dead()
            raise NotebookJailError(
                f"Cell exceeded the {self.timeout_sec:.1f}s jail timeout for "
                f"session '{self.session_id}'. The worker process was terminated; "
                "state accumulated in this session is lost. Reset the session to continue."
            )
        if line == "":
            self._mark_dead()
            detail = self._summarize_stderr()
            raise NotebookJailError(
                f"Worker for session '{self.session_id}' exited unexpectedly.{detail}"
            )

        try:
            response = json.loads(line)
        except (ValueError, TypeError) as exc:
            self._mark_dead()
            raise NotebookJailError(
                f"Worker for session '{self.session_id}' returned a malformed response: {exc}"
            ) from exc

        # Record the worker-reported execution count after a successful execute.
        if response.get("action") == "execute":
            self.execution_count = int(response.get("execution_count", 0))
        return response

    def _mark_dead(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None:
            self._force_kill(proc)

    def _summarize_stderr(self) -> str:
        if not self._stderr_tail:
            return ""
        tail = " ; ".join(self._stderr_tail[-6:])
        return f" Last worker stderr lines: {tail[:300]}"

    # ------------------------------------------------------------------ kernel API

    def execute(self, code: str) -> ExecutionOutput:
        try:
            response = self._request({"action": "execute", "code": code})
        except NotebookJailError as jail_err:
            return ExecutionOutput(
                status="error",
                execution_count=self.execution_count,
                error_message=str(jail_err),
                stderr=str(jail_err),
                variables=[],
            )
        return self._execution_from_response(response)

    def list_vars(self) -> list[Dict[str, Any]]:
        try:
            response = self._request({"action": "list_vars"})
        except NotebookJailError:
            return []
        return response.get("variables") or []

    def reset(self) -> None:
        try:
            self._request({"action": "reset"})
        except NotebookJailError:
            pass
        finally:
            self.execution_count = 0

    def close(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write(json.dumps({"action": "shutdown"}) + "\n")
                    proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                pass
            self._force_kill(proc)
        shutil.rmtree(self._work_dir, ignore_errors=True)

    @staticmethod
    def _execution_from_response(response: Dict[str, Any]) -> ExecutionOutput:
        return ExecutionOutput(
            status=str(response.get("status", "error")),
            execution_count=int(response.get("execution_count", 0)),
            stdout=str(response.get("stdout", "")),
            stderr=str(response.get("stderr", "")),
            result=response.get("result"),
            mime_outputs={k: str(v) for k, v in (response.get("mime_outputs") or {}).items()},
            duration_ms=float(response.get("duration_ms", 0.0)),
            variables=response.get("variables") or [],
            error_message=response.get("error_message"),
        )
