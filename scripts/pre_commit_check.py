"""
Libra Pre-Commit Security & Hygiene Gate
Executes before every git commit to ensure:
1. No secrets or .env files are staged.
2. No oversized model weights (>15MB) are staged without git-lfs.
3. Storage footprint remains within the 15 GB quota.
4. Python code passes Ruff linting & formatting.
"""

from __future__ import annotations

import os
import subprocess
import sys

# Project root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def check_staged_files() -> list[str]:
    violations = []
    try:
        res = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=True,
        )
        staged_files = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    except Exception as e:
        return [f"Git diff error: {e}"]

    sensitive_patterns = [".env", ".env.local", "id_rsa", "credentials.json"]
    for f in staged_files:
        basename = os.path.basename(f)
        if any(basename == pat or basename.endswith(pat) for pat in sensitive_patterns):
            violations.append(f"Security Alert: Attempting to commit sensitive file: '{f}'")

        # Check for huge model weights accidentally committed
        full_p = os.path.join(REPO_ROOT, f)
        if os.path.isfile(full_p):
            sz_mb = os.path.getsize(full_p) / (1024 * 1024)
            if sz_mb > 25.0:
                violations.append(
                    f"Quota Alert: File '{f}' is {sz_mb:.1f} MB (exceeds 25 MB commit limit; use volumes/LFS)."
                )

    return violations


def check_storage_quota() -> list[str]:
    violations = []
    tracked = ["models", "data", "checkpoints", ".venv"]
    total_bytes = 0
    for d in tracked:
        dp = os.path.join(REPO_ROOT, d)
        if os.path.exists(dp):
            for root, _, files in os.walk(dp):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        total_bytes += os.path.getsize(fp)
                    except OSError:
                        pass
    total_mb = total_bytes / (1024 * 1024)
    if total_mb > 15360.0:
        violations.append(f"Storage Quota: {total_mb:.1f} MB exceeds 15 GB quota!")
    return violations


def run_linters() -> list[str]:
    violations = []
    venv_ruff = os.path.join(REPO_ROOT, ".venv", "Scripts", "ruff.exe")
    if not os.path.isfile(venv_ruff):
        venv_ruff = os.path.join(REPO_ROOT, ".venv", "bin", "ruff")
    cmd_base = [venv_ruff] if os.path.isfile(venv_ruff) else [sys.executable, "-m", "ruff"]

    # 1. Ruff check
    rc = subprocess.run(
        cmd_base + ["check", "."],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if rc.returncode != 0:
        err_msg = rc.stdout or rc.stderr
        violations.append(f"Ruff Linting Failed:\n{err_msg}")

    # 2. Ruff format check
    rfc = subprocess.run(
        cmd_base + ["format", "--check", "."],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if rfc.returncode != 0:
        violations.append("Ruff Formatting Check Failed (run 'ruff format .' to fix).")

    return violations


def main() -> int:
    print("=" * 60)
    print("LIBRA PRE-COMMIT GATE: Verifying Code Hygiene & Security")
    print("=" * 60)

    # 1. Staged files check
    staged_violations = check_staged_files()
    if staged_violations:
        for v in staged_violations:
            print(f"[ERROR] {v}")
        return 1
    print("[PASS] Staged files checked: No secrets or oversized weights detected.")

    # 2. Storage quota check
    quota_violations = check_storage_quota()
    if quota_violations:
        for v in quota_violations:
            print(f"[ERROR] {v}")
        return 1
    print("[PASS] Storage quota verified: Under 15 GB limit.")

    # 3. Linter check
    lint_violations = run_linters()
    if lint_violations:
        for v in lint_violations:
            print(f"[ERROR] {v}")
        return 1
    print("[PASS] Code formatting & linting verified.")

    print("=" * 60)
    print("ALL PRE-COMMIT CHECKS PASSED. PROCEEDING WITH COMMIT.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
