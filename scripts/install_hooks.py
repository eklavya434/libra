"""
Installs the Project Libra git pre-commit hook by configuring core.hooksPath.
"""

from __future__ import annotations

import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def install() -> None:
    try:
        subprocess.run(
            ["git", "config", "core.hooksPath", ".githooks"],
            cwd=REPO_ROOT,
            check=True,
        )
        print("Git hooks configured successfully: core.hooksPath -> .githooks")
    except Exception as e:
        print(f"Error configuring git hooks: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    install()
