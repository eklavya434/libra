"""
Tests for CI Workflow and Pre-Commit Configuration (Phase 35)
"""

import os

import yaml

from scripts.pre_commit_check import check_staged_files, check_storage_quota


def test_ci_workflow_yaml_structure():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ci_path = os.path.join(repo_root, ".github", "workflows", "ci.yml")

    assert os.path.isfile(ci_path), "ci.yml must exist"

    with open(ci_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "jobs" in data
    jobs = data["jobs"]
    required_jobs = [
        "lint",
        "backend-test",
        "frontend-build",
        "deployment-audit",
        "regression-gate",
    ]
    for j in required_jobs:
        assert j in jobs, f"Job '{j}' missing from CI workflow"

    # Verify CPU PyTorch wheel is used in backend-test
    backend_steps = jobs["backend-test"]["steps"]
    step_commands = " ".join(s.get("run", "") for s in backend_steps)
    assert "download.pytorch.org/whl/cpu" in step_commands


def test_pre_commit_storage_check():
    # Current repository should be well within 15 GB quota
    violations = check_storage_quota()
    assert len(violations) == 0


def test_pre_commit_staged_check():
    # Staged check should execute without crashing
    violations = check_staged_files()
    assert isinstance(violations, list)
