"""Verification evidence must be real: every test cited by a report must exist.

Prime Directive #8 forbids fabricating verification. The reports in
``docs/reports/`` cite their evidence as ``tests/<path>.py::test_<name>`` inline
code. A citation that points at a deleted, renamed, or never-written test is a
false claim, so this test fails the build if any report citation dangles.
"""

from __future__ import annotations

import os
import re

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORTS_DIR = os.path.join(REPO_ROOT, "docs", "reports")

CITATION = re.compile(r"(tests/[A-Za-z0-9_/]+\.py)(?:::([A-Za-z0-9_]+))?")
TEST_DEF = re.compile(r"def\s+([A-Za-z0-9_]+)\b")


def _report_files() -> list[str]:
    if not os.path.isdir(REPORTS_DIR):
        return []
    return sorted(
        os.path.join(REPORTS_DIR, name) for name in os.listdir(REPORTS_DIR) if name.endswith(".md")
    )


REPORTS = _report_files()


def test_reports_directory_is_present():
    assert REPORTS, f"expected markdown reports in {REPORTS_DIR}"


@pytest.mark.skipif(not REPORTS, reason="no report files to audit")
def test_every_cited_test_file_exists():
    missing: list[str] = []
    for report in REPORTS:
        with open(report, "r", encoding="utf-8") as handle:
            text = handle.read()
        for rel_path, _ in CITATION.findall(text):
            if not os.path.isfile(os.path.join(REPO_ROOT, rel_path)):
                missing.append(f"{os.path.basename(report)} -> {rel_path}")
    assert not missing, f"reports cite non-existent test files: {sorted(set(missing))}"


@pytest.mark.skipif(not REPORTS, reason="no report files to audit")
def test_every_cited_test_function_exists():
    missing: list[str] = []
    for report in REPORTS:
        with open(report, "r", encoding="utf-8") as handle:
            text = handle.read()
        for rel_path, func_name in CITATION.findall(text):
            if not func_name:
                continue
            target = os.path.join(REPO_ROOT, rel_path)
            if not os.path.isfile(target):
                continue
            with open(target, "r", encoding="utf-8") as handle:
                source = handle.read()
            if func_name not in set(TEST_DEF.findall(source)):
                missing.append(f"{os.path.basename(report)} -> {rel_path}::{func_name}")
    assert not missing, f"reports cite non-existent test functions: {sorted(set(missing))}"


@pytest.mark.skipif(not REPORTS, reason="no report files to audit")
def test_reports_do_not_claim_fabricated_results():
    """Guard against inventing live smoke results for services we never reached."""
    forbidden = ("TODO: fill", "TBD result", "placeholder result", "XXX result")
    offenders: list[str] = []
    for report in REPORTS:
        with open(report, "r", encoding="utf-8") as handle:
            lowered = handle.read().lower()
        for phrase in forbidden:
            if phrase in lowered:
                offenders.append(f"{os.path.basename(report)}: {phrase}")
    assert not offenders, f"unfinished result placeholders in reports: {offenders}"
