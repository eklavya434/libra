"""
Tests for 10-Pillar Grand Capstone Audit Engine (Phase 50)
"""

import os

from packages.evaluation.grand_capstone import (
    GrandCapstoneAudit,
    GrandCapstoneReport,
    GrandPillarAudit,
)


def test_grand_capstone_pillars():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    auditor = GrandCapstoneAudit(repo_root=repo_root)

    p1 = auditor.audit_pillar_1_foundations()
    assert isinstance(p1, GrandPillarAudit)
    assert p1.passed is True
    assert p1.pillar_id == 1

    p2 = auditor.audit_pillar_2_inference()
    assert p2.passed is True

    p3 = auditor.audit_pillar_3_rag_search()
    assert p3.passed is True

    p4 = auditor.audit_pillar_4_agents_tools()
    assert p4.passed is True

    p10 = auditor.audit_pillar_10_frontier_self_evolution()
    assert p10.passed is True
    assert p10.pillar_id == 10


def test_grand_capstone_full_audit():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    auditor = GrandCapstoneAudit(repo_root=repo_root)

    report = auditor.run_full_grand_audit()
    assert isinstance(report, GrandCapstoneReport)
    assert report.total_pillars == 10
    assert report.pillars_passed == 10
    assert report.all_passed is True
    assert report.completion_score_pct == 100.0
    assert "SUMMA CUM LAUDE" in report.graduation_honors
