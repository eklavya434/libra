"""
Tests for Master Capstone System Audit Engine (Phase 36)
"""

import os

from packages.evaluation.capstone_audit import CapstoneAuditReport, LibraCapstoneAudit


def test_capstone_audit_engine():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    audit = LibraCapstoneAudit(repo_root)

    # Test individual pillars
    p1 = audit.audit_pillar_1_modeling()
    assert p1.passed is True
    assert len(p1.details) >= 3

    p2 = audit.audit_pillar_2_inference()
    assert p2.passed is True

    p3 = audit.audit_pillar_3_providers()
    assert p3.passed is True

    p6 = audit.audit_pillar_6_agents()
    assert p6.passed is True

    # Test full audit report
    report = audit.run_full_capstone_audit()
    assert isinstance(report, CapstoneAuditReport)
    assert report.total_pillars == 7
    assert report.pillars_passed == 7
    assert report.all_passed is True
    assert report.duration_sec > 0.0
