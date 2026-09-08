"""
Tests for Container & Deployment Configuration (Phase 34)
"""

import os
from packages.core.deployment_validator import DeploymentValidator


def test_deployment_validator_full_audit():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    validator = DeploymentValidator(repo_root)
    audit = validator.run_full_audit()

    assert audit["all_valid"] is True
    assert audit["compose"]["service_count"] >= 2
    assert audit["dockerignore"]["is_valid"] is True


def test_docker_compose_structure():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    validator = DeploymentValidator(repo_root)
    rep = validator.validate_compose()

    assert rep.is_valid is True
    assert rep.service_count >= 2
    assert any("backend" in c for c in rep.checks_passed)
    assert any("frontend" in c for c in rep.checks_passed)
    assert any("8000" in c for c in rep.checks_passed)
    assert any("3000" in c for c in rep.checks_passed)


def test_backend_dockerfile_security_and_stages():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    validator = DeploymentValidator(repo_root)
    res = validator.validate_dockerfile(os.path.join("infra", "docker", "Dockerfile.backend"))

    assert res["exists"] is True
    assert res["is_multi_stage"] is True
    assert res["has_non_root_user"] is True
    assert res["has_healthcheck"] is True
    assert "8000" in res["exposed_ports"]


def test_frontend_dockerfile_security_and_stages():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    validator = DeploymentValidator(repo_root)
    res = validator.validate_dockerfile(os.path.join("infra", "docker", "Dockerfile.frontend"))

    assert res["exists"] is True
    assert res["is_multi_stage"] is True
    assert res["has_non_root_user"] is True
    assert res["has_healthcheck"] is True
    assert "3000" in res["exposed_ports"]


def test_dockerignore_security():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    validator = DeploymentValidator(repo_root)
    res = validator.validate_dockerignore()

    assert res["exists"] is True
    assert res["is_valid"] is True
    assert ".venv" in res["critical_excluded"]
    assert "node_modules" in res["critical_excluded"]
    assert ".env" in res["critical_excluded"]
    assert ".git" in res["critical_excluded"]


def test_next_config_standalone():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    next_config_path = os.path.join(repo_root, "apps", "frontend", "next.config.mjs")
    assert os.path.isfile(next_config_path)

    with open(next_config_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "standalone" in content