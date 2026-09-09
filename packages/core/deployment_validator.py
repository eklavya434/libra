"""
Libra Core - Deployment & Container Configuration Validator (Phase 34)
Provides automated programmatic validation for:
1. Docker Compose schema, services, ports, healthchecks, networks, and volumes
2. Dockerfile multi-stage builds, non-root security users, and healthcheck commands
3. .dockerignore leak-prevention and context size optimization
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class ValidationReport:
    is_valid: bool
    service_count: int
    checks_passed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "service_count": self.service_count,
            "checks_passed": self.checks_passed,
            "warnings": self.warnings,
            "errors": self.errors,
        }


class DeploymentValidator:
    """Validates container orchestration configurations for Project Libra."""

    def __init__(self, repo_root: str) -> None:
        self.repo_root = os.path.abspath(repo_root)

    def validate_compose(self, compose_filename: str = "docker-compose.yml") -> ValidationReport:
        compose_path = os.path.join(self.repo_root, compose_filename)
        passed: list[str] = []
        errors: list[str] = []
        warnings: list[str] = []

        if not os.path.isfile(compose_path):
            return ValidationReport(
                is_valid=False,
                service_count=0,
                errors=[f"File not found: {compose_path}"],
            )

        try:
            with open(compose_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            return ValidationReport(
                is_valid=False,
                service_count=0,
                errors=[f"YAML parsing failed: {e}"],
            )

        if not isinstance(data, dict) or "services" not in data:
            return ValidationReport(
                is_valid=False,
                service_count=0,
                errors=["Invalid docker-compose: missing 'services' root key"],
            )

        services = data.get("services", {})
        passed.append(f"Parsed valid docker-compose with {len(services)} declared services")

        # 1. Check essential services
        for req_srv in ["backend", "frontend"]:
            if req_srv in services:
                passed.append(f"Required service '{req_srv}' found")
            else:
                errors.append(f"Missing required service: '{req_srv}'")

        # 2. Check backend configuration
        backend = services.get("backend", {})
        if backend:
            ports = backend.get("ports", [])
            if any("8000:8000" in str(p) or "8000" in str(p) for p in ports):
                passed.append("Backend exposes port 8000")
            else:
                warnings.append("Backend does not explicitly map port 8000")

            if "healthcheck" in backend:
                passed.append("Backend has healthcheck defined")
            else:
                warnings.append("Backend lacks healthcheck definition")

            volumes = backend.get("volumes", [])
            if any("models" in str(v) for v in volumes):
                passed.append("Backend mounts external models volume (quota compliance)")
            else:
                warnings.append("Backend does not mount external models volume")

        # 3. Check frontend configuration
        frontend = services.get("frontend", {})
        if frontend:
            ports = frontend.get("ports", [])
            if any("3000:3000" in str(p) or "3000" in str(p) for p in ports):
                passed.append("Frontend exposes port 3000")
            else:
                warnings.append("Frontend does not explicitly map port 3000")

            depends_on = frontend.get("depends_on", {})
            if "backend" in depends_on:
                passed.append("Frontend depends on backend service")
            else:
                warnings.append("Frontend lacks explicit dependency on backend service")

        # 4. Check networks
        if "networks" in data:
            passed.append(f"Bridge network defined: {list(data['networks'].keys())}")

        is_valid = len(errors) == 0
        return ValidationReport(
            is_valid=is_valid,
            service_count=len(services),
            checks_passed=passed,
            warnings=warnings,
            errors=errors,
        )

    def validate_dockerfile(self, rel_path: str) -> dict[str, Any]:
        """Validates security, multi-stage architecture, and healthcheck of a Dockerfile."""
        full_path = os.path.join(self.repo_root, rel_path)
        if not os.path.isfile(full_path):
            return {
                "path": rel_path,
                "exists": False,
                "is_valid": False,
                "errors": ["File not found"],
            }

        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        from_lines = [l.strip() for l in lines if l.strip().upper().startswith("FROM ")]
        user_lines = [l.strip() for l in lines if l.strip().upper().startswith("USER ")]
        health_lines = [l.strip() for l in lines if l.strip().upper().startswith("HEALTHCHECK ")]
        expose_lines = [l.strip() for l in lines if l.strip().upper().startswith("EXPOSE ")]

        is_multi_stage = len(from_lines) > 1
        has_non_root_user = len(user_lines) > 0
        has_healthcheck = len(health_lines) > 0

        checks = []
        if is_multi_stage:
            checks.append(f"Multi-stage build verified ({len(from_lines)} stages)")
        else:
            checks.append("Single-stage build (consider multi-stage for layer minimization)")

        if has_non_root_user:
            checks.append(f"Security: Non-root user specified ({user_lines[-1]})")
        else:
            checks.append("Security warning: Runs as root (no USER directive)")

        if has_healthcheck:
            checks.append("Healthcheck directive found")

        return {
            "path": rel_path,
            "exists": True,
            "is_valid": True,
            "stages_count": len(from_lines),
            "is_multi_stage": is_multi_stage,
            "has_non_root_user": has_non_root_user,
            "has_healthcheck": has_healthcheck,
            "exposed_ports": [l.split()[-1] for l in expose_lines],
            "checks": checks,
        }

    def validate_dockerignore(self, filename: str = ".dockerignore") -> dict[str, Any]:
        """Validates that large directories and secrets are excluded from build contexts."""
        full_path = os.path.join(self.repo_root, filename)
        if not os.path.isfile(full_path):
            return {"exists": False, "is_valid": False, "errors": [".dockerignore not found"]}

        with open(full_path, "r", encoding="utf-8") as f:
            patterns = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        critical_patterns = [".venv", "node_modules", ".git", ".env"]
        missing = [cp for cp in critical_patterns if not any(cp in p for p in patterns)]

        return {
            "exists": True,
            "is_valid": len(missing) == 0,
            "total_patterns": len(patterns),
            "critical_excluded": [cp for cp in critical_patterns if cp not in missing],
            "missing_critical": missing,
        }

    def run_full_audit(self) -> dict[str, Any]:
        compose_rep = self.validate_compose()
        backend_df = self.validate_dockerfile(os.path.join("infra", "docker", "Dockerfile.backend"))
        frontend_df = self.validate_dockerfile(
            os.path.join("infra", "docker", "Dockerfile.frontend")
        )
        dockerignore_res = self.validate_dockerignore()

        all_valid = (
            compose_rep.is_valid
            and backend_df.get("is_valid", False)
            and frontend_df.get("is_valid", False)
            and dockerignore_res.get("is_valid", False)
        )

        return {
            "all_valid": all_valid,
            "compose": compose_rep.to_dict(),
            "backend_dockerfile": backend_df,
            "frontend_dockerfile": frontend_df,
            "dockerignore": dockerignore_res,
        }
