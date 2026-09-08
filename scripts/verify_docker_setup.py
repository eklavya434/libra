"""
Phase 34 CLI Verification:
Container & Public Deployment Infrastructure Verification
Demonstrates:
1. Dockerfile security inspection (non-root UID, multi-stage architecture, healthchecks)
2. Docker Compose configuration & network validation
3. Leak prevention & .dockerignore coverage
4. Next.js standalone build presence
5. Zero-cost deployment readiness check
"""

import sys
import os

# Ensure project root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from packages.core.deployment_validator import DeploymentValidator


def main():
    print("=" * 80)
    print("LIBRA PHASE 34: CONTAINER & DEPLOYMENT READINESS AUDIT")
    print("=" * 80)

    validator = DeploymentValidator(REPO_ROOT)
    audit = validator.run_full_audit()

    # 1. Docker Compose Analysis
    print("\n[1] Docker Compose Orchestration (docker-compose.yml):")
    compose_rep = audit["compose"]
    print(f"  Valid Schema: {compose_rep['is_valid']}")
    print(f"  Declared Services ({compose_rep['service_count']}):")
    for chk in compose_rep["checks_passed"]:
        print(f"    [+] {chk}")
    for warn in compose_rep["warnings"]:
        print(f"    [!] {warn}")

    # 2. Backend Multi-Stage Dockerfile Analysis
    print("\n[2] Backend Multi-Stage Dockerfile (infra/docker/Dockerfile.backend):")
    b_df = audit["backend_dockerfile"]
    print(f"  Exists: {b_df['exists']}")
    print(f"  Stages Count: {b_df['stages_count']} (Multi-stage: {b_df['is_multi_stage']})")
    print(f"  Non-Root Security User: {b_df['has_non_root_user']}")
    print(f"  Exposed Ports: {b_df['exposed_ports']}")
    print(f"  Healthcheck Configured: {b_df['has_healthcheck']}")
    for chk in b_df["checks"]:
        print(f"    [+] {chk}")

    # 3. Frontend Standalone Dockerfile Analysis
    print("\n[3] Frontend Standalone Dockerfile (infra/docker/Dockerfile.frontend):")
    f_df = audit["frontend_dockerfile"]
    print(f"  Exists: {f_df['exists']}")
    print(f"  Stages Count: {f_df['stages_count']} (Multi-stage: {f_df['is_multi_stage']})")
    print(f"  Non-Root Security User: {f_df['has_non_root_user']}")
    print(f"  Exposed Ports: {f_df['exposed_ports']}")
    print(f"  Healthcheck Configured: {f_df['has_healthcheck']}")
    for chk in f_df["checks"]:
        print(f"    [+] {chk}")

    # 4. Context Leak Prevention (.dockerignore)
    print("\n[4] Build Context Leak Prevention (.dockerignore):")
    d_ign = audit["dockerignore"]
    print(f"  Exists: {d_ign['exists']}")
    print(f"  Total Patterns: {d_ign['total_patterns']}")
    print(f"  Protected Critical Items: {', '.join(d_ign['critical_excluded'])}")
    if d_ign["missing_critical"]:
        print(f"  [!] Missing Critical Exclusions: {', '.join(d_ign['missing_critical'])}")
    else:
        print("  [+] All sensitive paths (.env, .git, node_modules, .venv) safely excluded.")

    # 5. Standalone Build Artifacts Check
    standalone_path = os.path.join(REPO_ROOT, "apps", "frontend", ".next", "standalone")
    standalone_exists = os.path.isdir(standalone_path)
    print("\n[5] Next.js Standalone Build Check:")
    print(f"  Standalone Folder: {standalone_path}")
    print(f"  Compiled Successfully: {standalone_exists}")
    if standalone_exists:
        server_js = os.path.join(standalone_path, "server.js")
        print(f"  server.js Present: {os.path.isfile(server_js)} (Production entrypoint ready)")

    print("\n" + "=" * 80)
    if audit["all_valid"]:
        print("STATUS: ALL CONTAINER CONFIGURATIONS VALIDATED SUCCESSFULLY [PASS]")
        print("To launch containers locally when Docker Desktop is running:")
        print("  docker compose up --build -d")
    else:
        print("STATUS: VALIDATION FOUND WARNINGS OR ERRORS [FAIL]")
    print("=" * 80)


if __name__ == "__main__":
    main()
