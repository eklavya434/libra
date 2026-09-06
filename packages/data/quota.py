"""
Libra Data - Storage Quota & Hardware Safeguards
Enforces the strict 15 GB quota across all data, model weights, and caches.
"""

import os
import shutil
from typing import Any

MAX_STORAGE_QUOTA_MB = 15360.0  # 15 GB strict limit
MIN_HOST_FREE_DISK_GB = 5.0  # Safety floor for host OS drive


class QuotaExceededError(Exception):
    """Raised when an operation would exceed the project storage quota."""


class LowDiskSpaceError(Exception):
    """Raised when host machine free disk space is critically low."""


def get_directory_size_mb(path: str) -> float:
    """Recursively computes size of a directory in Megabytes."""
    if not os.path.exists(path):
        return 0.0

    total_bytes = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total_bytes += os.path.getsize(fp)
            except (OSError, FileNotFoundError):
                pass
    return round(total_bytes / (1024 * 1024), 2)


def get_current_project_footprint_mb(workspace_root: str = ".") -> dict[str, float]:
    """Measures current storage consumed by venv, node_modules, data, models, and checkpoints."""
    targets = {
        "venv": os.path.join(workspace_root, ".venv"),
        "node_modules": os.path.join(workspace_root, "apps", "frontend", "node_modules"),
        "data": os.path.join(workspace_root, "data"),
        "models": os.path.join(workspace_root, "models"),
        "checkpoints": os.path.join(workspace_root, "checkpoints"),
    }

    usage: dict[str, float] = {}
    total = 0.0
    for key, path in targets.items():
        size = get_directory_size_mb(path)
        usage[key] = size
        total += size

    usage["total"] = round(total, 2)
    return usage


def validate_storage_quota(additional_mb: float = 0.0, workspace_root: str = ".") -> dict[str, Any]:
    """Validates that a proposed data operation will not violate storage limits.

    Raises:
        QuotaExceededError: If total project storage exceeds 15 GB.
        LowDiskSpaceError: If host drive has less than 5 GB free.
    """
    # 1. Host OS disk check
    disk = shutil.disk_usage(workspace_root)
    host_free_gb = disk.free / (1024**3)
    if host_free_gb < MIN_HOST_FREE_DISK_GB:
        raise LowDiskSpaceError(
            f"Host drive has only {host_free_gb:.2f} GB free (Minimum required: {MIN_HOST_FREE_DISK_GB} GB)."
        )

    # 2. Project quota check
    usage = get_current_project_footprint_mb(workspace_root)
    project_total_mb = usage["total"]
    projected_total_mb = project_total_mb + additional_mb

    if projected_total_mb > MAX_STORAGE_QUOTA_MB:
        raise QuotaExceededError(
            f"Operation would exceed 15 GB storage quota! "
            f"Current: {project_total_mb:.1f} MB, Requested: +{additional_mb:.1f} MB, "
            f"Projected: {projected_total_mb:.1f} MB (Limit: {MAX_STORAGE_QUOTA_MB:.1f} MB)"
        )

    remaining_mb = round(MAX_STORAGE_QUOTA_MB - projected_total_mb, 2)
    return {
        "current_total_mb": project_total_mb,
        "additional_mb": additional_mb,
        "projected_total_mb": round(projected_total_mb, 2),
        "quota_limit_mb": MAX_STORAGE_QUOTA_MB,
        "remaining_mb": remaining_mb,
        "host_free_gb": round(host_free_gb, 2),
        "breakdown": usage,
    }
