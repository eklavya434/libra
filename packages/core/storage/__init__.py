"""
Libra Core Storage - Object Storage Abstraction

A tiny, honest object-store interface. Backends:

- ``MemoryObjectStore``: explicit per-process in-memory (lab/testing only, NOT durable).
- ``LocalDiskObjectStore``: the zero-cost durable default for single-node hosts.
- ``SupabaseStorageStore``: Supabase Storage (private bucket + signed URLs) for public
  cloud deployments where the host filesystem is ephemeral.

``get_object_store()`` resolves the backend from environment configuration and never
pretends to be durable when it is not.
"""

from __future__ import annotations

import logging
import os

from packages.core.storage.base import ObjectStore, StorageError, sanitize_key
from packages.core.storage.local import LocalDiskObjectStore
from packages.core.storage.memory import MemoryObjectStore
from packages.core.storage.supabase import SupabaseStorageStore

_logger = logging.getLogger("libra.storage")


def get_object_store() -> ObjectStore:
    """Resolve the configured object store backend (Singleton per process)."""
    backend = os.getenv("LIBRA_STORAGE_BACKEND", "").strip().lower()

    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    supabase_key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        or os.getenv("SUPABASE_ANON_KEY", "").strip()
    )

    if backend == "supabase" or (supabase_url and supabase_key):
        if backend not in ("", "supabase", "local"):
            _logger.warning(
                "Unknown LIBRA_STORAGE_BACKEND %r; using configured Supabase backend.", backend
            )
        bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "libra-files").strip()
        return SupabaseStorageStore(
            project_url=supabase_url,
            api_key=supabase_key,
            bucket=bucket,
        )

    if backend == "memory":
        return MemoryObjectStore()

    # Default: on-disk storage. This is durable on a single host (dev laptop /
    # persistent-volume VPS) and honest about the ephemeral-cloud caveat.
    if backend not in ("", "local", "disk"):
        _logger.warning("Unknown LIBRA_STORAGE_BACKEND %r; using LocalDisk storage.", backend)
    _logger.warning(
        "Using LocalDisk object storage (data/uploads). On ephemeral hosts (Render free/Railway) "
        "files are lost on redeploy — configure SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY for durable storage."
    )
    return LocalDiskObjectStore()


__all__ = [
    "ObjectStore",
    "StorageError",
    "sanitize_key",
    "MemoryObjectStore",
    "LocalDiskObjectStore",
    "SupabaseStorageStore",
    "get_object_store",
]
