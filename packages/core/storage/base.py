"""
Libra Core Storage - Common Interfaces & Helpers
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

_KEY_RE = re.compile(r"^[a-zA-Z0-9_./-]{1,512}$")


class StorageError(RuntimeError):
    """Raised when an object-storage operation fails."""


def sanitize_key(key: str) -> str:
    """Validate and normalize an object key, rejecting path traversal."""
    if not isinstance(key, str) or not key or not _KEY_RE.match(key):
        raise StorageError(f"Invalid storage key: {key!r}")
    if ".." in key:
        raise StorageError(f"Path traversal is not allowed in storage key: {key!r}")
    return key.strip("/")


@runtime_checkable
class ObjectStore(Protocol):
    """Minimal durable object-store contract."""

    @property
    def backend(self) -> str:
        """Human-readable backend name, e.g. 'local-disk' or 'supabase'."""
        ...

    async def save(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        """Persist ``data`` at ``key``. Returns the stored key."""
        ...

    async def read(self, key: str) -> bytes | None:
        """Return the bytes stored at ``key`` or None when missing."""
        ...

    async def exists(self, key: str) -> bool:
        """Return True when an object exists at ``key``."""
        ...

    async def delete(self, key: str) -> bool:
        """Remove the object at ``key``; True if one was removed."""
        ...

    async def signed_url(self, key: str, expires_seconds: int = 900) -> str | None:
        """Return a short-lived URL for the object, or None when unsupported."""
        ...
