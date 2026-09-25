"""
Libra Core Storage - Explicit In-Memory Backend

Intentionally NOT durable. Kept for CPU-lab offline experimentation and as a
test double. It is only selected when LIBRA_STORAGE_BACKEND=memory is set
explicitly; production deployments use LocalDisk or Supabase.
"""

from __future__ import annotations

from packages.core.storage.base import sanitize_key


class MemoryObjectStore:
    """Ephemeral dict-backed object store (explicit opt-in only)."""

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}
        self._content_types: dict[str, str] = {}

    @property
    def backend(self) -> str:
        return "memory"

    async def save(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        safe = sanitize_key(key)
        self._objects[safe] = data
        self._content_types[safe] = content_type
        return safe

    async def read(self, key: str) -> bytes | None:
        return self._objects.get(sanitize_key(key))

    async def exists(self, key: str) -> bool:
        return sanitize_key(key) in self._objects

    async def delete(self, key: str) -> bool:
        safe = sanitize_key(key)
        removed = safe in self._objects
        self._objects.pop(safe, None)
        self._content_types.pop(safe, None)
        return removed

    async def signed_url(self, key: str, expires_seconds: int = 900) -> str | None:
        return None
