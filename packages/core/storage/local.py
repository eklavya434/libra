"""
Libra Core Storage - Local Disk Backend

Zero-cost durable object storage on single-host deployments. Keys map directly
to files under a root directory with traversal protection on both write and read.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from packages.core.storage.base import StorageError, sanitize_key


class LocalDiskObjectStore:
    """Filesystem-backed object store rooted at ``root`` (default data/uploads)."""

    def __init__(self, root: str | Path = "data/uploads") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def backend(self) -> str:
        return "local-disk"

    def _path_for(self, key: str) -> Path:
        safe = sanitize_key(key)
        path = self.root.joinpath(*safe.split("/"))
        if (
            self.root.resolve() not in path.resolve().parents
            and path.resolve() != self.root.resolve()
        ):
            raise StorageError(f"Resolved path escapes storage root: {key!r}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    async def save(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        safe = sanitize_key(key)
        path = self._path_for(safe)
        path.write_bytes(data)
        return safe

    async def read(self, key: str) -> bytes | None:
        path = self._path_for(key)
        if not path.is_file():
            return None
        return path.read_bytes()

    async def exists(self, key: str) -> bool:
        return self._path_for(key).is_file()

    async def delete(self, key: str) -> bool:
        path = self._path_for(key)
        if not path.is_file():
            return False
        path.unlink()
        return True

    async def signed_url(self, key: str, expires_seconds: int = 900) -> str | None:
        # No remote URL is available: the file is served by the FastAPI backend.
        return None

    @staticmethod
    def guess_content_type(key: str) -> str:
        """Best-effort content type for serving stored files back to clients."""
        return mimetypes.guess_type(key)[0] or "application/octet-stream"
