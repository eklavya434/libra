"""
Libra Core Storage - Supabase Storage Backend

Durable object storage for public cloud deployments where the host filesystem
is ephemeral. Talks to the Supabase Storage REST API (private bucket) using a
service-role (or anon) key. URLs are never exposed: downloads flow through the
FastAPI backend, and browser-facing files use signed URLs.

Protocol-level behavior is verified in unit tests with an injected httpx
transport; live validation requires real Supabase credentials (owner boundary).
"""

from __future__ import annotations

import httpx

from packages.core.storage.base import StorageError, sanitize_key

_OBJECT_TIMEOUT = 30.0


class SupabaseStorageStore:
    """Supabase Storage (S3-backed) object store."""

    def __init__(
        self,
        project_url: str,
        api_key: str,
        bucket: str = "libra-files",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.project_url = project_url.rstrip("/")
        self.api_key = api_key
        self.bucket = bucket
        self._transport = transport

    @property
    def backend(self) -> str:
        return "supabase"

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=_OBJECT_TIMEOUT, transport=self._transport)

    def _headers(self, upsert: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Client-Info": "libra",
        }
        if upsert:
            headers["x-upsert"] = "true"
        return headers

    def _object_url(self, key: str) -> str:
        safe = sanitize_key(key)
        return f"{self.project_url}/storage/v1/object/{self.bucket}/{safe}"

    async def save(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        safe = sanitize_key(key)
        async with self._client() as client:
            resp = await client.post(
                f"{self.project_url}/storage/v1/object/{self.bucket}/{safe}",
                headers=self._headers(upsert=True),
                files={"file": (safe, data)},
            )
        if resp.status_code not in (200, 201):
            raise StorageError(
                f"Supabase storage PUT failed with status {resp.status_code} for key {key!r}."
            )
        return safe

    async def read(self, key: str) -> bytes | None:
        async with self._client() as client:
            resp = await client.get(self._object_url(key), headers=self._headers())
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            raise StorageError(f"Supabase storage GET failed with status {resp.status_code}.")
        return resp.content

    async def exists(self, key: str) -> bool:
        async with self._client() as client:
            resp = await client.head(self._object_url(key), headers=self._headers())
        return resp.status_code in (200, 201)

    async def delete(self, key: str) -> bool:
        async with self._client() as client:
            resp = await client.request("DELETE", self._object_url(key), headers=self._headers())
        return resp.status_code in (200, 204)

    async def signed_url(self, key: str, expires_seconds: int = 900) -> str | None:
        safe = sanitize_key(key)
        async with self._client() as client:
            resp = await client.post(
                f"{self.project_url}/storage/v1/object/sign/{self.bucket}/{safe}",
                headers=self._headers(),
                json={"expiresIn": expires_seconds},
            )
        if resp.status_code != 200:
            return None
        payload = resp.json()
        signed = payload.get("signedURL")
        if not signed:
            return None
        return f"{self.project_url}{signed}"
