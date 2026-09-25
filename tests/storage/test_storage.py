"""
Tests for the object-storage abstraction (memory, local disk, Supabase protocol)
and the get_object_store() backend resolver.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from packages.core.storage import (
    LocalDiskObjectStore,
    MemoryObjectStore,
    StorageError,
    SupabaseStorageStore,
    get_object_store,
)


def _run(coro):
    return asyncio.run(coro)


def test_local_disk_roundtrip(tmp_path):
    store = LocalDiskObjectStore(root=tmp_path / "uploads")
    assert store.backend == "local-disk"

    key = "documents/abc123/report.txt"
    _run(store.save(key, b"hello world", content_type="text/plain"))
    assert _run(store.exists(key)) is True
    assert _run(store.read(key)) == b"hello world"
    assert _run(store.signed_url(key)) is None
    assert _run(store.delete(key)) is True
    assert _run(store.read(key)) is None


def test_local_disk_rejects_path_traversal(tmp_path):
    store = LocalDiskObjectStore(root=tmp_path / "uploads")
    with pytest.raises(StorageError):
        _run(store.save("../../evil.txt", b"x"))
    with pytest.raises(StorageError):
        _run(store.save("..%2Fevil.txt", b"x"))


def test_memory_store_is_ephemeral_and_opt_in(tmp_path):
    store = MemoryObjectStore()
    key = "documents/x/a.bin"
    _run(store.save(key, b"\x00\x01"))
    assert _run(store.exists(key)) is True
    assert _run(store.read(key)) == b"\x00\x01"
    assert _run(store.signed_url(key)) is None
    assert _run(store.delete(key)) is True
    assert _run(store.exists(key)) is False


def test_resolver_defaults_to_local_disk(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    monkeypatch.setenv("LIBRA_STORAGE_BACKEND", "")
    assert isinstance(get_object_store(), LocalDiskObjectStore)


def test_resolver_selects_memory_backend_explicitly(monkeypatch):
    monkeypatch.setenv("LIBRA_STORAGE_BACKEND", "memory")
    assert isinstance(get_object_store(), MemoryObjectStore)


def test_resolver_selects_supabase_when_configured(monkeypatch):
    monkeypatch.setenv("LIBRA_STORAGE_BACKEND", "")
    monkeypatch.setenv("SUPABASE_URL", "https://xyz.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "svc-secret")
    store = get_object_store()
    assert isinstance(store, SupabaseStorageStore)
    assert store.bucket == "libra-files"


def test_resolver_honors_bucket_override(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://xyz.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-secret")
    monkeypatch.setenv("SUPABASE_STORAGE_BUCKET", "my-bucket")
    store = get_object_store()
    assert isinstance(store, SupabaseStorageStore)
    assert store.bucket == "my-bucket"


@pytest.mark.asyncio
async def test_supabase_store_save_uses_auth_and_upsert():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.startswith("/storage/v1/object/libra-files/documents/x/report.txt")
        assert request.headers["authorization"] == "Bearer svc-secret"
        assert request.headers["x-upsert"] == "true"
        assert request.method == "POST"
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    store = _store_with_transport(transport)
    assert await store.save("documents/x/report.txt", b"data") == "documents/x/report.txt"


@pytest.mark.asyncio
async def test_supabase_store_read_missing_returns_none():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    store = _store_with_transport(httpx.MockTransport(handler))
    assert await store.read("documents/x/missing.txt") is None


@pytest.mark.asyncio
async def test_supabase_store_exists_via_head():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "HEAD"
        return httpx.Response(200)

    store = _store_with_transport(httpx.MockTransport(handler))
    assert await store.exists("documents/x/report.txt") is True


@pytest.mark.asyncio
async def test_supabase_store_delete():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        return httpx.Response(200, json={"message": "Successfully deleted"})

    store = _store_with_transport(httpx.MockTransport(handler))
    assert await store.delete("documents/x/report.txt") is True


@pytest.mark.asyncio
async def test_supabase_store_signed_url():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/storage/v1/object/sign/libra-files/documents/x/report.txt"
        assert b'"expiresIn"' in request.content
        return httpx.Response(
            200,
            json={
                "signedURL": "/storage/v1/object/sign/libra-files/documents/x/report.txt?token=abc"
            },
        )

    store = _store_with_transport(httpx.MockTransport(handler))
    url = await store.signed_url("documents/x/report.txt", expires_seconds=600)
    assert (
        url
        == "https://xyz.supabase.co/storage/v1/object/sign/libra-files/documents/x/report.txt?token=abc"
    )


def _store_with_transport(transport):
    return SupabaseStorageStore(
        "https://xyz.supabase.co", "svc-secret", "libra-files", transport=transport
    )
