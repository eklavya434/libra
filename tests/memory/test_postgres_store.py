"""
Tests for the PostgreSQL conversation store and the DATABASE_URL resolver.

Resolver behavior is tested with fakes so no network is required. Live
parity tests run only when a real Postgres connection string is supplied by
the developer/owner (e.g. a Supabase URL) — they are skipped otherwise so the
suite stays deterministic and offline-friendly.
"""

from __future__ import annotations

import os

import pytest

import packages.core.memory.postgres_store as pg_module
import packages.core.memory.sqlite_store as sqlite_module


@pytest.fixture(autouse=True)
def reset_global_store(monkeypatch):
    """Restore the singleton store after each test."""
    monkeypatch.setattr(sqlite_module, "_global_store", None)


def _db_url():
    return os.getenv("LIBRA_TEST_DATABASE_URL") or os.getenv("DATABASE_URL") or ""


LIVE_POSTGRES = pytest.mark.skipif(
    not _db_url(),
    reason="Set LIBRA_TEST_DATABASE_URL (or DATABASE_URL) to a real Postgres/Supabase URL to run live parity tests.",
)


class _FakePostgresStore:
    """Records construction args; used to prove the resolver selects Postgres."""

    def __init__(self, database_url=None):
        self.database_url = database_url

    def init_schema(self) -> None:
        return None


def test_resolver_uses_sqlite_by_default(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    store = sqlite_module.get_conversation_store()
    assert isinstance(store, sqlite_module.SQLiteConversationStore)


def test_resolver_selects_postgres_when_database_url_set(monkeypatch):
    monkeypatch.setattr(pg_module, "PostgresConversationStore", _FakePostgresStore)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/libra")
    store = sqlite_module.get_conversation_store()
    assert isinstance(store, _FakePostgresStore)
    assert store.database_url == "postgresql://user:pass@host:5432/libra"


def test_resolver_falls_back_to_sqlite_when_postgres_unavailable(monkeypatch):
    def _boom(cls, database_url=None):
        raise RuntimeError("cannot reach database")

    monkeypatch.setattr(pg_module, "PostgresConversationStore", classmethod(_boom))
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/libra")
    store = sqlite_module.get_conversation_store()
    assert isinstance(store, sqlite_module.SQLiteConversationStore)


def test_postgres_store_backend_name_and_url_required():
    store = pg_module.PostgresConversationStore("postgresql://u:p@h/db")
    assert store.backend == "postgresql"
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        pg_module.PostgresConversationStore("").init_schema()


def test_migration_sql_matches_store_schema_parity():
    migration = os.path.join("supabase", "migrations", "0001_conversation_store.sql")
    assert os.path.exists(migration)
    sql = open(migration, "r", encoding="utf-8").read()
    for table in (
        "CREATE TABLE IF NOT EXISTS conversations",
        "CREATE TABLE IF NOT EXISTS messages",
        "CREATE TABLE IF NOT EXISTS sessions",
    ):
        assert table in sql
    assert "ON DELETE CASCADE" in sql


@LIVE_POSTGRES
def test_live_roundtrip_conversation_and_sessions():
    """End-to-end parity run against a real Postgres/Supabase instance."""
    store = pg_module.PostgresConversationStore(_db_url())
    store.delete_conversation("pg-test-conv")

    conv = store.create_conversation(
        title="Parity Test", model="gemini-2.5-flash", owner_id="pg-tester"
    )
    assert conv.id == "pg-test-conv" or conv.title == "Parity Test"

    m1 = store.add_message(conv.id, "user", "Hello Postgres", owner_id="pg-tester")
    m2 = store.add_message(conv.id, "assistant", "Hello back", owner_id="pg-tester")
    assert m1.content == "Hello Postgres"
    assert m2.token_count == 0

    detail = store.get_conversation(conv.id, owner_id="pg-tester")
    assert detail is not None
    assert len(detail.messages) == 2
    assert detail.message_count == 2

    listed = store.list_conversations(owner_id="pg-tester")
    assert any(c.id == conv.id for c in listed)

    updated = store.update_conversation(conv.id, model="gpt-4o-mini", owner_id="pg-tester")
    assert updated is not None and updated.model == "gpt-4o-mini"

    # Session lifecycle
    meta = store.create_session("pg-live-token-123", ttl_days=7)
    assert meta["token_hash"]
    sess = store.get_session("pg-live-token-123")
    assert sess is not None and sess["session_id"]
    store.touch_session("pg-live-token-123")
    assert store.delete_session("pg-live-token-123") is True

    assert store.delete_message(m2.id) is True
    assert store.delete_conversation(conv.id, owner_id="pg-tester") is True
    assert store.get_conversation(conv.id, owner_id="pg-tester") is None


@LIVE_POSTGRES
def test_live_add_message_auto_creates_conversation():
    store = pg_module.PostgresConversationStore(_db_url())
    store.delete_message("pg-auto-msg")
    store.delete_conversation("pg-auto-conv")
    msg = store.add_message(
        "pg-auto-conv",
        "user",
        "A long first user message that should be auto titled",
        owner_id="pg-tester",
    )
    detail = store.get_conversation("pg-auto-conv")
    assert detail is not None
    assert detail.title.startswith("A long first user message that should b")
    store.delete_conversation("pg-auto-conv")
