"""
Libra Core Memory - PostgreSQL Conversation Store

Behavioral twin of :class:`SQLiteConversationStore` backed by PostgreSQL
(Supabase). Uses the psycopg 3 driver with dict rows and one connection per
operation so it degrades gracefully in serverless/FaaS and Render free tiers.

The module must be importable even when psycopg is not installed: connections
are opened lazily inside :meth:`_connect` so the rest of Libra can start and
fall back to SQLite with a warning.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from packages.core.memory.models import (
    Conversation,
    ConversationDetail,
    Message,
    parse_iso,
    utc_now_iso,
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    model TEXT NOT NULL,
    system_prompt TEXT,
    owner_id TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    token_hash TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages (conversation_id, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations (updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_owner ON conversations (owner_id, updated_at DESC);
"""


class PostgresConversationStore:
    """PostgreSQL-backed conversation store with an API identical to SQLite's."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv("DATABASE_URL") or ""
        self._schema_applied = False

    @property
    def backend(self) -> str:
        return "postgresql"

    def _require_url(self) -> None:
        if not self.database_url:
            raise RuntimeError(
                "DATABASE_URL is not configured. Set DATABASE_URL to a PostgreSQL "
                "(Supabase) connection string to enable the Postgres store."
            )

    def _connect(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "psycopg is required for the Postgres conversation store. "
                "Install it with `pip install 'psycopg[binary]>=3.1'` or remove DATABASE_URL."
            ) from exc
        self._require_url()
        return psycopg.connect(self.database_url, row_factory=dict_row)

    @staticmethod
    def _fmt(value) -> str:
        """Normalize a TIMESTAMPTZ cell to the ISO string used by the API layer."""
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    def init_schema(self) -> None:
        """Apply the conversation store DDL (idempotent)."""
        with self._connect() as conn:
            conn.execute(_SCHEMA_SQL)
            conn.commit()
        self._schema_applied = True

    def _configured(self) -> None:
        if not self._schema_applied:
            self.init_schema()

    # ------------------------------------------------------------------
    # Conversations & messages
    # ------------------------------------------------------------------

    def create_conversation(
        self,
        title: Optional[str] = None,
        model: str = "libra-llama-tied",
        system_prompt: Optional[str] = None,
        conv_id: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> Conversation:
        self._configured()
        cid = conv_id or f"conv-{uuid.uuid4().hex[:12]}"
        now = utc_now_iso()
        conv_title = title.strip() if title and title.strip() else "New Conversation"

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, title, model, system_prompt, owner_id, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
                """,
                (cid, conv_title, model, system_prompt, owner_id, now, now),
            )
            conn.commit()

        return Conversation(
            id=cid,
            title=conv_title,
            model=model,
            system_prompt=system_prompt,
            created_at=now,
            updated_at=now,
            message_count=0,
        )

    def get_conversation(
        self, conv_id: str, owner_id: Optional[str] = None
    ) -> Optional[ConversationDetail]:
        self._configured()
        with self._connect() as conn:
            if owner_id is not None:
                row = conn.execute(
                    "SELECT * FROM conversations WHERE id = %s AND (owner_id = %s OR owner_id IS NULL);",
                    (conv_id, owner_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM conversations WHERE id = %s;", (conv_id,)
                ).fetchone()
            if not row:
                return None

            msg_rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = %s ORDER BY created_at ASC;",
                (conv_id,),
            ).fetchall()

        messages = [
            Message(
                id=m["id"],
                conversation_id=m["conversation_id"],
                role=m["role"],
                content=m["content"],
                token_count=m["token_count"],
                created_at=self._fmt(m["created_at"]),
            )
            for m in msg_rows
        ]

        return ConversationDetail(
            id=row["id"],
            title=row["title"],
            model=row["model"],
            system_prompt=row["system_prompt"],
            created_at=self._fmt(row["created_at"]),
            updated_at=self._fmt(row["updated_at"]),
            message_count=len(messages),
            messages=messages,
        )

    def list_conversations(
        self, limit: int = 50, offset: int = 0, owner_id: Optional[str] = None
    ) -> list[Conversation]:
        self._configured()
        with self._connect() as conn:
            if owner_id is not None:
                rows = conn.execute(
                    """
                    SELECT c.*, COUNT(m.id) AS message_count
                    FROM conversations c
                    LEFT JOIN messages m ON c.id = m.conversation_id
                    WHERE c.owner_id = %s OR c.owner_id IS NULL
                    GROUP BY c.id
                    ORDER BY c.updated_at DESC
                    LIMIT %s OFFSET %s;
                    """,
                    (owner_id, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT c.*, COUNT(m.id) AS message_count
                    FROM conversations c
                    LEFT JOIN messages m ON c.id = m.conversation_id
                    GROUP BY c.id
                    ORDER BY c.updated_at DESC
                    LIMIT %s OFFSET %s;
                    """,
                    (limit, offset),
                ).fetchall()

        return [
            Conversation(
                id=r["id"],
                title=r["title"],
                model=r["model"],
                system_prompt=r["system_prompt"],
                created_at=self._fmt(r["created_at"]),
                updated_at=self._fmt(r["updated_at"]),
                message_count=r["message_count"],
            )
            for r in rows
        ]

    def update_conversation(
        self,
        conv_id: str,
        title: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> Optional[Conversation]:
        detail = self.get_conversation(conv_id, owner_id=owner_id)
        if not detail:
            return None

        new_title = title.strip() if title is not None and title.strip() else detail.title
        new_model = model.strip() if model is not None and model.strip() else detail.model
        new_prompt = system_prompt if system_prompt is not None else detail.system_prompt
        now = utc_now_iso()

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE conversations
                SET title = %s, model = %s, system_prompt = %s, updated_at = %s
                WHERE id = %s;
                """,
                (new_title, new_model, new_prompt, now, conv_id),
            )
            conn.commit()

        return Conversation(
            id=conv_id,
            title=new_title,
            model=new_model,
            system_prompt=new_prompt,
            created_at=detail.created_at,
            updated_at=now,
            message_count=detail.message_count,
        )

    def delete_conversation(self, conv_id: str, owner_id: Optional[str] = None) -> bool:
        self._configured()
        with self._connect() as conn:
            if owner_id is not None:
                cursor = conn.execute(
                    "DELETE FROM conversations WHERE id = %s AND (owner_id = %s OR owner_id IS NULL);",
                    (conv_id, owner_id),
                )
            else:
                cursor = conn.execute("DELETE FROM conversations WHERE id = %s;", (conv_id,))
            conn.commit()
            return cursor.rowcount > 0

    def delete_message(self, message_id: str) -> bool:
        self._configured()
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM messages WHERE id = %s;", (message_id,))
            conn.commit()
            return cursor.rowcount > 0

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        token_count: int = 0,
        msg_id: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> Message:
        self._configured()
        mid = msg_id or f"msg-{uuid.uuid4().hex[:12]}"
        now = utc_now_iso()

        with self._connect() as conn:
            exists = conn.execute(
                "SELECT id, title FROM conversations WHERE id = %s;", (conversation_id,)
            ).fetchone()

            if not exists:
                conn.execute(
                    """
                    INSERT INTO conversations (id, title, model, system_prompt, owner_id, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        conversation_id,
                        "New Conversation",
                        "libra-llama-tied",
                        None,
                        owner_id,
                        now,
                        now,
                    ),
                )
                current_title = "New Conversation"
            else:
                current_title = exists["title"]

            if role == "user" and current_title == "New Conversation":
                auto_title = content.strip().split("\n")[0][:40]
                if len(content.strip().split("\n")[0]) > 40:
                    auto_title += "..."
                conn.execute(
                    "UPDATE conversations SET title = %s, updated_at = %s WHERE id = %s;",
                    (auto_title, now, conversation_id),
                )
            else:
                conn.execute(
                    "UPDATE conversations SET updated_at = %s WHERE id = %s;",
                    (now, conversation_id),
                )

            conn.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, token_count, created_at)
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                (mid, conversation_id, role, content, token_count, now),
            )
            conn.commit()

        return Message(
            id=mid,
            conversation_id=conversation_id,
            role=role,
            content=content,
            token_count=token_count,
            created_at=now,
        )

    def get_messages(self, conversation_id: str, owner_id: Optional[str] = None) -> list[Message]:
        if owner_id is not None and not self.get_conversation(conversation_id, owner_id=owner_id):
            return []
        self._configured()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = %s ORDER BY created_at ASC;",
                (conversation_id,),
            ).fetchall()

        return [
            Message(
                id=r["id"],
                conversation_id=r["conversation_id"],
                role=r["role"],
                content=r["content"],
                token_count=r["token_count"],
                created_at=self._fmt(r["created_at"]),
            )
            for r in rows
        ]

    def clear_messages(self, conversation_id: str, owner_id: Optional[str] = None) -> bool:
        if owner_id is not None and not self.get_conversation(conversation_id, owner_id=owner_id):
            return False
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = %s;", (conversation_id,))
            conn.execute(
                "UPDATE conversations SET updated_at = %s WHERE id = %s;",
                (now, conversation_id),
            )
            conn.commit()
            return True

    def conversation_owner(self, conv_id: str) -> Optional[str]:
        self._configured()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT owner_id FROM conversations WHERE id = %s;", (conv_id,)
            ).fetchone()
        return row["owner_id"] if row else None

    # ------------------------------------------------------------------
    # Session management (guest identity isolation)
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create_session(self, token: str, ttl_days: int = 30) -> dict[str, str]:
        self._configured()
        now = utc_now_iso()
        expires_at = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()
        token_hash = self._hash_token(token)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, token_hash, created_at, last_seen_at, expires_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (token_hash)
                DO UPDATE SET
                    id = EXCLUDED.id,
                    created_at = EXCLUDED.created_at,
                    last_seen_at = EXCLUDED.last_seen_at,
                    expires_at = EXCLUDED.expires_at;
                """,
                (f"sess-{uuid.uuid4().hex[:12]}", token_hash, now, now, expires_at),
            )
            conn.commit()
        return {
            "token_hash": token_hash,
            "created_at": now,
            "expires_at": expires_at,
        }

    def get_session(self, token: str) -> Optional[dict[str, str]]:
        self._configured()
        token_hash = self._hash_token(token)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE token_hash = %s;", (token_hash,)
            ).fetchone()
        if not row:
            return None
        expires_at = parse_iso(self._fmt(row["expires_at"]))
        if expires_at and expires_at < datetime.now(timezone.utc):
            return None
        return {
            "session_id": row["id"],
            "created_at": self._fmt(row["created_at"]),
            "last_seen_at": self._fmt(row["last_seen_at"]),
            "expires_at": self._fmt(row["expires_at"]),
        }

    def touch_session(self, token: str) -> None:
        self._configured()
        token_hash = self._hash_token(token)
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET last_seen_at = %s WHERE token_hash = %s;",
                (now, token_hash),
            )
            conn.commit()

    def delete_session(self, token: str) -> bool:
        self._configured()
        token_hash = self._hash_token(token)
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE token_hash = %s;", (token_hash,))
            conn.commit()
            return cursor.rowcount > 0

    def cleanup_expired_sessions(self) -> int:
        self._configured()
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE expires_at < %s;", (now,))
            conn.commit()
            return cursor.rowcount

    def ping(self) -> dict[str, str]:
        """Connectivity check used by /health dependency checks."""
        with self._connect() as conn:
            version = conn.execute("SELECT version();").fetchone()
        return {"connected": True, "version": version["version"] if version else "unknown"}
