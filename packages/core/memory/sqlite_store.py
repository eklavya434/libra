"""
Libra Core Memory - SQLite Conversation Store

Zero-dependency SQLite storage engine for multi-turn conversation sessions
and message history. Features WAL mode, cascading deletions, and thread safety.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from packages.core.memory.models import (
    Conversation,
    ConversationDetail,
    Message,
    parse_iso,
    utc_now_iso,
)


class SQLiteConversationStore:
    """Thread-safe, lightweight SQLite store for conversation memory."""

    def __init__(self, db_path: str | Path = "data/conversations.db") -> None:
        self.db_path = str(db_path)
        self._mem_conn: Optional[sqlite3.Connection] = None
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            if self._mem_conn is None:
                self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._mem_conn.row_factory = sqlite3.Row
                self._mem_conn.execute("PRAGMA foreign_keys = ON;")
            return self._mem_conn

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_db(self) -> None:
        """Initialize tables and indexes if they do not already exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    model TEXT NOT NULL,
                    system_prompt TEXT,
                    owner_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    token_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_token_hash
                ON sessions (token_hash);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_conv
                ON messages (conversation_id, created_at ASC);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_updated
                ON conversations (updated_at DESC);
                """
            )
            # Safe migration: add owner_id to databases created before multi-user support.
            columns = [
                row[1] for row in conn.execute("PRAGMA table_info(conversations);").fetchall()
            ]
            if "owner_id" not in columns:
                conn.execute("ALTER TABLE conversations ADD COLUMN owner_id TEXT;")
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_owner
                ON conversations (owner_id, updated_at DESC);
                """
            )
            conn.commit()

    def create_conversation(
        self,
        title: Optional[str] = None,
        model: str = "libra-llama-tied",
        system_prompt: Optional[str] = None,
        conv_id: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> Conversation:
        """Create a new conversation session."""
        cid = conv_id or f"conv-{uuid.uuid4().hex[:12]}"
        now = utc_now_iso()
        conv_title = title.strip() if title and title.strip() else "New Conversation"

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, title, model, system_prompt, owner_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
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
        """Fetch a conversation session along with all ordered messages.

        When ``owner_id`` is provided the conversation must be owned by that
        session (or be a legacy row with no owner) or ``None`` is returned.
        """
        with self._get_connection() as conn:
            if owner_id is not None:
                row = conn.execute(
                    "SELECT * FROM conversations WHERE id = ? AND (owner_id = ? OR owner_id IS NULL);",
                    (conv_id, owner_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM conversations WHERE id = ?;", (conv_id,)
                ).fetchone()
            if not row:
                return None

            msg_rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC;",
                (conv_id,),
            ).fetchall()

        messages = [
            Message(
                id=m["id"],
                conversation_id=m["conversation_id"],
                role=m["role"],
                content=m["content"],
                token_count=m["token_count"],
                created_at=m["created_at"],
            )
            for m in msg_rows
        ]

        return ConversationDetail(
            id=row["id"],
            title=row["title"],
            model=row["model"],
            system_prompt=row["system_prompt"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            message_count=len(messages),
            messages=messages,
        )

    def list_conversations(
        self, limit: int = 50, offset: int = 0, owner_id: Optional[str] = None
    ) -> list[Conversation]:
        """List conversation summaries sorted by most recently updated.

        When ``owner_id`` is provided only conversations owned by that session
        (plus legacy owner-less rows) are returned.
        """
        with self._get_connection() as conn:
            if owner_id is not None:
                rows = conn.execute(
                    """
                    SELECT c.*, COUNT(m.id) as message_count
                    FROM conversations c
                    LEFT JOIN messages m ON c.id = m.conversation_id
                    WHERE c.owner_id = ? OR c.owner_id IS NULL
                    GROUP BY c.id
                    ORDER BY c.updated_at DESC
                    LIMIT ? OFFSET ?;
                    """,
                    (owner_id, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT c.*, COUNT(m.id) as message_count
                    FROM conversations c
                    LEFT JOIN messages m ON c.id = m.conversation_id
                    GROUP BY c.id
                    ORDER BY c.updated_at DESC
                    LIMIT ? OFFSET ?;
                    """,
                    (limit, offset),
                ).fetchall()

        return [
            Conversation(
                id=r["id"],
                title=r["title"],
                model=r["model"],
                system_prompt=r["system_prompt"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
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
        """Update conversation properties (scoped to ``owner_id``)."""
        detail = self.get_conversation(conv_id, owner_id=owner_id)
        if not detail:
            return None

        new_title = title.strip() if title is not None and title.strip() else detail.title
        new_model = model.strip() if model is not None and model.strip() else detail.model
        new_prompt = system_prompt if system_prompt is not None else detail.system_prompt
        now = utc_now_iso()

        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE conversations
                SET title = ?, model = ?, system_prompt = ?, updated_at = ?
                WHERE id = ?;
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
        """Delete a conversation and cascade delete its messages.

        When ``owner_id`` is provided the row must be owned by that session
        (or be a legacy owner-less row).
        """
        with self._get_connection() as conn:
            if owner_id is not None:
                cursor = conn.execute(
                    "DELETE FROM conversations WHERE id = ? AND (owner_id = ? OR owner_id IS NULL);",
                    (conv_id, owner_id),
                )
            else:
                cursor = conn.execute("DELETE FROM conversations WHERE id = ?;", (conv_id,))
            conn.commit()
            return cursor.rowcount > 0

    def delete_message(self, message_id: str) -> bool:
        """Delete a single message by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM messages WHERE id = ?;", (message_id,))
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
        """Append a message to the conversation and bump updated_at.

        When ``owner_id`` is provided it is stamped onto any auto-created
        conversation. Access control itself is enforced by endpoints via
        :meth:`get_conversation`.
        """
        mid = msg_id or f"msg-{uuid.uuid4().hex[:12]}"
        now = utc_now_iso()

        with self._get_connection() as conn:
            # Check conversation exists
            exists = conn.execute(
                "SELECT id, title FROM conversations WHERE id = ?;", (conversation_id,)
            ).fetchone()

            if not exists:
                # Auto-create conversation if missing
                conn.execute(
                    """
                    INSERT INTO conversations (id, title, model, system_prompt, owner_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
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

            # Auto-title conversation on first user message if still default
            if role == "user" and current_title == "New Conversation":
                auto_title = content.strip().split("\n")[0][:40]
                if len(content.strip().split("\n")[0]) > 40:
                    auto_title += "..."
                conn.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?;",
                    (auto_title, now, conversation_id),
                )
            else:
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?;",
                    (now, conversation_id),
                )

            conn.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, token_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?);
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
        """Fetch all messages for a given conversation ordered chronologically.

        When ``owner_id`` is provided the conversation must be accessible to
        that session or an empty list is returned.
        """
        if owner_id is not None and not self.get_conversation(conversation_id, owner_id=owner_id):
            return []
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC;",
                (conversation_id,),
            ).fetchall()

        return [
            Message(
                id=r["id"],
                conversation_id=r["conversation_id"],
                role=r["role"],
                content=r["content"],
                token_count=r["token_count"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def clear_messages(self, conversation_id: str, owner_id: Optional[str] = None) -> bool:
        """Clear all messages from a conversation while preserving metadata.

        When ``owner_id`` is provided the conversation must be accessible to
        that session.
        """
        if owner_id is not None and not self.get_conversation(conversation_id, owner_id=owner_id):
            return False
        now = utc_now_iso()
        with self._get_connection() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?;", (conversation_id,))
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?;",
                (now, conversation_id),
            )
            conn.commit()
            return True

    def conversation_owner(self, conv_id: str) -> Optional[str]:
        """Return the owner_id for a conversation row, or None if it does not exist."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT owner_id FROM conversations WHERE id = ?;", (conv_id,)
            ).fetchone()
        return row["owner_id"] if row else None

    # ------------------------------------------------------------------
    # Session management (guest identity isolation)
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_token(token: str) -> str:
        """SHA-256 hash of a raw session token (tokens are high-entropy)."""
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create_session(self, token: str, ttl_days: int = 30) -> dict[str, str]:
        """Persist a new session from a raw token. Returns stored metadata."""
        now = utc_now_iso()
        expires_at = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()
        token_hash = self._hash_token(token)
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions (id, token_hash, created_at, last_seen_at, expires_at)
                VALUES (?, ?, ?, ?, ?);
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
        """Validate a raw token. Returns session metadata or None when invalid/expired."""
        token_hash = self._hash_token(token)
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE token_hash = ?;", (token_hash,)
            ).fetchone()
        if not row:
            return None
        expires_at = parse_iso(row["expires_at"])
        if expires_at and expires_at < datetime.now(timezone.utc):
            return None
        return {
            "session_id": row["id"],
            "created_at": row["created_at"],
            "last_seen_at": row["last_seen_at"],
            "expires_at": row["expires_at"],
        }

    def touch_session(self, token: str) -> None:
        """Refresh the last-seen timestamp and extend an unexpired session."""
        token_hash = self._hash_token(token)
        now = utc_now_iso()
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE sessions SET last_seen_at = ? WHERE token_hash = ?;",
                (now, token_hash),
            )
            conn.commit()

    def delete_session(self, token: str) -> bool:
        """Remove a session (logout)."""
        token_hash = self._hash_token(token)
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE token_hash = ?;", (token_hash,))
            conn.commit()
            return cursor.rowcount > 0

    def cleanup_expired_sessions(self) -> int:
        """Delete expired sessions. Returns the number of rows removed."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE expires_at < ?;", (now,))
            conn.commit()
            return cursor.rowcount


_global_store: Optional[SQLiteConversationStore] = None


def get_conversation_store() -> SQLiteConversationStore:
    """Singleton factory for application-wide SQLite conversation store."""
    global _global_store
    if _global_store is None:
        db_path = os.getenv("LIBRA_DB_PATH", "data/conversations.db")
        _global_store = SQLiteConversationStore(db_path=db_path)
    return _global_store
