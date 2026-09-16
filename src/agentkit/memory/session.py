"""Sessions and state persistence between runs (block 13).

Session
-------
Captures one user's conversation thread across multiple process runs.
Fields:
  session_id  — UUID string, auto-generated
  user_id     — caller-defined identifier (username, email, UUID, etc.)
  created_at  — float timestamp (time.time())
  updated_at  — float timestamp, updated on every write
  events      — full event history across all runs
  state       — dict that survives runs; tools write via context.state["session_state"]
  metadata    — caller annotations (source app, tags, model, etc.)

SessionStore (Protocol)
-----------------------
Structural interface — InMemorySessionStore and SqliteSessionStore both satisfy
it without inheriting from a common base class.

SqliteSessionStore
------------------
  - Database path: SESSION_DB_PATH env var, default ~/.agentkit/sessions.db
  - File permissions: 600 (owner read/write only; set after CREATE)
  - Tables: sessions, events, schema_version
  - Events stored as JSON via Event.model_dump_json()
  - ON DELETE CASCADE: event rows removed when session deleted
  - Indices on (session_id) and (user_id) for fast lookups
  - delete_user() runs in one transaction (privacy / GDPR)
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, runtime_checkable

from typing_extensions import Protocol

from agentkit.types import ContentItem, Event

logger = logging.getLogger(__name__)

_DEFAULT_DB = os.path.join(os.path.expanduser("~"), ".agentkit", "sessions.db")
_SCHEMA_VERSION = 1


# ── Domain model ──────────────────────────────────────────────────────────────


@dataclass
class Session:
    """One user's conversation thread, persisted across process restarts."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "default"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    events: list[Event] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Protocol ──────────────────────────────────────────────────────────────────


@runtime_checkable
class SessionStore(Protocol):
    """Persistence contract for sessions.

    All methods are synchronous — SQLite is synchronous and asyncio wrappers
    would add complexity without benefit for the single-threaded CLI use-case.
    """

    def create(self, user_id: str = "default", metadata: dict[str, Any] | None = None) -> Session:
        """Create and persist a new empty session for *user_id*."""
        ...

    def get(self, session_id: str) -> Session | None:
        """Return the session, or None if it does not exist."""
        ...

    def append_events(self, session_id: str, events: list[Event]) -> None:
        """Append *events* to the session's history and update updated_at."""
        ...

    def update_state(self, session_id: str, patch: dict[str, Any]) -> None:
        """Merge *patch* into the session's state dict and update updated_at."""
        ...

    def list_sessions(self, user_id: str, limit: int = 50) -> list[Session]:
        """Return up to *limit* sessions for *user_id*, newest first."""
        ...

    def delete(self, session_id: str) -> bool:
        """Delete the session. Returns True if it existed."""
        ...

    def delete_user(self, user_id: str) -> int:
        """Delete all sessions for *user_id*. Returns the count deleted."""
        ...


# ── InMemorySessionStore ───────────────────────────────────────────────────────


class InMemorySessionStore:
    """Thread-unsafe in-process store for tests and REPL demos."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self, user_id: str = "default", metadata: dict[str, Any] | None = None) -> Session:
        s = Session(user_id=user_id, metadata=dict(metadata or {}))
        self._sessions[s.session_id] = s
        return s

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def append_events(self, session_id: str, events: list[Event]) -> None:
        s = self._sessions.get(session_id)
        if s is None:
            raise KeyError(session_id)
        s.events.extend(events)
        s.updated_at = time.time()

    def update_state(self, session_id: str, patch: dict[str, Any]) -> None:
        s = self._sessions.get(session_id)
        if s is None:
            raise KeyError(session_id)
        s.state.update(patch)
        s.updated_at = time.time()

    def list_sessions(self, user_id: str, limit: int = 50) -> list[Session]:
        matching = [s for s in self._sessions.values() if s.user_id == user_id]
        matching.sort(key=lambda s: s.updated_at, reverse=True)
        return matching[:limit]

    def delete(self, session_id: str) -> bool:
        if session_id not in self._sessions:
            return False
        del self._sessions[session_id]
        return True

    def delete_user(self, user_id: str) -> int:
        to_delete = [sid for sid, s in self._sessions.items() if s.user_id == user_id]
        for sid in to_delete:
            del self._sessions[sid]
        return len(to_delete)


# ── SqliteSessionStore ────────────────────────────────────────────────────────

_CREATE_SCHEMA_VERSION = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
"""

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    state       TEXT NOT NULL DEFAULT '{}',
    metadata    TEXT NOT NULL DEFAULT '{}'
);
"""

_CREATE_EVENTS = """
CREATE TABLE IF NOT EXISTS events (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    position    INTEGER NOT NULL,
    data        TEXT NOT NULL
);
"""

_IDX_SESSION = "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, position);"
_IDX_USER = "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, updated_at);"


class SqliteSessionStore:
    """Persistent session store backed by SQLite.

    Thread-safety: SQLite's WAL mode allows concurrent readers; writes are
    serialised by the OS file lock.  For the single-process CLI this is
    sufficient.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or os.getenv("SESSION_DB_PATH") or _DEFAULT_DB
        self._ensure_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA foreign_keys = ON;")
        con.execute("PRAGMA journal_mode = WAL;")
        con.row_factory = sqlite3.Row
        return con

    def _ensure_db(self) -> None:
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        con = self._connect()
        with con:
            con.execute(_CREATE_SCHEMA_VERSION)
            con.execute(_CREATE_SESSIONS)
            con.execute(_CREATE_EVENTS)
            con.execute(_IDX_SESSION)
            con.execute(_IDX_USER)
            # Insert schema version if not present
            if con.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 0:
                con.execute("INSERT INTO schema_version VALUES (?)", (_SCHEMA_VERSION,))
        con.close()

        # Restrict file permissions to owner read/write (600)
        try:
            os.chmod(self.db_path, 0o600)
        except OSError:
            pass

    # ── helpers ───────────────────────────────────────────────────────────────

    def _row_to_session(self, row: sqlite3.Row, events: list[Event]) -> Session:
        return Session(
            session_id=row["session_id"],
            user_id=row["user_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            events=events,
            state=json.loads(row["state"]),
            metadata=json.loads(row["metadata"]),
        )

    def _load_events(self, con: sqlite3.Connection, session_id: str) -> list[Event]:
        rows = con.execute(
            "SELECT data FROM events WHERE session_id = ? ORDER BY position",
            (session_id,),
        ).fetchall()
        events: list[Event] = []
        for r in rows:
            try:
                events.append(Event.model_validate_json(r["data"]))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed event in session %s: %s", session_id, exc)
        return events

    # ── Protocol implementation ───────────────────────────────────────────────

    def create(self, user_id: str = "default", metadata: dict[str, Any] | None = None) -> Session:
        s = Session(user_id=user_id, metadata=dict(metadata or {}))
        con = self._connect()
        with con:
            con.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
                (
                    s.session_id,
                    s.user_id,
                    s.created_at,
                    s.updated_at,
                    json.dumps(s.state),
                    json.dumps(s.metadata),
                ),
            )
        con.close()
        return s

    def get(self, session_id: str) -> Session | None:
        con = self._connect()
        row = con.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            con.close()
            return None
        events = self._load_events(con, session_id)
        s = self._row_to_session(row, events)
        con.close()
        return s

    def append_events(self, session_id: str, events: list[Event]) -> None:
        if not events:
            return
        con = self._connect()
        with con:
            # Get current max position
            row = con.execute(
                "SELECT COALESCE(MAX(position), -1) AS pos FROM events WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            offset = row["pos"] + 1
            for i, evt in enumerate(events):
                con.execute(
                    "INSERT INTO events VALUES (?, ?, ?, ?)",
                    (evt.id, session_id, offset + i, evt.model_dump_json()),
                )
            con.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (time.time(), session_id),
            )
        con.close()

    def update_state(self, session_id: str, patch: dict[str, Any]) -> None:
        con = self._connect()
        row = con.execute(
            "SELECT state FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            con.close()
            raise KeyError(session_id)
        state = json.loads(row["state"])
        state.update(patch)
        with con:
            con.execute(
                "UPDATE sessions SET state = ?, updated_at = ? WHERE session_id = ?",
                (json.dumps(state), time.time(), session_id),
            )
        con.close()

    def list_sessions(self, user_id: str, limit: int = 50) -> list[Session]:
        con = self._connect()
        rows = con.execute(
            "SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        sessions = [self._row_to_session(r, []) for r in rows]
        con.close()
        return sessions

    def delete(self, session_id: str) -> bool:
        con = self._connect()
        with con:
            cur = con.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
        con.close()
        return cur.rowcount > 0

    def delete_user(self, user_id: str) -> int:
        con = self._connect()
        with con:
            cur = con.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        con.close()
        return cur.rowcount
