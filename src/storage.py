import sqlite3
from threading import Lock
from typing import Optional


class Storage:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS replied_ids (
                    comment_id TEXT PRIMARY KEY,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def get_last_seen_id(self) -> Optional[str]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM state WHERE key = 'last_seen_id'"
            ).fetchone()
            return row[0] if row else None

    def set_last_seen_id(self, fullname: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO state(key, value) VALUES('last_seen_id', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (fullname,),
            )
            conn.commit()

    def has_replied(self, comment_id: str) -> bool:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM replied_ids WHERE comment_id = ?",
                (comment_id,),
            ).fetchone()
            return row is not None

    def mark_replied(self, comment_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO replied_ids(comment_id) VALUES(?)",
                (comment_id,),
            )
            conn.commit()
