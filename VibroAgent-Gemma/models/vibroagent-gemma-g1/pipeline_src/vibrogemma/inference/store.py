from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import sqlite3
except ModuleNotFoundError as exc:
    if exc.name != "_sqlite3":
        raise
    import pysqlite3 as sqlite3  # type: ignore[no-redef]

from ..contracts import ClassificationResult
from ..utils import utc_now_iso


class EventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_utc TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    window_start_utc TEXT NOT NULL,
                    system_state TEXT NOT NULL,
                    alert_emitted INTEGER NOT NULL,
                    alert_reason TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    window_path TEXT,
                    UNIQUE(episode_id)
                );
                CREATE INDEX IF NOT EXISTS idx_events_created ON events(id DESC);
                CREATE INDEX IF NOT EXISTS idx_events_state ON events(system_state);
                """
            )

    def insert(
        self,
        result: ClassificationResult,
        *,
        alert_emitted: bool,
        alert_reason: str,
        window_path: str | None = None,
    ) -> int:
        payload = json.dumps(result.to_dict(), sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO events(
                    created_utc, episode_id, window_start_utc, system_state,
                    alert_emitted, alert_reason, result_json, window_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now_iso(),
                    str(result.window["episode_id"]),
                    str(result.window["start_utc"]),
                    result.system_state,
                    int(alert_emitted),
                    alert_reason,
                    payload,
                    window_path,
                ),
            )
            row = connection.execute(
                "SELECT id FROM events WHERE episode_id = ?", (str(result.window["episode_id"]),)
            ).fetchone()
            if row is None:
                raise RuntimeError("Failed to persist event")
            return int(row["id"])

    def latest(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM events ORDER BY id DESC LIMIT 1").fetchone()
        return self._row(row) if row else None

    def list(self, *, limit: int = 100, alerts_only: bool = False) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 10_000))
        query = "SELECT * FROM events"
        parameters: tuple[Any, ...] = ()
        if alerts_only:
            query += " WHERE alert_emitted = 1"
        query += " ORDER BY id DESC LIMIT ?"
        parameters = (limit,)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._row(row) for row in rows]

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "created_utc": row["created_utc"],
            "episode_id": row["episode_id"],
            "window_start_utc": row["window_start_utc"],
            "system_state": row["system_state"],
            "alert_emitted": bool(row["alert_emitted"]),
            "alert_reason": row["alert_reason"],
            "result": json.loads(row["result_json"]),
            "window_path": row["window_path"],
        }
