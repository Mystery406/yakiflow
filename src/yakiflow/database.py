from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence

from .models import Cue, JobStatus, utc_now


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, input TEXT NOT NULL, status TEXT NOT NULL,
    config_json TEXT NOT NULL, error TEXT, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS media_artifacts (
    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, path TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cues (
    id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL UNIQUE, start REAL NOT NULL,
    end REAL NOT NULL, source TEXT NOT NULL, translated TEXT,
    timing_confidence REAL, metadata_json TEXT NOT NULL DEFAULT '{}',
    stable INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS translation_batches (
    id INTEGER PRIMARY KEY, cue_ids_json TEXT NOT NULL,
    backend TEXT NOT NULL, model TEXT, status TEXT NOT NULL,
    request_json TEXT, response_json TEXT, error TEXT,
    created_at TEXT NOT NULL, finished_at TEXT
);
CREATE TABLE IF NOT EXISTS checkpoints (
    name TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS process_logs (
    id INTEGER PRIMARY KEY, process TEXT NOT NULL, stream TEXT NOT NULL,
    line TEXT NOT NULL, created_at TEXT NOT NULL
);
"""


class JobDatabase:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)

    def close(self) -> None:
        self.connection.close()

    def create_job(self, job_id: str, input_value: str, config: dict[str, Any]) -> None:
        now = utc_now()
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO jobs VALUES (?, ?, ?, ?, NULL, ?, ?)",
                (job_id, input_value, JobStatus.CREATED, json.dumps(config, default=str), now, now),
            )

    def job(self) -> sqlite3.Row | None:
        return self.connection.execute("SELECT * FROM jobs ORDER BY created_at LIMIT 1").fetchone()

    def set_status(self, status: JobStatus | str, error: str | None = None) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE jobs SET status=?, error=?, updated_at=?",
                (str(status), error, utc_now()),
            )

    def add_artifact(self, kind: str, path: Path) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO media_artifacts(kind,path,created_at) VALUES(?,?,?)",
                (kind, str(path), utc_now()),
            )

    def artifact(self, kind: str) -> Path | None:
        row = self.connection.execute(
            "SELECT path FROM media_artifacts WHERE kind=? ORDER BY id DESC LIMIT 1", (kind,)
        ).fetchone()
        return Path(row[0]) if row else None

    def upsert_cues(self, cues: Sequence[Cue], *, stable: bool = True, reset_order: bool = False) -> None:
        sql = """
        INSERT INTO cues(id,ordinal,start,end,source,translated,
          timing_confidence,metadata_json,stable)
        VALUES(?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
          ordinal=excluded.ordinal,start=excluded.start,end=excluded.end,
          source=excluded.source,
          translated=COALESCE(excluded.translated,cues.translated),
          timing_confidence=excluded.timing_confidence,metadata_json=excluded.metadata_json,
          stable=excluded.stable
        """
        existing = {
            row["id"]: row["ordinal"]
            for row in self.connection.execute("SELECT id,ordinal FROM cues").fetchall()
        }
        next_ordinal = max(existing.values(), default=-1) + 1
        ordinals = [
            i if reset_order else existing.get(cue.id, next_ordinal + i)
            for i, cue in enumerate(cues)
        ]
        with self.connection:
            self.connection.executemany(
                sql,
                [
                    (
                        cue.id,
                        i,
                        cue.start,
                        cue.end,
                        cue.source,
                        cue.translated,
                        cue.timing_confidence,
                        json.dumps(cue.metadata),
                        stable,
                    )
                    for i, cue in zip(ordinals, cues, strict=True)
                ],
            )

    def replace_transcript(self, cues: Sequence[Cue]) -> None:
        """Replace provisional cues with an authoritative transcript."""
        # A draft Agent can finish while Whisper is still producing its final
        # JSON. Preserve that completed source correction and translation while
        # accepting Whisper's authoritative timing and metadata.
        existing = {
            row["id"]: row
            for row in self.connection.execute(
                "SELECT id,source,translated FROM cues"
            ).fetchall()
        }
        merged: list[Cue] = []
        for cue in cues:
            prior = existing.get(cue.id)
            completed = prior is not None and prior["translated"] is not None
            metadata = dict(cue.metadata)
            merged.append(Cue(
                cue.id,
                cue.start,
                cue.end,
                prior["source"] if completed else cue.source,
                prior["translated"] if completed else cue.translated,
                cue.timing_confidence,
                metadata,
            ))
        with self.connection:
            self.connection.execute("UPDATE cues SET stable=0, ordinal=ordinal+1000000")
        self.upsert_cues(merged, stable=True, reset_order=True)

    def discard_provisional_transcript(self) -> None:
        """Hide live cues and their translations before the authoritative pass."""
        with self.connection:
            self.connection.execute(
                "UPDATE cues SET translated=NULL, stable=0"
            )

    def replace_aligned_timeline(self, cues: Sequence[Cue]) -> None:
        """Atomically install a renumbered timeline and mark alignment durable."""
        sql = """
        INSERT INTO cues(id,ordinal,start,end,source,translated,
          timing_confidence,metadata_json,stable)
        VALUES(?,?,?,?,?,?,?,?,1)
        """
        now = utc_now()
        with self.connection:
            self.connection.execute("DELETE FROM cues")
            self.connection.executemany(
                sql,
                [
                    (
                        cue.id,
                        ordinal,
                        cue.start,
                        cue.end,
                        cue.source,
                        cue.translated,
                        cue.timing_confidence,
                        json.dumps(cue.metadata),
                    )
                    for ordinal, cue in enumerate(cues)
                ],
            )
            self.connection.execute(
                "INSERT INTO checkpoints VALUES(?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET "
                "value_json=excluded.value_json,updated_at=excluded.updated_at",
                ("alignment_complete", json.dumps(True), now),
            )

    def list_cues(self, *, stable_only: bool = False) -> list[Cue]:
        where = " WHERE stable=1" if stable_only else ""
        rows = self.connection.execute(f"SELECT * FROM cues{where} ORDER BY ordinal").fetchall()
        return [
            Cue(
                id=row["id"], start=row["start"], end=row["end"], source=row["source"],
                translated=row["translated"],
                timing_confidence=row["timing_confidence"], metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]

    def checkpoint(self, name: str, value: Any) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO checkpoints VALUES(?,?,?) ON CONFLICT(name) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at",
                (name, json.dumps(value), utc_now()),
            )

    def get_checkpoint(self, name: str, default: Any = None) -> Any:
        row = self.connection.execute("SELECT value_json FROM checkpoints WHERE name=?", (name,)).fetchone()
        return json.loads(row[0]) if row else default

    def log(self, process: str, stream: str, line: str) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO process_logs(process,stream,line,created_at) VALUES(?,?,?,?)",
                (process, stream, line.rstrip(), utc_now()),
            )

    def start_batch(self, cue_ids: Sequence[str], backend: str, model: str | None, request: Any) -> int:
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO translation_batches(cue_ids_json,backend,model,status,request_json,created_at) VALUES(?,?,?,?,?,?)",
                (json.dumps(cue_ids), backend, model, "running", json.dumps(request), utc_now()),
            )
        return int(cursor.lastrowid)

    def finish_batch(self, batch_id: int, response: Any = None, error: str | None = None) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE translation_batches SET status=?,response_json=?,error=?,finished_at=? WHERE id=?",
                ("failed" if error else "complete", json.dumps(response) if response is not None else None,
                 error, utc_now(), batch_id),
            )
