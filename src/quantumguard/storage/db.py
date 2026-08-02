from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from quantumguard.config import load_config

SCHEMA = """
CREATE TABLE IF NOT EXISTS health_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_id INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    cds REAL NOT NULL,
    qds REAL NOT NULL,
    pc REAL NOT NULL,
    at REAL NOT NULL,
    mhs REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('healthy', 'warning', 'critical'))
);

CREATE TABLE IF NOT EXISTS drift_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_id INTEGER NOT NULL,
    detector_name TEXT NOT NULL,
    subset TEXT NOT NULL CHECK (subset IN ('population', 'fraud')),
    score REAL NOT NULL,
    wall_clock_ms REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS feature_drift (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_id INTEGER NOT NULL,
    feature TEXT NOT NULL,
    classical_drift REAL NOT NULL,
    quantum_drift REAL NOT NULL,
    hybrid_drift REAL NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('ok', 'warn', 'crit')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS trend_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_id INTEGER NOT NULL,
    slope REAL NOT NULL,
    p_value REAL NOT NULL,
    cusum_stat REAL NOT NULL,
    triggered INTEGER NOT NULL CHECK (triggered IN (0, 1)),
    retrain_window_width INTEGER,
    series_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS retrain_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    trend_decision_id INTEGER REFERENCES trend_decisions(id),
    offline_metrics_json TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    decided_at TEXT,
    decided_by TEXT
);

CREATE TABLE IF NOT EXISTS shadow_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL REFERENCES retrain_candidates(id),
    window_id INTEGER NOT NULL,
    agreement_rate REAL NOT NULL,
    divergence_cases_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _resolve_db_path(db_path: str | Path | None) -> Path:
    if db_path is not None:
        return Path(db_path)
    return load_config().path("db_path")


def init_db(db_path: str | Path | None = None) -> Path:
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)
    return path


@contextmanager
def get_connection(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    path = _resolve_db_path(db_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_health_score(
    conn: sqlite3.Connection,
    *,
    window_id: int,
    cds: float,
    qds: float,
    pc: float,
    at: float,
    mhs: float,
    status: str,
) -> int:
    cur = conn.execute(
        "INSERT INTO health_scores (window_id, cds, qds, pc, at, mhs, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (window_id, cds, qds, pc, at, mhs, status),
    )
    return cur.lastrowid


def insert_drift_score(
    conn: sqlite3.Connection,
    *,
    window_id: int,
    detector_name: str,
    subset: str,
    score: float,
    wall_clock_ms: float | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO drift_scores (window_id, detector_name, subset, score, wall_clock_ms) "
        "VALUES (?, ?, ?, ?, ?)",
        (window_id, detector_name, subset, score, wall_clock_ms),
    )
    return cur.lastrowid


def insert_feature_drift(
    conn: sqlite3.Connection,
    *,
    window_id: int,
    rows: list[dict[str, Any]],
) -> None:
    conn.executemany(
        "INSERT INTO feature_drift "
        "(window_id, feature, classical_drift, quantum_drift, hybrid_drift, severity) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                window_id,
                row["feature"],
                row["classical_drift"],
                row["quantum_drift"],
                row["hybrid_drift"],
                row["severity"],
            )
            for row in rows
        ],
    )


def get_latest_feature_drift(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    # Return ONLY the most recently inserted window's batch. Each window's rows
    # are inserted consecutively, so the last batch is the contiguous run of
    # highest ids that all share the final window_id. Keying off id (not
    # MAX(window_id)) means a fresh run whose window ids restart at 0 is not
    # masked by, or merged with, stale rows from a previous run that reused the
    # same window_id number.
    return conn.execute(
        "SELECT * FROM feature_drift WHERE id > ("
        "  SELECT COALESCE(MAX(id), 0) FROM feature_drift"
        "  WHERE window_id <> (SELECT window_id FROM feature_drift ORDER BY id DESC LIMIT 1)"
        ") ORDER BY hybrid_drift DESC"
    ).fetchall()


def insert_trend_decision(
    conn: sqlite3.Connection,
    *,
    window_id: int,
    slope: float,
    p_value: float,
    cusum_stat: float,
    triggered: bool,
    retrain_window_width: int | None,
    series: list[float],
) -> int:
    cur = conn.execute(
        "INSERT INTO trend_decisions "
        "(window_id, slope, p_value, cusum_stat, triggered, retrain_window_width, series_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            window_id,
            slope,
            p_value,
            cusum_stat,
            int(triggered),
            retrain_window_width,
            json.dumps(series),
        ),
    )
    return cur.lastrowid


def insert_retrain_candidate(
    conn: sqlite3.Connection,
    *,
    model_version: str,
    trend_decision_id: int | None,
    offline_metrics: dict[str, Any] | None,
) -> int:
    cur = conn.execute(
        "INSERT INTO retrain_candidates (model_version, trend_decision_id, offline_metrics_json) "
        "VALUES (?, ?, ?)",
        (
            model_version,
            trend_decision_id,
            json.dumps(offline_metrics) if offline_metrics is not None else None,
        ),
    )
    return cur.lastrowid


def insert_shadow_log(
    conn: sqlite3.Connection,
    *,
    candidate_id: int,
    window_id: int,
    agreement_rate: float,
    divergence_cases: list[dict[str, Any]] | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO shadow_logs (candidate_id, window_id, agreement_rate, divergence_cases_json) "
        "VALUES (?, ?, ?, ?)",
        (
            candidate_id,
            window_id,
            agreement_rate,
            json.dumps(divergence_cases) if divergence_cases is not None else None,
        ),
    )
    return cur.lastrowid


def get_shadow_logs(conn: sqlite3.Connection, candidate_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM shadow_logs WHERE candidate_id = ? ORDER BY window_id", (candidate_id,)
    ).fetchall()


def decide_retrain_candidate(
    conn: sqlite3.Connection,
    *,
    candidate_id: int,
    status: str,
    decided_by: str,
) -> None:
    if status not in ("approved", "rejected"):
        raise ValueError(f"invalid decision status: {status}")
    conn.execute(
        "UPDATE retrain_candidates SET status = ?, decided_at = datetime('now'), decided_by = ? "
        "WHERE id = ?",
        (status, decided_by, candidate_id),
    )


def get_pending_candidates(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM retrain_candidates WHERE status = 'pending' ORDER BY created_at"
    ).fetchall()


def get_recent_health_scores(conn: sqlite3.Connection, limit: int = 100) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM health_scores ORDER BY window_id DESC LIMIT ?", (limit,)
    ).fetchall()
