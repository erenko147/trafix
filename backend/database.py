"""
TraFix — PostgreSQL persistence layer (FR-04)
=============================================
Logs telemetry batches, AI phase decisions, and emergency events to PostgreSQL.

Environment variable:
  DATABASE_URL  postgresql://user:pass@host/dbname
                default: postgresql://localhost/trafix

Quick setup (run once):
  createdb trafix
  python -c "from backend.database import init_db; init_db()"
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional

import psycopg2
from psycopg2 import pool

log = logging.getLogger("trafix.db")

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL", "dbname=trafix"
)

_pool: Optional[pool.ThreadedConnectionPool] = None

# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    id         SERIAL PRIMARY KEY,
    scenario   VARCHAR(64)  NOT NULL DEFAULT 'live',
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS step_log (
    id            BIGSERIAL PRIMARY KEY,
    session_id    INTEGER     NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    sim_step      INTEGER     NOT NULL,
    logged_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    junction_id   SMALLINT    NOT NULL,
    queue_length  REAL        NOT NULL,
    current_phase SMALLINT    NOT NULL,
    next_phase    SMALLINT    NOT NULL,
    confidence    REAL        NOT NULL,
    total_vehicles SMALLINT   NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_step_log_session
    ON step_log (session_id, sim_step);

CREATE TABLE IF NOT EXISTS emergency_log (
    id              SERIAL PRIMARY KEY,
    session_id      INTEGER     NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    tls_id          VARCHAR(32),
    junction_name   VARCHAR(16),
    ambulance_id    VARCHAR(32),
    start_step      INTEGER,
    end_step        INTEGER,
    transit_steps   INTEGER,
    vehicles_waited INTEGER,
    total_wait_steps INTEGER,
    avg_wait_steps  REAL,
    result          VARCHAR(16),
    logged_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# ── Connection pool ───────────────────────────────────────────────────────────

def init_db(url: str = DATABASE_URL) -> bool:
    """
    Open connection pool and create tables.
    Returns True on success, False if PostgreSQL is unreachable.
    """
    global _pool
    try:
        _pool = pool.ThreadedConnectionPool(
            minconn=1, maxconn=5, dsn=url
        )
        _exec_ddl()
        log.info("[DB] Connected to PostgreSQL and tables verified.")
        return True
    except Exception as exc:
        log.warning(f"[DB] PostgreSQL unavailable — data will NOT be persisted. ({exc})")
        _pool = None
        return False


def _exec_ddl():
    conn = _pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(_DDL)
    finally:
        _pool.putconn(conn)


def _conn():
    if _pool is None:
        return None
    return _pool.getconn()


def _put(conn):
    if _pool and conn:
        _pool.putconn(conn)


# ── Public API ────────────────────────────────────────────────────────────────

def create_session(scenario: str = "live") -> Optional[int]:
    """Insert a session row and return its id."""
    conn = _conn()
    if conn is None:
        return None
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO sessions (scenario) VALUES (%s) RETURNING id",
                    (scenario,),
                )
                return cur.fetchone()[0]
    except Exception as exc:
        log.warning(f"[DB] create_session failed: {exc}")
        return None
    finally:
        _put(conn)


def close_session(session_id: int):
    """Stamp ended_at on a session row."""
    if session_id is None:
        return
    conn = _conn()
    if conn is None:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET ended_at = NOW() WHERE id = %s",
                    (session_id,),
                )
    except Exception as exc:
        log.warning(f"[DB] close_session failed: {exc}")
    finally:
        _put(conn)


def log_step(
    session_id: int,
    sim_step: int,
    decisions: list,
    telemetry: list,
):
    """
    Persist one telemetry batch.

    decisions  — list of dicts from /telemetry_batch response
    telemetry  — list of Telemetry dicts (already .dict() expanded)
    """
    if session_id is None or _pool is None:
        return

    # Build a lookup keyed by junction_id for O(1) merging
    tele_by_id = {t["intersection_id"]: t for t in telemetry}

    rows = []
    for dec in decisions:
        jid = dec["intersection_id"]
        t = tele_by_id.get(jid, {})
        rows.append((
            session_id,
            sim_step,
            jid,
            float(dec.get("queue_length", t.get("queue_length", 0.0))),
            int(t.get("current_phase", dec.get("next_phase", 0))),
            int(dec["next_phase"]),
            float(dec.get("confidence", 0.0)),
            int(dec.get("total_vehicles", 0)),
        ))

    if not rows:
        return

    conn = _conn()
    if conn is None:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO step_log
                        (session_id, sim_step, junction_id,
                         queue_length, current_phase, next_phase,
                         confidence, total_vehicles)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    rows,
                )
    except Exception as exc:
        log.warning(f"[DB] log_step failed: {exc}")
    finally:
        _put(conn)


def log_emergency(session_id: int, event: dict):
    """Persist one completed emergency preemption event."""
    if session_id is None or _pool is None:
        return
    conn = _conn()
    if conn is None:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO emergency_log
                        (session_id, tls_id, junction_name, ambulance_id,
                         start_step, end_step, transit_steps,
                         vehicles_waited, total_wait_steps, avg_wait_steps, result)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        session_id,
                        event.get("tls_id"),
                        event.get("junction_name"),
                        event.get("ambulance_id"),
                        event.get("start_step"),
                        event.get("end_step"),
                        event.get("transit_steps"),
                        event.get("vehicles_waited"),
                        event.get("total_wait_steps"),
                        event.get("avg_wait_steps"),
                        event.get("result"),
                    ),
                )
    except Exception as exc:
        log.warning(f"[DB] log_emergency failed: {exc}")
    finally:
        _put(conn)


# ── Read helpers (for dashboard / test verification) ──────────────────────────

def query_session_summary(session_id: int) -> dict:
    """Return row counts and basic stats for a session."""
    conn = _conn()
    if conn is None:
        return {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*)                        AS step_rows,
                    COUNT(DISTINCT sim_step)        AS steps,
                    COUNT(DISTINCT junction_id)     AS junctions,
                    ROUND(AVG(queue_length)::numeric, 2) AS avg_queue,
                    MAX(sim_step)                   AS last_step
                FROM step_log
                WHERE session_id = %s
                """,
                (session_id,),
            )
            row = cur.fetchone()
            cur.execute(
                "SELECT COUNT(*) FROM emergency_log WHERE session_id = %s",
                (session_id,),
            )
            emerg = cur.fetchone()[0]
        return {
            "step_rows": row[0],
            "steps": row[1],
            "junctions": row[2],
            "avg_queue": float(row[3] or 0),
            "last_step": row[4],
            "emergency_events": emerg,
        }
    except Exception as exc:
        log.warning(f"[DB] query_session_summary failed: {exc}")
        return {}
    finally:
        _put(conn)
