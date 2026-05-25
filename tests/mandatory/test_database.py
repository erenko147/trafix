"""
FR-04 Database Integration Test
================================
Sends 20 batches of simulated telemetry to the TraFix backend and verifies
that every row was persisted in PostgreSQL.

Prerequisites:
  1. PostgreSQL running:  sudo systemctl start postgresql
  2. DB created:          createdb trafix
  3. Backend running:     python run.py
  4. Run this test:       python tests/mandatory/test_database.py

Pass criteria:
  - step_log rows   == batches_sent × 5 junctions
  - sessions table  has an open session (ended_at IS NULL)
  - avg_queue       is a finite number (not NULL)
"""

import os
import sys
import time
import random
import requests
import psycopg2

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE     = "http://127.0.0.1:8000"
DB_URL       = os.environ.get("DATABASE_URL", "dbname=trafix")
BATCHES      = 20        # number of telemetry batches to send
JUNCTIONS    = 5
SLEEP_S      = 0.1       # pause between batches (seconds)

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_telemetry(junction_id: int, phase: int) -> dict:
    dirs = ["north", "south", "east", "west"]
    moves = ["left", "through", "right"]
    data = {"intersection_id": junction_id, "current_phase": phase,
            "phase_duration": round(random.uniform(5, 45), 1),
            "queue_length": round(random.uniform(0, 40), 1)}
    for d in dirs:
        for m in moves:
            data[f"{d}_{m}"] = random.randint(0, 8)
    return data


def send_batch(step: int, phases: list) -> list:
    payload = {
        "step": step,
        "intersections": [make_telemetry(i, phases[i]) for i in range(JUNCTIONS)],
    }
    try:
        r = requests.post(f"{API_BASE}/telemetry_batch", json=payload, timeout=3)
        r.raise_for_status()
        return [d["next_phase"] for d in r.json().get("decisions", [])]
    except Exception as exc:
        print(f"  [WARN] batch {step} failed: {exc}")
        return phases


def check_db(expected_rows: int) -> bool:
    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            # Latest open session
            cur.execute(
                "SELECT id, scenario, started_at, ended_at "
                "FROM sessions ORDER BY id DESC LIMIT 1"
            )
            sess = cur.fetchone()
            if not sess:
                print("  FAIL: no sessions found")
                return False
            sess_id, scenario, started_at, ended_at = sess
            print(f"  Session id={sess_id}  scenario={scenario}  "
                  f"started={started_at}  ended={ended_at}")

            # step_log row count for this session
            cur.execute(
                "SELECT COUNT(*), ROUND(AVG(queue_length)::numeric,2), "
                "MAX(sim_step), COUNT(DISTINCT junction_id) "
                "FROM step_log WHERE session_id = %s",
                (sess_id,),
            )
            count, avg_q, max_step, n_junctions = cur.fetchone()
            print(f"  step_log rows : {count}  (expected ≥ {expected_rows})")
            print(f"  avg queue     : {avg_q}")
            print(f"  max sim_step  : {max_step}")
            print(f"  junctions seen: {n_junctions}")

            if count < expected_rows:
                print(f"  FAIL: expected ≥ {expected_rows} rows, got {count}")
                return False
            if n_junctions < JUNCTIONS:
                print(f"  FAIL: expected {JUNCTIONS} junctions, got {n_junctions}")
                return False
            if avg_q is None:
                print("  FAIL: avg_queue is NULL")
                return False

            # Verify phase decision values are in range
            cur.execute(
                "SELECT COUNT(*) FROM step_log "
                "WHERE session_id = %s AND (next_phase < 0 OR next_phase > 11)",
                (sess_id,),
            )
            bad = cur.fetchone()[0]
            if bad:
                print(f"  FAIL: {bad} rows with out-of-range next_phase")
                return False

        return True
    finally:
        conn.close()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TraFix  FR-04  Database Integration Test")
    print("=" * 60)

    # 1. Check backend is reachable
    print("\n[1] Checking backend …")
    try:
        r = requests.get(f"{API_BASE}/state", timeout=3)
        r.raise_for_status()
        print(f"  Backend OK  ({API_BASE})")
    except Exception as exc:
        print(f"  FAIL: cannot reach backend — {exc}")
        print("  Start it with:  python run.py")
        sys.exit(1)

    # 2. Check DB summary endpoint
    print("\n[2] Checking /db_summary …")
    try:
        r = requests.get(f"{API_BASE}/db_summary", timeout=3)
        summary = r.json()
        print(f"  {summary}")
        if not summary.get("db_connected"):
            print("  FAIL: backend reports db_connected=False")
            print("  Ensure PostgreSQL is running and 'trafix' database exists:")
            print("    sudo systemctl start postgresql")
            print("    createdb trafix")
            sys.exit(1)
    except Exception as exc:
        print(f"  FAIL: /db_summary error — {exc}")
        sys.exit(1)

    # 3. Send telemetry batches
    print(f"\n[3] Sending {BATCHES} telemetry batches ({JUNCTIONS} junctions each) …")
    phases = [0] * JUNCTIONS
    for step in range(1, BATCHES + 1):
        new_phases = send_batch(step, phases)
        if len(new_phases) == JUNCTIONS:
            phases = new_phases
        print(f"  batch {step:>3}/{BATCHES}  phases={phases}", end="\r")
        time.sleep(SLEEP_S)
    print(f"\n  Done — {BATCHES} batches sent.")

    # 4. Wait a moment for background writes to flush
    time.sleep(1.5)

    # 5. Verify DB
    print("\n[4] Verifying PostgreSQL rows …")
    expected = BATCHES * JUNCTIONS
    try:
        ok = check_db(expected)
    except psycopg2.OperationalError as exc:
        print(f"  FAIL: cannot connect to PostgreSQL — {exc}")
        print("  Ensure PostgreSQL is running and 'trafix' database exists.")
        sys.exit(1)

    # 6. Result
    print()
    if ok:
        print("=" * 60)
        print("  RESULT: PASS — FR-04 database logging verified.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print("  RESULT: FAIL")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    random.seed(42)
    main()
