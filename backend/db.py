"""
TraFix — PostgreSQL Veritabanı Modülü
======================================
Kavşak ışık durumu, agent kararı ve reward bilgilerini kaydeder.

Ortam değişkenleri (tümü isteğe bağlı, varsayılanlar aşağıda):
  POSTGRES_HOST      localhost
  POSTGRES_PORT      5432
  POSTGRES_DB        trafix
  POSTGRES_USER      postgres
  POSTGRES_PASSWORD  (boş)
"""

import os
import json
import logging
import math
from typing import Any, Dict, List, Optional
from pathlib import Path

import asyncpg

# .env dosyasını otomatik yükle (varsa)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=True)
except ImportError:
    pass

logger = logging.getLogger("trafix.db")

# ─── Bağlantı Parametreleri ────────────────────────────────────────────────
_DB_CONFIG: Dict[str, Any] = {
    "host":     os.environ.get("POSTGRES_HOST", "localhost"),
    "port":     int(os.environ.get("POSTGRES_PORT", "5432")),
    "database": os.environ.get("POSTGRES_DB", "trafix"),
    "user":     os.environ.get("POSTGRES_USER", "postgres"),
    "password": os.environ.get("POSTGRES_PASSWORD", ""),
}

_pool: Optional[asyncpg.Pool] = None

# ─── DDL ───────────────────────────────────────────────────────────────────
_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS traffic_events (
    id               BIGSERIAL PRIMARY KEY,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Simülasyon adımı
    step             INTEGER,

    -- Kavşak no
    intersection_id  INTEGER NOT NULL,

    -- Işık durumu (gelen telemetri)
    north_count      INTEGER,
    south_count      INTEGER,
    east_count       INTEGER,
    west_count       INTEGER,
    queue_length     DOUBLE PRECISION,
    current_phase    INTEGER,
    phase_duration   DOUBLE PRECISION,

    -- Agent kararı
    next_phase       INTEGER,
    confidence       DOUBLE PRECISION,
    decision_source  VARCHAR(20) NOT NULL DEFAULT 'heuristic',

    -- Reward
    reward           DOUBLE PRECISION,
    reward_breakdown JSONB
);

CREATE INDEX IF NOT EXISTS idx_te_intersection_id ON traffic_events (intersection_id);
CREATE INDEX IF NOT EXISTS idx_te_created_at      ON traffic_events (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_te_step            ON traffic_events (step);
"""

_INSERT_SQL = """
INSERT INTO traffic_events (
    step, intersection_id,
    north_count, south_count, east_count, west_count,
    queue_length, current_phase, phase_duration,
    next_phase, confidence, decision_source,
    reward, reward_breakdown
) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
"""


# ─── Bağlantı Yönetimi ─────────────────────────────────────────────────────

async def init_db() -> bool:
    """
    asyncpg bağlantı havuzunu başlatır ve tabloları oluşturur.
    Bağlantı kurulamazsa False döner; uygulama DB olmadan çalışmaya devam eder.
    """
    global _pool
    cfg = {k: v for k, v in _DB_CONFIG.items() if v != "" or k != "password"}
    try:
        _pool = await asyncpg.create_pool(**cfg, min_size=2, max_size=10)
        async with _pool.acquire() as conn:
            await conn.execute(_CREATE_TABLES_SQL)
        logger.info(
            "PostgreSQL bağlantısı kuruldu: %(user)s@%(host)s:%(port)s/%(database)s",
            _DB_CONFIG,
        )
        print(
            f"[DB] PostgreSQL bağlandı → "
            f"{_DB_CONFIG['user']}@{_DB_CONFIG['host']}:{_DB_CONFIG['port']}"
            f"/{_DB_CONFIG['database']}"
        )
        return True
    except Exception as exc:
        logger.warning("PostgreSQL bağlanamadı: %s — DB kaydı devre dışı.", exc)
        print(f"[DB] PostgreSQL bağlantısı başarısız: {exc}")
        print("[DB] Uygulama DB olmadan çalışmaya devam edecek.")
        _pool = None
        return False


async def close_db() -> None:
    """Bağlantı havuzunu kapatır."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def is_connected() -> bool:
    return _pool is not None


# ─── Reward Hesaplama ──────────────────────────────────────────────────────

def compute_reward_inline(
    current_obs: List[Dict],
    previous_obs: Optional[List[Dict]],
    current_actions: List[int],
    previous_actions: Optional[List[int]],
) -> Dict[int, Dict]:
    """
    Inference sırasında her kavşak için ödül bileşenlerini hesaplar.
    trafix_v2.compute_reward ile aynı formülü kullanır.

    Döner: { intersection_id: {"reward": float, "breakdown": {...}} }
    """
    results: Dict[int, Dict] = {}

    cur = sorted(current_obs, key=lambda d: d["intersection_id"])
    prev = sorted(previous_obs, key=lambda d: d["intersection_id"]) if previous_obs else None

    # Reward ağırlıkları (trafix_v2.RewardWeights varsayılanlarıyla aynı)
    W_PRESSURE      = -0.30
    W_QUEUE         = -0.25
    W_THROUGHPUT    =  0.25
    W_FAIRNESS      = -0.10
    W_PHASE_PENALTY = -0.08
    W_WAIT_PENALTY  = -0.05
    W_GREEN_WAVE    =  0.20

    def total_vehicles(o: Dict) -> int:
        return o["north_count"] + o["south_count"] + o["east_count"] + o["west_count"]

    rewards_raw: List[float] = []
    breakdowns: List[Dict] = []

    for i, o in enumerate(cur):
        pressure   = total_vehicles(o) / 40.0
        queue      = o["queue_length"] / 100.0

        throughput = 0.0
        if prev is not None:
            prev_total = total_vehicles(prev[i])
            cur_total  = total_vehicles(o)
            throughput = (prev_total - cur_total) / max(prev_total, 1.0)
            throughput = max(throughput, -1.0)

        counts = [o["north_count"], o["south_count"], o["east_count"], o["west_count"]]
        mean_c = sum(counts) / 4.0
        var_c  = sum((c - mean_c) ** 2 for c in counts) / 4.0
        fairness = math.sqrt(var_c) / max(mean_c, 1.0)

        phase_change = 0.0
        if previous_actions is not None and i < len(previous_actions):
            phase_change = float(current_actions[i] != previous_actions[i])

        wait = 0.0
        if o["phase_duration"] > 60.0:
            wait = (o["phase_duration"] - 60.0) / 60.0

        r = (
            W_PRESSURE      * pressure
            + W_QUEUE       * queue
            + W_THROUGHPUT  * throughput
            + W_FAIRNESS    * fairness
            + W_PHASE_PENALTY * phase_change
            + W_WAIT_PENALTY  * wait
        )
        rewards_raw.append(r)
        breakdowns.append({
            "pressure":      round(W_PRESSURE      * pressure,    4),
            "queue":         round(W_QUEUE         * queue,       4),
            "throughput":    round(W_THROUGHPUT    * throughput,  4),
            "fairness":      round(W_FAIRNESS      * fairness,    4),
            "phase_penalty": round(W_PHASE_PENALTY * phase_change,4),
            "wait_penalty":  round(W_WAIT_PENALTY  * wait,        4),
        })

    # Green-wave bonusu (kavşaklar arası koordinasyon)
    GREEN_WAVE_EDGES = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)]
    PLATOON_THRESHOLD = 5
    gw_score = 0.0
    for src_id, dst_id in GREEN_WAVE_EDGES:
        if src_id >= len(cur) or dst_id >= len(cur):
            continue
        src = cur[src_id]
        dst = cur[dst_id]
        src_total    = total_vehicles(src)
        src_phase    = src["current_phase"]
        dst_phase    = dst["current_phase"]
        src_is_green = src_phase in (0, 2)
        dst_aligned  = dst_phase in (0, 2) and (dst_phase % 2 == src_phase % 2)
        if src_total >= PLATOON_THRESHOLD and src_is_green and dst_aligned:
            base      = min(src_total / PLATOON_THRESHOLD, 2.0)
            gw_score += base * 0.5

    gw_per_node = W_GREEN_WAVE * gw_score / max(len(cur), 1)

    for i, o in enumerate(cur):
        iid = o["intersection_id"]
        final_reward = rewards_raw[i] + gw_per_node
        breakdowns[i]["green_wave"] = round(gw_per_node, 4)
        results[iid] = {
            "reward":    round(final_reward, 5),
            "breakdown": breakdowns[i],
        }

    return results


# ─── Kayıt Fonksiyonu ──────────────────────────────────────────────────────

async def insert_traffic_events(
    step: int,
    decisions: List[Dict[str, Any]],
    telemetry_map: Dict[str, Dict],
    decision_source: str,
    reward_map: Optional[Dict[int, Dict]] = None,
) -> None:
    """
    Her kavşak için bir satır yazar.

    Args:
        step            : Simülasyon adımı
        decisions       : /telemetry_batch yanıtındaki decisions listesi
        telemetry_map   : state_dict (intersection_id → telemetri)
        decision_source : 'ai' | 'heuristic'
        reward_map      : compute_reward_inline() çıktısı
                          { intersection_id: {"reward": float, "breakdown": {...}} }
    """
    if _pool is None:
        return

    rows = []
    for d in decisions:
        iid = d["intersection_id"]
        tel = telemetry_map.get(str(iid), {})
        rw  = (reward_map or {}).get(iid, {})

        breakdown_json = json.dumps(rw.get("breakdown")) if rw.get("breakdown") else None

        rows.append((
            step,
            iid,
            tel.get("north_count"),
            tel.get("south_count"),
            tel.get("east_count"),
            tel.get("west_count"),
            tel.get("queue_length"),
            tel.get("current_phase"),
            tel.get("phase_duration"),
            d.get("next_phase"),
            d.get("confidence"),
            decision_source,
            rw.get("reward"),
            breakdown_json,
        ))

    try:
        async with _pool.acquire() as conn:
            await conn.executemany(_INSERT_SQL, rows)
    except Exception as exc:
        logger.error("DB insert hatası: %s", exc)
