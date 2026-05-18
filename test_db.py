"""
TraFix — PostgreSQL Bağlantı Test Scripti
==========================================
Kullanım:
    python test_db.py

Ne yapar:
  1. .env dosyasını okur
  2. PostgreSQL'e bağlanır
  3. 'trafix' veritabanı yoksa oluşturur
  4. traffic_events tablosunu oluşturur (yoksa)
  5. Örnek bir kayıt ekler
  6. Kaydı geri okur ve gösterir
"""

import asyncio
import os
import sys
from pathlib import Path

# .env dosyasını yükle
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"[OK] .env dosyası yüklendi: {env_path}")
    else:
        print("[WARN] .env dosyası bulunamadı, ortam değişkenleri kullanılıyor.")
except ImportError:
    print("[WARN] python-dotenv yüklü değil. 'pip install python-dotenv' çalıştırın.")

import asyncpg

HOST     = os.environ.get("POSTGRES_HOST", "localhost")
PORT     = int(os.environ.get("POSTGRES_PORT", "5432"))
DB       = os.environ.get("POSTGRES_DB", "trafix")
USER     = os.environ.get("POSTGRES_USER", "postgres")
PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")

print(f"\n{'='*50}")
print(f"  Bağlantı bilgileri:")
print(f"  Host     : {HOST}:{PORT}")
print(f"  Veritabanı: {DB}")
print(f"  Kullanıcı: {USER}")
print(f"  Şifre    : {'(boş)' if not PASSWORD else '***'}")
print(f"{'='*50}\n")


async def ensure_database():
    """'trafix' veritabanı yoksa oluşturur."""
    try:
        conn = await asyncpg.connect(
            host=HOST, port=PORT,
            user=USER, password=PASSWORD,
            database="postgres",
        )
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", DB
        )
        if not exists:
            await conn.execute(f'CREATE DATABASE "{DB}"')
            print(f"[OK] '{DB}' veritabanı oluşturuldu.")
        else:
            print(f"[OK] '{DB}' veritabanı zaten mevcut.")
        await conn.close()
        return True
    except Exception as e:
        print(f"[HATA] postgres veritabanına bağlanılamadı: {e}")
        return False


async def test_db():
    # 1. Önce 'postgres' DB'ye bağlanıp trafix DB'yi oluştur
    if not await ensure_database():
        return False

    # 2. trafix veritabanına bağlan
    try:
        conn = await asyncpg.connect(
            host=HOST, port=PORT,
            user=USER, password=PASSWORD,
            database=DB,
        )
        print(f"[OK] '{DB}' veritabanına bağlantı başarılı!\n")
    except Exception as e:
        print(f"[HATA] '{DB}' veritabanına bağlanılamadı: {e}")
        return False

    # 3. Tabloyu oluştur (yoksa)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS traffic_events (
            id               BIGSERIAL PRIMARY KEY,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            step             INTEGER,
            intersection_id  INTEGER NOT NULL,
            north_count      INTEGER,
            south_count      INTEGER,
            east_count       INTEGER,
            west_count       INTEGER,
            queue_length     DOUBLE PRECISION,
            current_phase    INTEGER,
            phase_duration   DOUBLE PRECISION,
            next_phase       INTEGER,
            confidence       DOUBLE PRECISION,
            decision_source  VARCHAR(20) NOT NULL DEFAULT 'heuristic',
            reward           DOUBLE PRECISION,
            reward_breakdown JSONB
        );
        CREATE INDEX IF NOT EXISTS idx_te_intersection_id ON traffic_events (intersection_id);
        CREATE INDEX IF NOT EXISTS idx_te_created_at      ON traffic_events (created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_te_step            ON traffic_events (step);
    """)
    print("[OK] 'traffic_events' tablosu hazır.\n")

    # 4. Örnek kayıt ekle
    await conn.execute("""
        INSERT INTO traffic_events (
            step, intersection_id,
            north_count, south_count, east_count, west_count,
            queue_length, current_phase, phase_duration,
            next_phase, confidence, decision_source,
            reward, reward_breakdown
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
    """,
        1, 0,
        5, 3, 7, 2,
        12.5, 1, 15.0,
        2, 0.87, "ai",
        -0.312, '{"pressure": -0.15, "queue": -0.03, "throughput": 0.12}'
    )
    print("[OK] Örnek kayıt eklendi.\n")

    # 5. Kaydı geri oku ve göster
    row = await conn.fetchrow(
        "SELECT * FROM traffic_events ORDER BY created_at DESC LIMIT 1"
    )
    print("─── Son Kayıt ─────────────────────────────────────")
    for key, val in dict(row).items():
        print(f"  {key:<20}: {val}")
    print("─────────────────────────────────────────────────────\n")

    # 6. Toplam kayıt sayısı
    total = await conn.fetchval("SELECT COUNT(*) FROM traffic_events")
    print(f"[OK] Tablodaki toplam kayıt sayısı: {total}\n")

    await conn.close()
    print("✓ Tüm testler başarılı! PostgreSQL entegrasyonu çalışıyor.")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_db())
    sys.exit(0 if success else 1)
