"""
TraFix SUMO-CARLA Co-Simulation Bridge
=======================================
SUMO araclarini ve trafik isikalarini gercek zamanli olarak CARLA'ya yansitir.
FastAPI AI backend'e her API_EVERY adimda telemetri gonderir, kararlari SUMO'ya uygular.

Kullanim:
    python sumo/carla_bridge/run_synchronization.py
    python sumo/carla_bridge/run_synchronization.py --sumo-gui
    python sumo/carla_bridge/run_synchronization.py --opendrive-file sumo/map.xodr
    python sumo/carla_bridge/run_synchronization.py --opendrive-file sumo/map.xodr --sumo-gui
    python sumo/carla_bridge/run_synchronization.py --carla-host 192.168.1.10 --step-length 0.05
    python sumo/carla_bridge/run_synchronization.py --sumo-cfg C:/harita/baska.sumocfg
"""

import argparse
import os
import sys
import time
import traceback

import requests

HERE         = os.path.dirname(os.path.abspath(__file__))
SUMO_DIR     = os.path.dirname(HERE)          # trafix/sumo/
PROJECT_ROOT = os.path.dirname(SUMO_DIR)      # trafix/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── CARLA import kontrolu ────────────────────────────────────────────────────
try:
    import carla  # noqa: F401
except ModuleNotFoundError:
    py = sys.version_info
    print(
        f"\n[HATA] 'carla' modulu bulunamadi!\n"
        f"\n"
        f"Mevcut Python: {py.major}.{py.minor}.{py.micro}\n"
        f"CARLA 0.9.16 gerektiriyor: Python 3.12\n"
        f"\n"
        f"Cozum — kurulum_python312.bat dosyasini calistir:\n"
        f"  {os.path.join(PROJECT_ROOT, 'kurulum_python312.bat')}\n"
        f"\n"
        f"Adimlar:\n"
        f"  1. https://www.python.org/downloads/release/python-3129/ adresinden\n"
        f"     Python 3.12'yi indirip kur (PATH secenegini isaretle)\n"
        f"  2. kurulum_python312.bat dosyasini cift tikla\n"
        f"  3. Kurulum tamamlaninca: python baslat.py --sumo-gui\n"
    )
    sys.exit(1)

from sumo.carla_bridge.sumo_simulation import SumoSimulation
from sumo.carla_bridge.carla_simulation import CarlaSimulation

# ── Sabitler ────────────────────────────────────────────────────────────────
DEFAULT_SUMO_CFG   = os.path.join(SUMO_DIR, "demo.sumocfg")
DEFAULT_XODR       = os.path.join(SUMO_DIR, "map.xodr")
DEFAULT_STEP       = 0.05
FASTAPI_URL        = "http://127.0.0.1:8001/telemetry_batch"
API_EVERY          = 5
MIN_GREEN_STEPS    = 10
MAX_EMPTY_STEPS    = 200
LOG_EVERY          = 200


# ── TLS yon haritasi ─────────────────────────────────────────────────────────

def _build_tls_direction_map(sumo: SumoSimulation) -> dict:
    """Her TLS icin N/S/E/W gelen kenar ID'lerini belirle."""
    dirs_map = {}
    for tid in sumo.get_tls_ids():
        dm = {"N": None, "S": None, "E": None, "W": None}
        jx, jy = sumo.get_junction_position(tid)
        for grp in sumo.get_controlled_links(tid):
            for lnk in grp:
                if not lnk:
                    continue
                eid = lnk[0].split("_")[0]
                if eid.startswith(":"):
                    continue
                shape = sumo.get_lane_shape(f"{eid}_0")
                if not shape:
                    continue
                dx, dy = shape[0][0] - jx, shape[0][1] - jy
                if abs(dx) > abs(dy):
                    dm["W" if dx < 0 else "E"] = eid
                else:
                    dm["S" if dy < 0 else "N"] = eid
        dirs_map[tid] = dm
    return dirs_map


# ── Drift monitoru ───────────────────────────────────────────────────────────

class DriftMonitor:
    def __init__(self, warn_ms=15.0):
        self.warn_ms    = warn_ms
        self._wall_prev = None
        self._sim_prev  = None

    def update(self, sim_time: float, step: int):
        now = time.perf_counter()
        if self._wall_prev is not None:
            wall_dt  = now - self._wall_prev
            sim_dt   = sim_time - self._sim_prev
            drift_ms = abs(sim_dt - wall_dt) * 1000.0
            if drift_ms > self.warn_ms:
                print(f"[DRIFT] {drift_ms:.1f}ms kayma tespit edildi (adim={step})")
        self._wall_prev = now
        self._sim_prev  = sim_time


# ── AI kaplama: SUMO TLS faz kararlari ───────────────────────────────────────

def _apply_ai_decisions(sumo, tls_ids, tls_dirs, step, step_length,
                        last_phase_change):
    intersections = []
    for idx, tid in enumerate(tls_ids):
        d  = tls_dirs[tid]
        nc = sumo.get_edge_vehicle_count(d["N"])
        sc = sumo.get_edge_vehicle_count(d["S"])
        ec = sumo.get_edge_vehicle_count(d["E"])
        wc = sumo.get_edge_vehicle_count(d["W"])
        wt = sum(sumo.get_edge_waiting_time(e) for e in d.values() if e)

        cp, elapsed = sumo.get_tls_phase_info(tid)
        api_phase   = 0 if cp in [0, 1] else 2

        intersections.append({
            "intersection_id": idx,
            "north_count":     nc,
            "south_count":     sc,
            "east_count":      ec,
            "west_count":      wc,
            "queue_length":    min(200.0, wt / 10.0),
            "current_phase":   api_phase,
            "phase_duration":  elapsed,
        })

    try:
        r = requests.post(
            FASTAPI_URL,
            json={"step": step, "intersections": intersections},
            timeout=0.3,
        )
        if r.status_code != 200:
            return

        for dec in r.json().get("decisions", []):
            idx = dec.get("intersection_id")
            nap = dec.get("next_phase", 0)
            if idx is None or idx >= len(tls_ids):
                continue

            tid = tls_ids[idx]
            tp  = 0 if nap in [0, 1] else 2
            cp, _ = sumo.get_tls_phase_info(tid)

            if tp == cp:
                continue
            if cp in {0, 2}:
                min_steps = MIN_GREEN_STEPS / step_length
                if step - last_phase_change.get(tid, 0) < min_steps:
                    continue

            if cp == 0 and tp in [2, 3]:
                tp = 1
            elif cp == 2 and tp in [0, 1]:
                tp = 3

            sumo.set_tls_phase(tid, tp)
            last_phase_change[tid] = step

    except Exception:
        pass


# ── Ana dongu ────────────────────────────────────────────────────────────────

def run(args):
    # --opendrive-file belirtilmemisse ve map.xodr varsa otomatik kullan
    opendrive_file = args.opendrive_file
    if opendrive_file is None and os.path.exists(DEFAULT_XODR):
        opendrive_file = DEFAULT_XODR
        print(f"[BRIDGE] Ozel harita bulundu, kullanilacak: {opendrive_file}")

    sumo = SumoSimulation(
        cfg_path=args.sumo_cfg,
        step_length=args.step_length,
        use_gui=args.sumo_gui,
    )
    carla_sim = CarlaSimulation(
        host=args.carla_host,
        port=args.carla_port,
        step_length=args.step_length,
        sync_vehicle_color=args.sync_vehicle_color,
        sync_vehicle_lights=args.sync_vehicle_lights,
        opendrive_file=opendrive_file,
    )

    # ── Baslat ──────────────────────────────────────────────────────────────
    try:
        sumo.start()
    except SystemExit as e:
        print(f"[BRIDGE] SUMO baslatma hatasi: {e}")
        return

    try:
        carla_sim.start()
    except FileNotFoundError as e:
        print(f"[BRIDGE] {e}")
        sumo.close()
        return
    except Exception as e:
        print(f"[BRIDGE] CARLA baglanamadi: {e}")
        print("[BRIDGE] CARLA sunucusunun calistiginden emin olun (CarlaUE4.exe).")
        sumo.close()
        return

    tls_ids           = sumo.get_tls_ids()
    tls_dirs          = _build_tls_direction_map(sumo)
    drift_mon         = DriftMonitor(warn_ms=15.0)
    last_phase_change = {tid: -MIN_GREEN_STEPS for tid in tls_ids}

    # Kavşak konumlarini topla ve CARLA TLS eslestirmesi olustur
    if tls_ids:
        junction_positions = {
            tid: sumo.get_junction_position(tid) for tid in tls_ids
        }
        tls_lane_positions = {
            tid: sumo.get_tls_lane_positions(tid) for tid in tls_ids
        }
        carla_sim.build_tls_mapping(
            junction_positions, tls_lane_positions, sumo.net_offset
        )

    step        = 0
    empty_steps = 0

    print(f"\n[BRIDGE] Senkronizasyon basladi.")
    print(f"[BRIDGE] Adim suresi : {args.step_length}s | "
          f"TLS manager: {args.tls_manager} | "
          f"TLS sayisi : {len(tls_ids)}")
    if opendrive_file:
        print(f"[BRIDGE] Ozel harita : {opendrive_file}")
    print("[BRIDGE] Durdurmak icin Ctrl+C\n")
    if opendrive_file and not args.no_draw_lane_markings:
        print(
            "[BRIDGE] Not: CARLA generate_opendrive_world cogu surumde .xodr roadMark "
            "mesh'i cizmez; seritler SUMO aginden debug cizgi olarak yansitilir.\n"
        )

    lane_polylines = None

    try:
        while True:
            expected = sumo.get_min_expected()
            if expected == 0:
                empty_steps += 1
                if empty_steps >= MAX_EMPTY_STEPS:
                    print(f"[BRIDGE] Simulasyon tamamlandi (adim={step}).")
                    break
            else:
                empty_steps = 0

            # 1 — SUMO adimi
            sumo.step()
            if lane_polylines is None:
                lane_polylines = sumo.build_lane_edge_polylines()
                if lane_polylines:
                    print(
                        f"[BRIDGE] Serit cizgileri: SUMO'dan {len(lane_polylines)} "
                        f"kenar polyline (beyaz debug)."
                    )
                else:
                    print("[BRIDGE] UYARI: Serit poliline uretilemedi.")

            sim_time = sumo.get_sim_time()
            drift_mon.update(sim_time, step)

            # 2 — Arac durumlarini al
            vehicle_states = sumo.get_vehicle_states()
            current_ids    = set(vehicle_states.keys())
            spawned, destroyed = sumo.compute_diff(current_ids)

            # 3 — CARLA aktörlerini senkronize et
            for sid in spawned:
                carla_sim.spawn_actor(sid, vehicle_states[sid], sumo.net_offset)

            for sid in destroyed:
                carla_sim.destroy_actor(sid)

            for sid in (current_ids - spawned):
                if sid in vehicle_states:
                    carla_sim.update_actor(sid, vehicle_states[sid], sumo.net_offset)

            # 4 — Trafik isigi senkronizasyonu
            if args.tls_manager == "carla" and tls_ids:
                tls_states = sumo.get_tls_states()
                carla_sim.sync_traffic_lights(tls_states)

            # 5 — FastAPI AI telemetri + karar uygulama
            if step % API_EVERY == 0 and tls_ids:
                _apply_ai_decisions(
                    sumo, tls_ids, tls_dirs, step,
                    args.step_length, last_phase_change
                )

            # 6 — Seritler (OpenDRIVE mesh yok) + CARLA adimi
            if not args.no_draw_lane_markings and lane_polylines:
                carla_sim.draw_lane_markings_debug(
                    lane_polylines, sumo.net_offset, args.step_length
                )
            carla_sim.tick()

            if step % LOG_EVERY == 0:
                print(
                    f"[BRIDGE] adim={step:>6} | "
                    f"t={sim_time:>7.1f}s | "
                    f"SUMO arac={len(current_ids):>4} | "
                    f"CARLA aktor={carla_sim.get_actor_count():>4}"
                )

            step += 1

    except KeyboardInterrupt:
        print("\n[BRIDGE] Kullanici tarafindan durduruldu.")
    except Exception as e:
        print(f"[BRIDGE] Beklenmeyen hata: {e}")
        traceback.print_exc()
    finally:
        sumo.close()
        carla_sim.cleanup()
        print("[BRIDGE] Kapatildi.")


# ── Argüman ayrıştırıcı ───────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="TraFix SUMO-CARLA Co-Simulation Bridge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sumo-cfg", default=DEFAULT_SUMO_CFG,
        help="SUMO config dosyasi (.sumocfg)",
    )
    p.add_argument(
        "--opendrive-file", default=None,
        help=(
            "Ozel OpenDRIVE haritasi (.xodr) — verilirse CARLA'ya generate_opendrive_world()\n"
            "ile yuklenir. Belirtilmezse sumo/map.xodr varsa otomatik kullanilir."
        ),
    )
    p.add_argument("--carla-host", default="localhost", help="CARLA sunucu IP")
    p.add_argument("--carla-port", type=int, default=2000, help="CARLA TCP portu")
    p.add_argument(
        "--step-length", type=float, default=DEFAULT_STEP,
        help="Simulasyon adim suresi (saniye) — SUMO ve CARLA icin esit olmali",
    )
    p.add_argument("--sumo-gui", action="store_true", help="SUMO'yu GUI ile baslat")
    p.add_argument(
        "--tls-manager", choices=["sumo", "carla"], default="carla",
        help="Trafik lambasi yoneticisi",
    )
    p.add_argument(
        "--sync-vehicle-color", action="store_true", default=True,
        help="SUMO arac renklerini CARLA'ya yansit",
    )
    p.add_argument(
        "--sync-vehicle-lights", action="store_true", default=False,
        help="SUMO arac farlarini CARLA'ya yansit",
    )
    p.add_argument(
        "--no-draw-lane-markings", action="store_true", default=False,
        help=(
            "SUMO serit kenarlarini CARLA debug cizgisi olarak cizme "
            "(generate_opendrive_world'da roadMark mesh'i olmadigi icin varsayilan: acik)"
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
