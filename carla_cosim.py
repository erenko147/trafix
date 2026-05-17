"""
SUMO – CARLA Ko-Simülasyonu  (Otomatik harita algılama)
=======================================================
Herhangi bir SUMO haritası için çalışır.  Yeni bir .net.xml / .sumocfg
eklendiğinde otomatik algılanır, XODR üretilir, kameralar, yollar
ve trafik ışıkları tam entegre edilir.

Akış:
  1. Arama dizinlerinde .sumocfg bul (ya da parametre ver)
  2. sumocfg içindeki net-file referansını çözümle
  3. net-file → netconvert → .xodr  (önbellek varsa üretme, her zaman patch'le)
  4. CARLA'ya yükle
  5. TraCI ile SUMO'yu adım adım çalıştır
  6. Her adımda:
       • SUMO araçlarını CARLA'ya kinematik yansıt
       • SUMO faz string'ini (G/Y/R) bearing eşlemesiyle CARLA ışıklarına uygula
  7. Her DECISION_STEPS adımda AI kararı için FastAPI'ye telemetri gönder

Kamera yerleşimi:
  Her kavşak koluna 1 kamera.  Kol yönü gerçek şerit şekil vektöründen
  hesaplanır; N/S/E/W sabiti yok, T/Y/çok kollu kavşaklar desteklenir.

Koordinat dönüşümü:
    CARLA.x =  SUMO.x
    CARLA.y = -SUMO.y
    CARLA.yaw = SUMO.angle - 90
"""

from __future__ import annotations
import math
import os
import subprocess
import sys
import threading
import time

import requests

# ── SUMO TraCI ────────────────────────────────────────────────────────
_SUMO_HOME = os.environ.get("SUMO_HOME", "")
if _SUMO_HOME:
    sys.path.append(os.path.join(_SUMO_HOME, "tools"))
try:
    import traci
except ImportError:
    traci = None

# ── Paylaşılan durum ──────────────────────────────────────────────────
from carla_integration import CARLA_STATUS, set_frame

# ── YOLOv8 ───────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO
    _yolo = _YOLO("yolo11s.pt")
    _yolo.fuse()
    print("[CoSim] YOLO11s yüklendi.")
except Exception as _ye:
    _yolo = None
    print(f"[CoSim] YOLO yüklenemedi, ham akış: {_ye}")

_VEHICLE_CLASSES = [2, 3, 5, 7]
_yolo_counts: dict[int, int] = {}
_frame_counters: dict[int, int] = {}   # per-camera tick counter for data-only skip

# ── Sabitler ─────────────────────────────────────────────────────────
HERE              = os.path.dirname(os.path.abspath(__file__))
CAMERA_Z          =  5.5
CAMERA_Z_OFFSET   = -2.0
CAMERA_PITCH      = -50.0
CAMERA_YAW_OFFSET =  0.0
CAMERA_ROLL       =  0.0
CAMERA_X_OFFSET   = -2.0
CAMERA_Y_OFFSET   =  -0.50
TL_POLE_SIDE      =  2.5
DECISION_STEPS    = 10
MIN_GREEN         = 10
MAX_TLS           = 50   # işlenecek maksimum TL sayısı
MAX_APPROACHES    =  8   # kavşak başına maks kol (cam ID hesabı için)

_SIGNAL_ZOFFSET_CORRECT = 0.0

# ── ROI Polygon sözlüğü ───────────────────────────────────────────────
# cam_id → liste of [x, y] piksel noktaları (istenen sayıda köşe).
# Boşsa o kamera için tüm alan geçerli sayılır.
# main_carla.py üzerinden POST /camera/{id}/roi ile runtime'da değiştirilebilir.
ROI_POLYGONS: dict[int, list[list[int]]] = {}

# ── Active camera set — dashboard POSTs /carla/active_cameras to update ──
# Empty  → all cameras run full pipeline.
# Non-empty → only listed IDs run YOLO/BGR/ROI; others return immediately.
ACTIVE_CAMERA_IDS: set[int] = set()


# ─────────────────────────────────────────────────────────────────────
# Harita keşfi
# ─────────────────────────────────────────────────────────────────────

def discover_sumo_cfg(search_dirs: list[str] | None = None) -> tuple[str, str] | None:
    """
    Arama dizinlerinde ilk geçerli .sumocfg + net-file çiftini döndürür.
    Bulunamazsa None döner; varsayılan sumo/ dizinine düşülür.
    """
    import xml.etree.ElementTree as ET
    if search_dirs is None:
        search_dirs = [HERE]

    for base in search_dirs:
        for root_dir, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d != '__pycache__')
            for fname in sorted(files):
                if not fname.endswith(".sumocfg"):
                    continue
                cfg_path = os.path.join(root_dir, fname)
                try:
                    tree = ET.parse(cfg_path)
                    net_elem = tree.find(".//net-file")
                    if net_elem is None:
                        continue
                    net_val = net_elem.get("value", "").strip()
                    if not net_val:
                        continue
                    if not os.path.isabs(net_val):
                        net_val = os.path.join(root_dir, net_val)
                    net_val = os.path.normpath(net_val)
                    if os.path.exists(net_val):
                        print(f"[CoSim] SUMO config: {cfg_path}")
                        print(f"[CoSim] Net file   : {net_val}")
                        return cfg_path, net_val
                except Exception as exc:
                    print(f"[CoSim] {cfg_path} okunamadı: {exc}")
    return None


# ─────────────────────────────────────────────────────────────────────
# XODR üretimi + patch'ler
# ─────────────────────────────────────────────────────────────────────

def _patch_xodr_road_marks(xodr_path: str) -> None:
    import xml.etree.ElementTree as ET
    tree      = ET.parse(xodr_path)
    root_elem = tree.getroot()
    MIN_W     = 0.20
    patched   = 0

    for ls in root_elem.iter("laneSection"):
        center = ls.find("center")
        if center:
            for lane in center.findall("lane"):
                for rm in lane.findall("roadMark"):
                    if rm.get("type", "none") != "none":
                        rm.set("color", "yellow")
                        if float(rm.get("width", "0")) < MIN_W:
                            rm.set("width", str(MIN_W))
                        patched += 1

        for side_tag in ("left", "right"):
            side = ls.find(side_tag)
            if side is None:
                continue
            for lane in side.findall("lane"):
                for rm in lane.findall("roadMark"):
                    if rm.get("type", "none") != "none":
                        if rm.get("color", "standard") == "standard":
                            rm.set("color", "white")
                        if float(rm.get("width", "0")) < MIN_W:
                            rm.set("width", str(MIN_W))
                        patched += 1

    tree.write(xodr_path, encoding="unicode", xml_declaration=True)
    print(f"[CoSim] XODR patch: {patched} roadMark color/width düzeltildi.")


def _patch_xodr_signal_z(xodr_path: str) -> None:
    import xml.etree.ElementTree as ET
    tree    = ET.parse(xodr_path)
    root    = tree.getroot()
    patched = 0
    for sig in root.iter("signal"):
        if sig.get("zOffset", "?") != str(_SIGNAL_ZOFFSET_CORRECT):
            sig.set("zOffset", str(_SIGNAL_ZOFFSET_CORRECT))
            patched += 1
    tree.write(xodr_path, encoding="unicode", xml_declaration=True)
    print(f"[CoSim] XODR patch: {patched} sinyal zOffset={_SIGNAL_ZOFFSET_CORRECT}.")


def _generate_xodr(net_file: str, xodr_file: str) -> None:
    """net.xml → xodr dönüşümü + patch'ler (herhangi bir harita dosyasıyla çalışır)."""
    if not os.path.exists(xodr_file):
        print(f"[CoSim] {os.path.basename(net_file)} → OpenDRIVE dönüştürülüyor …")
        ret = subprocess.call(
            ["netconvert", "--sumo-net-file", net_file, "--opendrive-output", xodr_file]
        )
        if ret != 0 or not os.path.exists(xodr_file):
            raise RuntimeError(
                f"netconvert başarısız!\nnet_file={net_file}\nSUMO_HOME={_SUMO_HOME}"
            )
        print(f"[CoSim] XODR oluşturuldu: {xodr_file}")
    else:
        print(f"[CoSim] XODR mevcut: {xodr_file}")
    _patch_xodr_road_marks(xodr_file)
    _patch_xodr_signal_z(xodr_file)


# ─────────────────────────────────────────────────────────────────────
# CARLA harita yükleme
# ─────────────────────────────────────────────────────────────────────

def _load_map(client, xodr_file: str) -> "carla.World":
    import carla
    with open(xodr_file, encoding="utf-8") as f:
        xodr = f.read()
    params = carla.OpendriveGenerationParameters(
        vertex_distance=2.0,
        max_road_length=50.0,
        wall_height=0.0,
        additional_width=0.6,
        smooth_junctions=True,
        enable_mesh_visibility=True,
    )
    print("[CoSim] CARLA'ya SUMO haritası yükleniyor (30-60 sn) …")
    client.set_timeout(120.0)
    client.generate_opendrive_world(xodr, params)
    client.set_timeout(15.0)
    world    = client.get_world()
    settings = world.get_settings()
    if settings.synchronous_mode:
        settings.synchronous_mode = False
        world.apply_settings(settings)
    print(f"[CoSim] Harita yüklendi: {world.get_map().name}")
    return world


# ─────────────────────────────────────────────────────────────────────
# Koordinat dönüşümü
# ─────────────────────────────────────────────────────────────────────

def _to_carla(sx: float, sy: float, angle: float, z: float = 0.3):
    import carla
    return carla.Transform(
        carla.Location(x=sx, y=-sy, z=z),
        carla.Rotation(yaw=angle - 90.0),
    )


# ─────────────────────────────────────────────────────────────────────
# Trafik ışığı senkronizasyonu — gerçek SUMO faz string'i
# ─────────────────────────────────────────────────────────────────────

def _sumo_char_to_carla_state(ch: str):
    """G/g → Green, Y/y/u/U → Yellow, diğer → Red."""
    import carla
    c = ch.upper()
    if c == "G":
        return carla.TrafficLightState.Green
    if c in ("Y", "U"):
        return carla.TrafficLightState.Yellow
    return carla.TrafficLightState.Red


def _apply_sumo_tl_state(
    tls_id: str,
    carla_tls: list,
    jx: float,
    jy_carla: float,
) -> None:
    """
    SUMO'nun gerçek faz string'ini okur (örn. "GGGrrrGGGrrr") ve
    bearing eşlemesiyle CARLA trafik ışıklarını günceller.

    Her CARLA ışığı için en yakın SUMO yaklaşım kolunun durumu uygulanır.
    4 fazlı sabit şema yok; T/Y kavşaklar ve çok fazlı döngüler desteklenir.
    """
    if not carla_tls:
        return
    try:
        state_str = traci.trafficlight.getRedYellowGreenState(tls_id)
        links     = traci.trafficlight.getControlledLinks(tls_id)
    except Exception:
        return
    if not state_str or not links:
        return

    # Her yaklaşım edge'i için dominant durum (G > Y > R)
    edge_state: dict[str, str] = {}
    for i, link in enumerate(links):
        if not link or i >= len(state_str):
            continue
        edge_id = link[0][0].rsplit("_", 1)[0]
        ch      = state_str[i].upper()
        prev    = edge_state.get(edge_id, "R")
        if prev == "R" or (prev == "Y" and ch == "G"):
            edge_state[edge_id] = ch

    # Her edge için CARLA uzayında bearing hesapla
    approach_info: list[tuple[float, str]] = []  # (bearing_deg, state_char)
    for edge_id, ch in edge_state.items():
        try:
            shape = traci.lane.getShape(f"{edge_id}_0")
            if not shape:
                continue
            x0, y0_sumo = shape[0]
            dx  = x0        - jx
            dy  = (-y0_sumo) - jy_carla
            bearing = math.degrees(math.atan2(dy, dx))
        except Exception:
            continue
        approach_info.append((bearing, ch))

    if not approach_info:
        return

    for tl in carla_tls:
        loc = tl.get_location()
        tl_bearing = math.degrees(math.atan2(loc.y - jy_carla, loc.x - jx))
        best_ch, best_diff = "R", 360.0
        for (bear, ch) in approach_info:
            diff = abs((tl_bearing - bear + 180) % 360 - 180)
            if diff < best_diff:
                best_diff, best_ch = diff, ch
        try:
            tl.freeze(True)
            tl.set_state(_sumo_char_to_carla_state(best_ch))
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────
# Yardımcı: çakışan bbox deduplication (cross-class NMS)
# ─────────────────────────────────────────────────────────────────────

def _dedup_boxes(xyxy, scores, iou_thresh: float = 0.45) -> list[int]:
    """
    Greedy NMS — YOLO'nun class bazlı NMS'inden sızan çakışan kutuları temizler.
    Aynı araç birden fazla class'ta (örn. car + truck) tespit edildiğinde
    confidence'ı düşük olanı eler.  Döndürülen liste tutulacak indisler.
    """
    import numpy as _np
    if len(xyxy) == 0:
        return []
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        ix1  = _np.maximum(x1[i], x1[order[1:]])
        iy1  = _np.maximum(y1[i], y1[order[1:]])
        ix2  = _np.minimum(x2[i], x2[order[1:]])
        iy2  = _np.minimum(y2[i], y2[order[1:]])
        inter = _np.maximum(0.0, ix2 - ix1) * _np.maximum(0.0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[_np.where(iou <= iou_thresh)[0] + 1]
    return keep


_COCO_VEHICLE_NAMES = {2: "car", 3: "moto", 5: "bus", 7: "truck"}


# ─────────────────────────────────────────────────────────────────────
# Yardımcı: yol yüzeyi z
# ─────────────────────────────────────────────────────────────────────

def _road_z(carla_map, x: float, y: float, fallback: float = 0.0) -> float:
    import carla
    wp = carla_map.get_waypoint(
        carla.Location(x=x, y=y, z=0.0),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    return wp.transform.location.z if wp else fallback


# ─────────────────────────────────────────────────────────────────────
# Şerit çizgisi overlay
# ─────────────────────────────────────────────────────────────────────

def _draw_lane_markings_overlay(world, carla_map) -> int:
    import carla

    STEP_M     = 1.0
    THICKNESS  = 0.03
    Z_LIFT     = 0.01
    DASH_ON    = 3.0
    DASH_CYCLE = 12.0
    debug      = world.debug
    white      = carla.Color(75, 75, 75)
    yellow     = carla.Color(80, 60,  0)
    none_types = (carla.LaneMarkingType.NONE, carla.LaneMarkingType.Other)

    def _mark_color(m):
        return yellow if m.color == carla.LaneMarkingColor.Yellow else white

    def _boundary(wp, right: bool) -> carla.Location:
        yaw_r = math.radians(wp.transform.rotation.yaw)
        rx, ry = -math.sin(yaw_r), math.cos(yaw_r)
        sign = 1.0 if right else -1.0
        half = wp.lane_width / 2.0
        loc  = wp.transform.location
        return carla.Location(
            x=loc.x + rx * half * sign,
            y=loc.y + ry * half * sign,
            z=loc.z + Z_LIFT,
        )

    drawn     = 0
    waypoints = carla_map.generate_waypoints(STEP_M)
    for wp in waypoints:
        nexts = wp.next(STEP_M)
        if not nexts:
            continue
        nw = nexts[0]
        s  = wp.s
        for is_right, get_mark in (
            (True,  lambda w: w.right_lane_marking),
            (False, lambda w: w.left_lane_marking),
        ):
            rm = get_mark(wp)
            if rm.type in none_types:
                continue
            if rm.type == carla.LaneMarkingType.Broken and (s % DASH_CYCLE) >= DASH_ON:
                continue
            debug.draw_line(
                _boundary(wp, is_right), _boundary(nw, is_right),
                thickness=THICKNESS, color=_mark_color(rm), life_time=0.0,
            )
            drawn += 1

    print(f"[CoSim] Şerit overlay: {drawn} segment ({len(waypoints)} waypoint, adım={STEP_M}m).")
    return drawn


# ─────────────────────────────────────────────────────────────────────
# TL konumlarını yol yüzeyine + kenarına düzelt
# ─────────────────────────────────────────────────────────────────────

def _fix_tl_positions(world, carla_map) -> None:
    import carla
    tls = list(world.get_actors().filter("traffic.traffic_light"))
    if not tls:
        print("[CoSim] Haritada CARLA TL yok; konum düzeltmesi atlandı.")
        return
    moved = 0
    for tl in tls:
        loc = tl.get_location()
        rz  = _road_z(carla_map, loc.x, loc.y, fallback=0.0)
        wp  = carla_map.get_waypoint(
            carla.Location(x=loc.x, y=loc.y, z=rz),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if wp:
            yaw_r = math.radians(wp.transform.rotation.yaw)
            rx    = -math.sin(yaw_r)
            ry    =  math.cos(yaw_r)
            side  = wp.lane_width / 2.0 + 0.8
            new_loc = carla.Location(x=loc.x + rx * side, y=loc.y + ry * side, z=rz)
        else:
            new_loc = carla.Location(x=loc.x, y=loc.y, z=rz)
        try:
            tl.set_location(new_loc)
            moved += 1
        except Exception:
            pass
    print(f"[CoSim] {moved}/{len(tls)} TL yol yüzeyine taşındı.")


# ─────────────────────────────────────────────────────────────────────
# Kamera debug kutusu
# ─────────────────────────────────────────────────────────────────────

def _draw_camera_debug(world, camera) -> None:
    import carla
    t   = camera.get_transform()
    fwd = t.get_forward_vector()
    body_loc = carla.Location(
        x=t.location.x - fwd.x * 0.35,
        y=t.location.y - fwd.y * 0.35,
        z=t.location.z - fwd.z * 0.35,
    )
    world.debug.draw_box(
        carla.BoundingBox(body_loc, carla.Vector3D(0.15, 0.08, 0.08)),
        t.rotation, thickness=0.05,
        color=carla.Color(220, 220, 220), life_time=0.0,
    )


# ─────────────────────────────────────────────────────────────────────
# Ana ko-sim döngüsü
# ─────────────────────────────────────────────────────────────────────

def run_cosim(
    api_port: int = 8000,
    stop_event: threading.Event | None = None,
    no_sumo_gui: bool = True,
    sumo_cfg: str | None = None,
    net_file: str | None = None,
    xodr_file: str | None = None,
) -> None:
    """
    sumo_cfg / net_file verilmezse HERE altında otomatik .sumocfg aranır.
    Yeni bir SUMO haritası klasörü eklendiğinde hiçbir kod değişikliği gerekmez.
    """
    if traci is None:
        CARLA_STATUS["error"] = "TraCI bulunamadı. SUMO_HOME ayarlı mı?"
        print("[CoSim] HATA: TraCI import edilemedi.")
        return

    # ── Harita dosyalarını çözümle ───────────────────────────────────
    if sumo_cfg is None:
        found = discover_sumo_cfg()
        if found is None:
            # Geriye dönük uyumluluk: eski varsayılan yollar
            sumo_cfg = os.path.join(HERE, "sumo", "demo.sumocfg")
            net_file = net_file or os.path.join(HERE, "sumo", "map.net.xml")
            if not os.path.exists(sumo_cfg) or not os.path.exists(net_file):
                CARLA_STATUS["error"] = "SUMO haritası bulunamadı."
                print("[CoSim] HATA: Hiçbir .sumocfg bulunamadı.")
                return
            print("[CoSim] Varsayılan sumo/demo.sumocfg kullanılıyor.")
        else:
            sumo_cfg, discovered_net = found
            net_file = net_file or discovered_net

    if xodr_file is None:
        xodr_file = os.path.splitext(net_file)[0] + ".xodr"

    api_url = f"http://127.0.0.1:{api_port}/telemetry_batch"

    # ── CARLA bağlantısı ─────────────────────────────────────────────
    import carla
    client = None
    for attempt in range(1, 13):
        try:
            client = carla.Client("127.0.0.1", 2000)
            client.set_timeout(10.0)
            _ = client.get_server_version()
            print(f"[CoSim] CARLA'ya bağlandı (deneme {attempt})")
            break
        except Exception as exc:
            print(f"[CoSim] CARLA bekleniyor ({attempt}/12): {exc}")
            if stop_event and stop_event.is_set():
                return
            time.sleep(10)
    else:
        CARLA_STATUS["error"] = "CARLA 2 dakika içinde başlatılamadı."
        return

    # ── Harita yükle ─────────────────────────────────────────────────
    try:
        _generate_xodr(net_file, xodr_file)
        world = _load_map(client, xodr_file)
    except Exception as exc:
        CARLA_STATUS["error"] = str(exc)
        print(f"[CoSim] Harita yükleme hatası: {exc}")
        return

    carla_map = world.get_map()
    _fix_tl_positions(world, carla_map)
    _draw_lane_markings_overlay(world, carla_map)
    CARLA_STATUS.update({
        "connected": True,
        "map": os.path.basename(net_file),
        "error": None,
    })

    # ── SUMO başlat ─────────────────────────────────────────────────
    try:
        sumo_bin = "sumo" if no_sumo_gui else "sumo-gui"
        traci.start([sumo_bin, "-c", sumo_cfg, "--no-warnings"])
    except Exception as exc:
        CARLA_STATUS["error"] = f"SUMO başlatılamadı: {exc}"
        print(f"[CoSim] SUMO hatası: {exc}")
        return

    all_tls = sorted(traci.trafficlight.getIDList())
    tls_ids = all_tls[:MAX_TLS]
    if len(all_tls) > MAX_TLS:
        print(f"[CoSim] {len(all_tls)} TL bulundu; ilk {MAX_TLS} işleniyor (MAX_TLS artırılabilir).")

    # ── Kavşak bilgilerini topla ─────────────────────────────────────
    # Hem N/S/E/W binning (telemetri uyumluluğu) hem gerçek bearing (kamera+TL)
    intersections: dict[str, dict] = {}
    for tls_id in tls_ids:
        try:
            jx, jy = traci.junction.getPosition(tls_id)
        except Exception:
            jx, jy = 0.0, 0.0

        links = traci.trafficlight.getControlledLinks(tls_id)
        approach_edges: set[str] = set()
        for link in links:
            if link:
                approach_edges.add(link[0][0].rsplit("_", 1)[0])

        # N/S/E/W binning (telemetri API uyumluluğu)
        dmap: dict[str, str | None] = {"north": None, "south": None, "east": None, "west": None}
        for edge_id in approach_edges:
            try:
                shape = traci.lane.getShape(f"{edge_id}_0")
                if not shape:
                    continue
                x0, y0 = shape[0]
                dx, dy = x0 - jx, y0 - jy
                key = ("west" if dx < 0 else "east") if abs(dx) > abs(dy) \
                      else ("south" if dy < 0 else "north")
                if dmap[key] is None:
                    dmap[key] = edge_id
            except Exception:
                pass

        # Bearing sıralı kol listesi (kamera ID'leri tutarlı olsun)
        edge_bearing: list[tuple[float, str]] = []
        for edge_id in approach_edges:
            try:
                shape = traci.lane.getShape(f"{edge_id}_0")
                if not shape:
                    continue
                x0, y0 = shape[0]
                bearing = math.degrees(math.atan2(y0 - jy, x0 - jx))
                edge_bearing.append((bearing, edge_id))
            except Exception:
                pass
        edge_bearing.sort(key=lambda t: t[0])

        # edge_id → cam_id eşlemi (telemetri için N/S/E/W sayısı almak için)
        edge_to_cam: dict[str, int] = {
            edge_id: i_junc * MAX_APPROACHES + arm_idx
            for arm_idx, (_, edge_id) in enumerate(edge_bearing[:MAX_APPROACHES])
            for i_junc in [list(tls_ids).index(tls_id)]
        }

        intersections[tls_id] = {
            "jx": jx, "jy": jy,
            "dmap": dmap,
            "edge_bearing": edge_bearing,
            "edge_to_cam": edge_to_cam,
        }

    # edge_to_cam içindeki i_junc bağımlılığını düzelt (yukarıda list.index O(n), tek seferlik OK)
    for i_j, tls_id in enumerate(tls_ids):
        info = intersections[tls_id]
        info["edge_to_cam"] = {
            edge_id: i_j * MAX_APPROACHES + arm_idx
            for arm_idx, (_, edge_id) in enumerate(info["edge_bearing"][:MAX_APPROACHES])
        }

    # ── CARLA TL'lerini kavşaklara eşle ─────────────────────────────
    carla_all_tls = list(world.get_actors().filter("traffic.traffic_light"))
    junction_carla_tls: dict[int, list] = {}
    for i, tls_id in enumerate(tls_ids):
        info = intersections[tls_id]
        cx, cy = info["jx"], -info["jy"]
        junction_carla_tls[i] = [
            tl for tl in carla_all_tls
            if (tl.get_location().x - cx) ** 2 + (tl.get_location().y - cy) ** 2 < 60 ** 2
        ]

    # ── Kameraları kur ──────────────────────────────────────────────
    # Kamera yönü gerçek bearing'den hesaplanır; N/S/E/W sabit değil.
    # Kamera gelen trafiğe bakar (kavşaktan uzak yön).
    # SUMO bearing → CARLA yaw: yaw = -bearing (y ekseni çevrildiği için)
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "640")
    cam_bp.set_attribute("image_size_y", "640")
    cam_bp.set_attribute("fov", "100")

    cameras: list = []
    CARLA_STATUS["camera_ids"] = []   # aktif kamera ID'leri dashboard'a açılır

    for i, tls_id in enumerate(tls_ids):
        info  = intersections[tls_id]
        jx_s  = info["jx"]
        jy_s  = info["jy"]

        for arm_idx, (sumo_bearing, edge_id) in enumerate(info["edge_bearing"][:MAX_APPROACHES]):
            cam_id = i * MAX_APPROACHES + arm_idx

            try:
                shape = traci.lane.getShape(f"{edge_id}_0")
                if not shape:
                    continue
                # Durma çizgisi = şeklin kavşağa en yakın ucu (son nokta)
                px_s, py_s = shape[-1]
            except Exception:
                px_s, py_s = jx_s, jy_s

            px_c = px_s
            py_c = -py_s   # CARLA koordinatı

            # Kamera yaw: gelen trafiğe bakar (kavşaktan uzağa)
            # SUMO bearing (math açısı, 0=Doğu, 90=Kuzey) → CARLA yaw (0=Doğu, 90=Güney)
            cam_yaw     = (-sumo_bearing) % 360.0
            eff_yaw_r   = math.radians(cam_yaw + CAMERA_YAW_OFFSET)
            crx = -math.sin(eff_yaw_r)   # CARLA sol-el sağ vektörü
            cry =  math.cos(eff_yaw_r)
            fwd_x = math.cos(eff_yaw_r)
            fwd_y = math.sin(eff_yaw_r)

            pole_x = px_c + crx * TL_POLE_SIDE
            pole_y = py_c + cry * TL_POLE_SIDE
            rz     = _road_z(carla_map, pole_x, pole_y)

            transform = carla.Transform(
                carla.Location(
                    x=pole_x + crx * CAMERA_Y_OFFSET + fwd_x * CAMERA_X_OFFSET,
                    y=pole_y + cry * CAMERA_Y_OFFSET + fwd_y * CAMERA_X_OFFSET,
                    z=rz + CAMERA_Z + CAMERA_Z_OFFSET,
                ),
                carla.Rotation(
                    pitch=CAMERA_PITCH,
                    yaw=cam_yaw + CAMERA_YAW_OFFSET,
                    roll=CAMERA_ROLL,
                ),
            )

            try:
                cam = world.spawn_actor(cam_bp, transform)
            except Exception as exc:
                print(f"[CoSim] Kamera spawn hatası ({tls_id} kol{arm_idx}): {exc}")
                continue

            _on_frame_errors = [0]

            def _on_frame(image, _cid=cam_id, _errs=_on_frame_errors):
                try:
                    import numpy as _np
                    import cv2 as _cv2

                    # watched = camera is open on the dashboard right now
                    watched = not ACTIVE_CAMERA_IDS or _cid in ACTIVE_CAMERA_IDS

                    # ── Data-only mode: skip most frames ────────────
                    # YOLO still runs every DECISION_STEPS ticks so the
                    # traffic-light AI always has fresh vehicle counts.
                    if not watched:
                        _frame_counters[_cid] = _frame_counters.get(_cid, 0) + 1
                        if _frame_counters[_cid] % DECISION_STEPS != 0:
                            return

                    # ── Decode BGRA buffer ───────────────────────────
                    arr = _np.frombuffer(image.raw_data, dtype=_np.uint8).reshape(
                        (image.height, image.width, 4)
                    )
                    if watched:
                        # Full cvtColor — colour-accurate for visual output
                        bgr   = _cv2.cvtColor(arr, _cv2.COLOR_BGRA2BGR)
                        frame = bgr.copy()
                    else:
                        # Fast numpy slice (CARLA BGRA → drop A; channel order stays BGR)
                        # No cvtColor, no display frame allocation
                        bgr = _np.ascontiguousarray(arr[:, :, :3])

                    # ── ROI polygon (shared by both modes) ───────────
                    poly_raw  = ROI_POLYGONS.get(_cid)
                    has_roi   = bool(poly_raw and len(poly_raw) >= 3)
                    pts_check = _np.array(poly_raw, dtype=_np.int32) if has_roi else None

                    # Draw ROI overlay only for watched cameras
                    if watched and has_roi:
                        overlay = frame.copy()
                        _cv2.fillPoly(overlay, [pts_check], (0, 180, 255))
                        _cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
                        _cv2.polylines(frame, [pts_check], True, (0, 180, 255), 2)
                        for pt in poly_raw:
                            _cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 180, 255), -1)

                    # ── YOLO — always runs (AI needs counts) ─────────
                    count = 0
                    if _yolo is not None:
                        results = _yolo(bgr, classes=_VEHICLE_CLASSES,
                                        verbose=False, agnostic_nms=True, iou=0.4)
                        boxes = results[0].boxes
                        if len(boxes):
                            xyxy    = boxes.xyxy.cpu().numpy()
                            scores  = boxes.conf.cpu().numpy()
                            cls_ids = boxes.cls.cpu().numpy().astype(int)
                            keep    = _dedup_boxes(xyxy, scores, iou_thresh=0.45)
                            xyxy    = xyxy[keep]; scores = scores[keep]; cls_ids = cls_ids[keep]
                        else:
                            xyxy    = _np.empty((0, 4))
                            scores  = _np.empty(0)
                            cls_ids = _np.empty(0, dtype=int)

                        for (x1, y1, x2, y2), sc, cl in zip(xyxy, scores, cls_ids):
                            foot   = (int((x1 + x2) / 2), int(y2))
                            in_roi = pts_check is None or \
                                     _cv2.pointPolygonTest(pts_check, foot, False) >= 0
                            if in_roi:
                                count += 1
                            # ── Visual overlays: watched only ────────
                            if watched:
                                color = (0, 200, 0) if in_roi else (70, 70, 70)
                                _cv2.rectangle(frame,
                                               (int(x1), int(y1)), (int(x2), int(y2)),
                                               color, 2)
                                lbl = f"{_COCO_VEHICLE_NAMES.get(int(cl), 'veh')} {sc:.2f}"
                                _cv2.putText(frame, lbl,
                                             (int(x1), max(int(y1) - 5, 10)),
                                             _cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                             color, 1, _cv2.LINE_AA)
                        _yolo_counts[_cid] = count

                    # ── HUD + MJPEG encode: watched only ─────────────
                    if watched:
                        lbl_txt = f"ARAC: {count}"
                        fnt     = _cv2.FONT_HERSHEY_SIMPLEX
                        fscale  = 0.9; thick = 2
                        (tw, th), bl = _cv2.getTextSize(lbl_txt, fnt, fscale, thick)
                        pad = 8
                        rx  = frame.shape[1] - tw - pad * 2; ry = pad
                        _cv2.rectangle(frame,
                                       (rx - pad, ry),
                                       (rx + tw + pad, ry + th + bl + pad),
                                       (0, 0, 0), -1)
                        _cv2.putText(frame, lbl_txt, (rx, ry + th),
                                     fnt, fscale, (0, 255, 0), thick, _cv2.LINE_AA)
                        _, enc = _cv2.imencode(".jpg", frame, [_cv2.IMWRITE_JPEG_QUALITY, 72])
                        set_frame(_cid, enc.tobytes())

                except Exception as _e:
                    if _errs[0] < 3:
                        print(f"[CoSim] Kamera {_cid} frame hatası: {_e}")
                        _errs[0] += 1

            cam.listen(_on_frame)
            cameras.append(cam)
            CARLA_STATUS["camera_ids"].append(cam_id)
            _draw_camera_debug(world, cam)
            print(
                f"[CoSim] Kamera {cam_id}: kavşak={tls_id} kol={arm_idx} "
                f"bearing={sumo_bearing:.0f}° yaw={cam_yaw:.0f}° "
                f"pos=({pole_x:.1f},{pole_y:.1f}) z={rz+CAMERA_Z:.1f}m"
            )

    # ── SUMO vType → CARLA blueprint ────────────────────────────────
    def _first(patterns):
        for pat in patterns:
            hits = bp_lib.filter(pat)
            if hits:
                return hits[0]
        return None

    _BP_CAR        = _first(("vehicle.lincoln*", "vehicle.tesla.model3", "vehicle.audi.a2"))
    _BP_TRUCK      = _first(("vehicle.carlamotors*", "vehicle.tesla.cybertruck", "vehicle.nissan.patrol*"))
    _BP_BUS        = _first(("vehicle.mitsubishi.fusorosa", "vehicle.volkswagen.t2*"))
    _BP_MOTORCYCLE = _first(("vehicle.yamaha*", "vehicle.kawasaki*", "vehicle.harley*"))
    _BP_DEFAULT    = _BP_CAR or bp_lib.filter("vehicle.*")[0]

    _SUMO_BP_MAP = [
        (("truck", "trailer", "heavy", "lorry"),            _BP_TRUCK      or _BP_DEFAULT),
        (("bus", "coach", "minibus"),                       _BP_BUS        or _BP_DEFAULT),
        (("motorcycle", "moped", "bike"),                   _BP_MOTORCYCLE or _BP_DEFAULT),
        (("passenger", "car", "sedan", "taxi",
          "police", "emergency", "vehicle"),                _BP_CAR        or _BP_DEFAULT),
    ]

    def _bp_for_sumo_type(type_id: str):
        tid = type_id.lower()
        for keywords, bp in _SUMO_BP_MAP:
            if bp and any(k in tid for k in keywords):
                return bp
        return _BP_DEFAULT

    # ── Ana döngü ───────────────────────────────────────────────────
    actor_map: dict[str, object] = {}
    last_phase_change: dict[str, int] = {tid: -MIN_GREEN for tid in tls_ids}
    step = 0

    CARLA_STATUS["num_intersections"] = len(tls_ids)
    print(f"[CoSim] Ko-sim başladı — {len(tls_ids)} kavşak, {len(cameras)} kamera")

    try:
        while not (stop_event and stop_event.is_set()):
            if traci.simulation.getMinExpectedNumber() == 0:
                print("[CoSim] SUMO simülasyonu tamamlandı.")
                break

            traci.simulationStep()
            step += 1

            # ── Araç senkronizasyonu ──────────────────────────────
            current_ids = set(traci.vehicle.getIDList())
            for sid in list(actor_map.keys()):
                if sid not in current_ids:
                    try:
                        actor_map[sid].destroy()
                    except Exception:
                        pass
                    del actor_map[sid]

            for vid in current_ids:
                try:
                    sx, sy = traci.vehicle.getPosition(vid)
                    angle  = traci.vehicle.getAngle(vid)
                    tf     = _to_carla(sx, sy, angle)
                    if vid in actor_map:
                        actor_map[vid].set_transform(tf)
                    else:
                        bp    = _bp_for_sumo_type(traci.vehicle.getTypeID(vid))
                        actor = world.try_spawn_actor(bp, tf)
                        if actor:
                            actor.set_simulate_physics(False)
                            actor_map[vid] = actor
                except Exception:
                    pass

            CARLA_STATUS["num_vehicles"] = len(actor_map)

            # ── TL senkronizasyonu (gerçek SUMO faz string'i) ─────
            for i, tls_id in enumerate(tls_ids):
                info = intersections[tls_id]
                _apply_sumo_tl_state(
                    tls_id, junction_carla_tls[i],
                    info["jx"], -info["jy"],
                )

            # ── AI kararı ─────────────────────────────────────────
            if step % DECISION_STEPS != 0:
                continue

            batch: dict = {"step": step, "intersections": []}
            for i, tls_id in enumerate(tls_ids):
                info        = intersections[tls_id]
                dmap        = info["dmap"]
                edge_to_cam = info["edge_to_cam"]

                def _dir_count(direction: str) -> int:
                    edge = dmap.get(direction)
                    if edge is None:
                        return 0
                    return _yolo_counts.get(edge_to_cam.get(edge, -1), 0)

                total_wait = sum(
                    float(traci.edge.getWaitingTime(e))
                    for e in dmap.values() if e
                )
                queue = min(200.0, total_wait / 10.0)
                curr  = int(traci.trafficlight.getPhase(tls_id))
                try:
                    prog = traci.trafficlight.getPhaseDuration(tls_id)
                    nxt  = traci.trafficlight.getNextSwitch(tls_id)
                    sim  = traci.simulation.getTime()
                    dur  = max(0.0, prog - max(0.0, nxt - sim))
                except Exception:
                    dur = 0.0

                batch["intersections"].append({
                    "intersection_id": i,
                    "north_count":     _dir_count("north"),
                    "south_count":     _dir_count("south"),
                    "east_count":      _dir_count("east"),
                    "west_count":      _dir_count("west"),
                    "queue_length":    queue,
                    "current_phase":   curr,
                    "phase_duration":  dur,
                })

            try:
                resp = requests.post(api_url, json=batch, timeout=1.0)
                if resp.status_code == 200:
                    for dec in resp.json().get("decisions", []):
                        idx = dec["intersection_id"]
                        if idx >= len(tls_ids):
                            continue
                        tls_id = tls_ids[idx]
                        target = dec["next_phase"]
                        cur    = int(traci.trafficlight.getPhase(tls_id))
                        if target == cur:
                            continue
                        if cur in {0, 2}:
                            if step - last_phase_change.get(tls_id, 0) < MIN_GREEN:
                                continue
                        # Sarı geçiş (mevcut yeşil fazdan çıkış)
                        if cur == 0 and target in (2, 3):
                            traci.trafficlight.setPhase(tls_id, 1)
                            time.sleep(0.05)
                        elif cur == 2 and target in (0, 1):
                            traci.trafficlight.setPhase(tls_id, 3)
                            time.sleep(0.05)
                        traci.trafficlight.setPhase(tls_id, target)
                        last_phase_change[tls_id] = step
            except Exception:
                pass

    except KeyboardInterrupt:
        pass
    finally:
        print("[CoSim] Temizleniyor …")
        for cam in cameras:
            try:
                cam.stop()
                cam.destroy()
            except Exception:
                pass
        for actor in actor_map.values():
            try:
                actor.destroy()
            except Exception:
                pass
        try:
            traci.close()
        except Exception:
            pass
        CARLA_STATUS.update({"connected": False, "num_vehicles": 0})
        print("[CoSim] Kapatıldı.")
