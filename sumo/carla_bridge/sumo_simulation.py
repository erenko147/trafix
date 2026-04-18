"""
SUMO Simulasyon Yoneticisi
===========================
TraCI uzerinden SUMO'yu baslatir, adimlar, arac/TLS durumlarini okur.
"""

import math
import os
import sys
import warnings
import xml.etree.ElementTree as ET

SUMO_HOME = os.environ.get("SUMO_HOME")
if SUMO_HOME:
    sys.path.append(os.path.join(SUMO_HOME, "tools"))
else:
    sys.exit("HATA: SUMO_HOME ortam degiskeni tanimlanmamis!")

import traci
warnings.filterwarnings("ignore", category=UserWarning, module="traci")


def _offset_polyline(shape, offset):
    """
    Serit merkez cizgisine dik ofset (SUMO duzlem).
    offset > 0: sol kenar (ileri yon solu), < 0: sag kenar.
    """
    n = len(shape)
    if n < 2:
        return []
    out = []
    for i in range(n):
        if i == 0:
            dx = shape[1][0] - shape[0][0]
            dy = shape[1][1] - shape[0][1]
        elif i == n - 1:
            dx = shape[i][0] - shape[i - 1][0]
            dy = shape[i][1] - shape[i - 1][1]
        else:
            dx = shape[i + 1][0] - shape[i - 1][0]
            dy = shape[i + 1][1] - shape[i - 1][1]
        L = math.hypot(dx, dy)
        if L < 1e-8:
            L = 1e-8
        nx = -dy / L
        ny = dx / L
        out.append(
            (float(shape[i][0] + nx * offset), float(shape[i][1] + ny * offset))
        )
    return out


def _parse_net_offset(net_xml_path):
    """net.xml'deki netOffset degerini parse eder → (ox, oy)."""
    try:
        root = ET.parse(net_xml_path).getroot()
        loc = root.find("location")
        if loc is not None:
            ox, oy = [float(v) for v in loc.get("netOffset", "0,0").split(",")]
            return (ox, oy)
    except Exception:
        pass
    return (0.0, 0.0)


def _find_net_xml(cfg_path):
    """sumocfg icinden net-file degerini bul, tam yolu don."""
    cfg_dir = os.path.dirname(cfg_path)
    try:
        root = ET.parse(cfg_path).getroot()
        for tag in ("net-file", "net_file"):
            el = root.find(f".//{tag}")
            if el is not None:
                val = el.get("value", "")
                if val:
                    p = os.path.join(cfg_dir, val)
                    if os.path.exists(p):
                        return p
    except Exception:
        pass
    # fallback: klasorde .net.xml ara
    for f in os.listdir(cfg_dir):
        if f.endswith(".net.xml"):
            return os.path.join(cfg_dir, f)
    return None


class SumoSimulation:
    def __init__(self, cfg_path, step_length=0.05, use_gui=False):
        self.cfg_path    = cfg_path
        self.step_length = step_length
        self.use_gui     = use_gui
        self.net_offset  = (0.0, 0.0)
        self._prev_vehicles = set()

    def start(self):
        if not os.path.exists(self.cfg_path):
            sys.exit(f"HATA: SUMO config bulunamadi: {self.cfg_path}")

        net_xml = _find_net_xml(self.cfg_path)
        if net_xml:
            self.net_offset = _parse_net_offset(net_xml)
            print(f"[SUMO] Net offset: {self.net_offset}")

        exe = "sumo-gui" if self.use_gui else "sumo"
        print(f"[SUMO] Baslatiliyor: {exe} -c {self.cfg_path}")
        traci.start([
            exe, "-c", self.cfg_path,
            "--no-warnings",
            "--step-length", str(self.step_length),
            "--collision.action", "warn",
        ])
        self._prev_vehicles = set()
        print("[SUMO] TraCI baglantisi kuruldu.")

    def step(self):
        traci.simulationStep()

    def get_sim_time(self):
        return traci.simulation.getTime()

    def get_min_expected(self):
        try:
            return traci.simulation.getMinExpectedNumber()
        except Exception:
            return 0

    def get_tls_ids(self):
        return sorted(traci.trafficlight.getIDList())

    # ── Arac durumu ──────────────────────────────────────────────────────────

    def get_vehicle_states(self):
        """Aktif tum araclarin anlık durumunu don: {vid: {...}}"""
        states = {}
        for vid in traci.vehicle.getIDList():
            try:
                states[vid] = {
                    "pos":    traci.vehicle.getPosition(vid),
                    "angle":  traci.vehicle.getAngle(vid),
                    "speed":  traci.vehicle.getSpeed(vid),
                    "vclass": traci.vehicle.getVehicleClass(vid),
                    "color":  ",".join(
                        str(v) for v in traci.vehicle.getColor(vid)[:3]
                    ),
                }
            except Exception:
                pass
        return states

    def compute_diff(self, current_ids: set):
        """Onceki adima gore spawned/destroyed araclari hesapla."""
        spawned   = current_ids - self._prev_vehicles
        destroyed = self._prev_vehicles - current_ids
        self._prev_vehicles = current_ids
        return spawned, destroyed

    # ── TLS ─────────────────────────────────────────────────────────────────

    def get_tls_states(self):
        """Tum TLS'lerin guncel faz ve durum stringini don."""
        states = {}
        for tid in self.get_tls_ids():
            try:
                states[tid] = {
                    "phase": int(traci.trafficlight.getPhase(tid)),
                    "state": traci.trafficlight.getRedYellowGreenState(tid),
                }
            except Exception:
                pass
        return states

    def get_tls_phase_info(self, tls_id):
        """(current_phase, elapsed_seconds) don."""
        try:
            cp  = int(traci.trafficlight.getPhase(tls_id))
            pd  = traci.trafficlight.getPhaseDuration(tls_id)
            ns  = traci.trafficlight.getNextSwitch(tls_id)
            sim = traci.simulation.getTime()
            elapsed = pd - max(0.0, ns - sim)
            return cp, max(0.0, elapsed)
        except Exception:
            return 0, 0.0

    def set_tls_phase(self, tls_id, phase):
        try:
            traci.trafficlight.setPhase(tls_id, phase)
        except Exception:
            pass

    # ── Kenar/kavşak sorgulari ───────────────────────────────────────────────

    def get_controlled_links(self, tls_id):
        try:
            return traci.trafficlight.getControlledLinks(tls_id)
        except Exception:
            return []

    def get_tls_lane_positions(self, tls_id):
        """
        getRedYellowGreenState ile ayni sirada: kontrol edilen seritlerin
        SUMO duzlemde yaklasim noktasi (serit geometrisinin ortasi).
        """
        try:
            lanes = traci.trafficlight.getControlledLanes(tls_id)
        except Exception:
            return []
        out = []
        for lane_id in lanes:
            try:
                shape = traci.lane.getShape(lane_id)
                if shape:
                    mid = shape[len(shape) // 2]
                    out.append((float(mid[0]), float(mid[1])))
                else:
                    out.append((0.0, 0.0))
            except Exception:
                out.append((0.0, 0.0))
        return out

    def get_junction_position(self, tls_id):
        try:
            return traci.junction.getPosition(tls_id)
        except Exception:
            return (0.0, 0.0)

    def get_lane_shape(self, lane_id):
        try:
            return traci.lane.getShape(lane_id)
        except Exception:
            return []

    def get_edge_vehicle_count(self, edge_id):
        if not edge_id:
            return 0
        try:
            return int(traci.edge.getLastStepVehicleNumber(edge_id))
        except Exception:
            return 0

    def get_edge_waiting_time(self, edge_id):
        if not edge_id:
            return 0.0
        try:
            return float(traci.edge.getWaitingTime(edge_id))
        except Exception:
            return 0.0

    def build_lane_edge_polylines(self):
        """
        SUMO agindan serit isaretleme primitifleri uret (tek yonlu edge bazinda):
        - sag dis kenar (sag serit): beyaz solid
        - ayni yondeki seritler arasi: beyaz kesikli
        - sol dis kenar (sol serit, karsi trafik tarafi): sari solid — yol ortasi
        Ayni yondeki seritlerin geometrik ortasina sari cizilmez.
        """
        markings = []
        try:
            edge_ids = traci.edge.getIDList()
        except Exception:
            return []
        for eid in edge_ids:
            if eid.startswith(":"):
                continue
            try:
                lane_count = int(traci.edge.getLaneNumber(eid))
            except Exception:
                continue
            if lane_count <= 0:
                continue

            lane_shapes = []
            lane_widths = []
            for i in range(lane_count):
                lid = f"{eid}_{i}"
                try:
                    shp = traci.lane.getShape(lid)
                    if len(shp) < 2:
                        continue
                    lane_shapes.append(shp)
                    lane_widths.append(float(traci.lane.getWidth(lid)))
                except Exception:
                    continue

            if not lane_shapes:
                continue

            # Dis sag kenar (yol kenari)
            right_outer = _offset_polyline(lane_shapes[0], -lane_widths[0] * 0.5)
            if right_outer:
                markings.append(
                    {"shape": right_outer, "color": "white", "style": "solid", "width": 0.055}
                )

            # Ayni yone giden seritler arasi: kesikli beyaz
            for i in range(len(lane_shapes) - 1):
                sep = _offset_polyline(lane_shapes[i], lane_widths[i] * 0.5)
                if sep:
                    markings.append(
                        {"shape": sep, "color": "white", "style": "dashed", "width": 0.045}
                    )

            # Sol dis kenar = karsi trafikle ayrim (yol ortasi), sari solid
            left_outer = _offset_polyline(
                lane_shapes[-1], lane_widths[len(lane_shapes) - 1] * 0.5
            )
            if left_outer:
                markings.append(
                    {"shape": left_outer, "color": "yellow", "style": "solid", "width": 0.06}
                )

        return markings

    def close(self):
        try:
            traci.close()
            print("[SUMO] TraCI baglantisi kapatildi.")
        except Exception:
            pass
