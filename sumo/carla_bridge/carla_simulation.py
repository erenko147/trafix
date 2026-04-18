"""
CARLA Simulasyon Yoneticisi
============================
CARLA sunucusuna baglanir, senkron modu aktif eder.
SUMO arac durumlarini CARLA actor'lerine yansitir.

opendrive_file verilirse, kullanicinin ozel SUMO haritasindan uretilen .xodr
dosyasini CARLA'ya generate_opendrive_world() ile yukler.
"""

import math
import carla
from .bridge_helper import (
    get_blueprint,
    get_carla_transform,
    sumo_color_to_carla,
    TLS_STATE_MAP,
)

# generate_opendrive_world() parametreleri
# wall_height=0 → yol kenarinda SUMO'daki gibi serit cizgisi kalir; duvar/bariyer olusmaz
_OPENDRIVE_PARAMS = carla.OpendriveGenerationParameters(
    vertex_distance=0.5,
    max_road_length=500.0,
    wall_height=0.0,
    additional_width=0.6,
    smooth_junctions=True,
    enable_mesh_visibility=True,
)

# TLS yon eslestirmesi icin kavşak arama yarıcapı (metre)
_TLS_MATCH_RADIUS = 35.0


class CarlaSimulation:
    def __init__(
        self,
        host="localhost",
        port=2000,
        step_length=0.05,
        sync_vehicle_color=True,
        sync_vehicle_lights=False,
        opendrive_file=None,
    ):
        self.host                = host
        self.port                = port
        self.step_length         = step_length
        self.sync_vehicle_color  = sync_vehicle_color
        self.sync_vehicle_lights = sync_vehicle_lights
        self.opendrive_file      = opendrive_file

        self.client     = None
        self.world      = None
        self.bp_lib     = None
        self._actor_map = {}          # sumo_id → carla.Actor
        self._prev_xy   = {}          # sumo_id → (x, y) CARLA duzlemi
        # sumo_tls_id → [(carla.TrafficLight, link_idx), ...]
        self._tls_map   = {}

    # ── Baglanti / Kapat ─────────────────────────────────────────────────────

    def start(self):
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(120.0)

        if self.opendrive_file:
            self._load_opendrive_map()
        else:
            self.world = self.client.get_world()

        self.bp_lib = self.world.get_blueprint_library()

        settings = self.world.get_settings()
        settings.synchronous_mode       = True
        settings.fixed_delta_seconds    = self.step_length
        settings.substepping            = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps           = 10
        self.world.apply_settings(settings)

        map_name = self.world.get_map().name
        print(
            f"[CARLA] Baglandi: {self.host}:{self.port} | "
            f"Harita: {map_name} | "
            f"Adim: {self.step_length}s | Senkron: AKTIF"
        )

    def _load_opendrive_map(self):
        """Kullanicinin ozel haritasini .xodr olarak CARLA'ya yukle."""
        import os
        if not os.path.exists(self.opendrive_file):
            raise FileNotFoundError(
                f"[CARLA] OpenDRIVE dosyasi bulunamadi: {self.opendrive_file}\n"
                f"  → once 'python baslat.py --generate-map' calistirin"
            )
        with open(self.opendrive_file, "r", encoding="utf-8") as f:
            xodr_content = f.read()

        print(f"[CARLA] Ozel harita yukleniyor: {self.opendrive_file}")
        print("[CARLA] generate_opendrive_world() cagrildi — 10-30sn surebilir...")
        self.world = self.client.generate_opendrive_world(xodr_content, _OPENDRIVE_PARAMS)
        print("[CARLA] Harita basariyla yuklendi.")

    def tick(self):
        self.world.tick()

    def cleanup(self):
        print(f"[CARLA] {len(self._actor_map)} aktor siliniyor...")
        for actor in list(self._actor_map.values()):
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        self._actor_map.clear()

        if self.world is not None:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
            except Exception:
                pass
        print("[CARLA] Temizlendi, senkron mod kapatildi.")

    # ── TLS eşleştirme (kavşak konumuna gore) ────────────────────────────────

    def build_tls_mapping(
        self,
        sumo_junction_positions: dict,
        tls_lane_positions: dict,
        sumo_offset=(0.0, 0.0),
    ):
        """
        SUMO TLS ID'lerini, kavşak koordinatlarına en yakin CARLA trafik
        lambasi aktörlerine eslestirir; her lamba icin RYG state indeksi atanir.

        sumo_junction_positions: {tls_id: (sumo_x, sumo_y)}
        tls_lane_positions:      {tls_id: [(lx, ly), ...]} — getControlledLanes sirasi
        sumo_offset:             net.xml netOffset degeri
        """
        carla_tls = list(self.world.get_actors().filter("traffic.traffic_light"))
        if not carla_tls:
            print("[CARLA] UYARI: CARLA dunyasinda trafik lambasi aktoru bulunamadi.")
            return

        self._tls_map = {}
        unmatched = []

        for tid, (sx, sy) in sumo_junction_positions.items():
            # SUMO → CARLA koordinat donusumu (kavşak merkezi)
            cx = sx - sumo_offset[0]
            cy = -(sy - sumo_offset[1])

            lane_pts = tls_lane_positions.get(tid) or []
            carla_lane_xy = [
                (lx - sumo_offset[0], -(ly - sumo_offset[1]))
                for lx, ly in lane_pts
            ]

            # Yaricap icindeki tum CARLA TL'leri topla
            group = [
                tl for tl in carla_tls
                if math.hypot(
                    tl.get_transform().location.x - cx,
                    tl.get_transform().location.y - cy,
                ) < _TLS_MATCH_RADIUS
            ]

            if not group:
                # Fallback: en yakin tek lamba
                nearest = min(
                    carla_tls,
                    key=lambda tl: math.hypot(
                        tl.get_transform().location.x - cx,
                        tl.get_transform().location.y - cy,
                    ),
                )
                group = [nearest]
                unmatched.append(tid)

            paired = []
            for tl in group:
                loc = tl.get_transform().location
                if carla_lane_xy:
                    best_i = min(
                        range(len(carla_lane_xy)),
                        key=lambda i: math.hypot(
                            loc.x - carla_lane_xy[i][0],
                            loc.y - carla_lane_xy[i][1],
                        ),
                    )
                else:
                    best_i = 0
                paired.append((tl, best_i))

            self._tls_map[tid] = paired

        print(
            f"[CARLA] TLS mapping: {len(self._tls_map)} SUMO kavşak → "
            f"{sum(len(v) for v in self._tls_map.values())} CARLA lamba | "
            f"fallback: {len(unmatched)}"
        )

    def _vehicle_z_on_road(self, x, y):
        """
        (x,y) konumunu yola projekte ederek aracin taban yuksekligini dondur.
        Sabit z=0.5 havada kalma yapar; SUMO ile hizali yol yuzeyi icin waypoint kullanilir.
        """
        try:
            carla_map = self.world.get_map()
            loc = carla.Location(x=x, y=y, z=8.0)
            wp = carla_map.get_waypoint(
                loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if wp is None:
                wp = carla_map.get_waypoint(
                    loc,
                    project_to_road=True,
                    lane_type=carla.LaneType.Any,
                )
            if wp:
                return float(wp.transform.location.z) + 0.08
        except Exception:
            pass
        return 0.08

    # ── Arac yonetimi ────────────────────────────────────────────────────────

    def spawn_actor(self, sumo_id, vehicle_state, sumo_offset):
        bp = get_blueprint(self.bp_lib, vehicle_state["vclass"])
        if bp is None:
            return None

        if self.sync_vehicle_color and bp.has_attribute("color"):
            c = sumo_color_to_carla(vehicle_state["color"])
            bp.set_attribute("color", f"{c.r},{c.g},{c.b}")

        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", f"sumo_{sumo_id}")

        transform = get_carla_transform(
            vehicle_state["pos"],
            vehicle_state["angle"],
            sumo_offset,
            z=0.0,
        )
        transform.location.z = self._vehicle_z_on_road(
            transform.location.x, transform.location.y
        )

        actor = self.world.try_spawn_actor(bp, transform)
        if actor is not None:
            actor.set_simulate_physics(False)
            self._actor_map[sumo_id] = actor
            self._prev_xy[sumo_id] = (transform.location.x, transform.location.y)
        return actor

    def update_actor(self, sumo_id, vehicle_state, sumo_offset):
        actor = self._actor_map.get(sumo_id)
        if actor is None or not actor.is_alive:
            self._actor_map.pop(sumo_id, None)
            return
        transform = get_carla_transform(
            vehicle_state["pos"],
            vehicle_state["angle"],
            sumo_offset,
            z=0.0,
        )
        transform.location.z = self._vehicle_z_on_road(
            transform.location.x, transform.location.y
        )
        # Donuslerde yon: hareket vektorunden; kirmizi isikta durunca konum neredeyse
        # sabit kalir — o zaman SUMO acisina dusmek ani ters donmeye yol acar.
        # Dusuk hiz veya cok kucuk adimda onceki CARLA yaw'unu koru.
        x = transform.location.x
        y = transform.location.y
        prev = self._prev_xy.get(sumo_id)
        speed = abs(float(vehicle_state.get("speed", 0.0)))
        move_eps = 0.05
        min_move_speed = 0.12  # m/s — altinda "duruyor" say, yaw'u koru

        if prev is not None:
            dx = x - prev[0]
            dy = y - prev[1]
            moved = math.hypot(dx, dy) > move_eps
            if moved and speed > min_move_speed:
                transform.rotation.yaw = math.degrees(math.atan2(dy, dx))
            else:
                try:
                    transform.rotation.yaw = actor.get_transform().rotation.yaw
                except Exception:
                    pass

        actor.set_transform(transform)
        self._prev_xy[sumo_id] = (x, y)

    def destroy_actor(self, sumo_id):
        actor = self._actor_map.pop(sumo_id, None)
        self._prev_xy.pop(sumo_id, None)
        if actor is not None and actor.is_alive:
            actor.destroy()

    # ── TLS senkronizasyonu ──────────────────────────────────────────────────

    def sync_traffic_lights(self, tls_states: dict):
        """
        SUMO TLS durumlarini, kavşak bazli CARLA trafik lambasi aktörlerine uygula.
        build_tls_mapping() onceden cagrilmis olmalidir.
        """
        if not tls_states:
            return

        if self._tls_map:
            # Kavşak eslestirilmis — her lamba SUMO link indeksine gore
            for tid, state in tls_states.items():
                paired = self._tls_map.get(tid)
                if not paired:
                    continue
                state_str = state.get("state", "r") or "r"
                for tl, link_idx in paired:
                    if link_idx < len(state_str):
                        ch = state_str[link_idx]
                    else:
                        ch = state_str[0] if state_str else "r"
                    carla_state = TLS_STATE_MAP.get(
                        ch, carla.TrafficLightState.Red
                    )
                    try:
                        tl.set_state(carla_state)
                        tl.freeze(True)
                    except Exception:
                        pass
        else:
            # Mapping henuz olusturulmamis — round-robin fallback
            carla_tls = list(self.world.get_actors().filter("traffic.traffic_light"))
            if not carla_tls:
                return
            tls_list = list(tls_states.values())
            for i, carla_light in enumerate(carla_tls):
                state_str     = tls_list[i % len(tls_list)].get("state", "r")
                dominant_char = state_str[0] if state_str else "r"
                carla_state   = TLS_STATE_MAP.get(dominant_char, carla.TrafficLightState.Red)
                try:
                    carla_light.set_state(carla_state)
                    carla_light.freeze(True)
                except Exception:
                    pass

    # ── Yardimcilar ──────────────────────────────────────────────────────────

    def get_actor_count(self):
        return len(self._actor_map)

    def draw_lane_markings_debug(
        self,
        lane_markings,
        sumo_offset,
        step_length,
        z=None,
    ):
        """
        CARLA generate_opendrive_world() cogu surumde roadMark mesh'i cizmiyor.
        SUMO'dan uretilen serit isaretlerini world.debug ile cizer.
        """
        if not lane_markings or self.world is None:
            return
        debug = self.world.debug
        ox, oy = sumo_offset[0], sumo_offset[1]
        lt = max(float(step_length) * 4.0, 0.25)
        # Debug cizgileri dogasi geregi parlak; renkleri mumkun oldugunca dusuk tut
        white = carla.Color(68, 68, 65)
        yellow = carla.Color(78, 64, 22)

        for mark in lane_markings:
            shape = mark.get("shape", [])
            if len(shape) < 2:
                continue
            style = mark.get("style", "solid")
            col = yellow if mark.get("color") == "yellow" else white
            thickness = float(mark.get("width", 0.06))
            for i in range(len(shape) - 1):
                ax, ay = shape[i]
                bx, by = shape[i + 1]
                xa = ax - ox
                ya = -(ay - oy)
                xb = bx - ox
                yb = -(by - oy)
                if z is None:
                    zline = self._vehicle_z_on_road(0.5 * (xa + xb), 0.5 * (ya + yb)) + 0.02
                else:
                    zline = z
                try:
                    if style == "dashed":
                        dx = xb - xa
                        dy = yb - ya
                        seg_len = math.hypot(dx, dy)
                        if seg_len < 1e-3:
                            continue
                        ux = dx / seg_len
                        uy = dy / seg_len
                        dash = 2.0
                        gap = 2.8
                        s = 0.0
                        while s < seg_len:
                            e = min(s + dash, seg_len)
                            sx = xa + ux * s
                            sy = ya + uy * s
                            ex = xa + ux * e
                            ey = ya + uy * e
                            debug.draw_line(
                                carla.Location(x=sx, y=sy, z=zline),
                                carla.Location(x=ex, y=ey, z=zline),
                                thickness=thickness,
                                life_time=lt,
                                color=col,
                            )
                            s += dash + gap
                    else:
                        debug.draw_line(
                            carla.Location(x=xa, y=ya, z=zline),
                            carla.Location(x=xb, y=yb, z=zline),
                            thickness=thickness,
                            life_time=lt,
                            color=col,
                        )
                except Exception:
                    pass
