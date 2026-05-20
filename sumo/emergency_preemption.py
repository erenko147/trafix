"""
Acil Araç Preemption Denetleyicisi  (ayrı, bağımsız modül)
==========================================================

Eren'in isteği: bu mantık `run_sumo_live.py`'ın içinde gömülü DEĞİL,
ayrı bir denetim modülü olsun — SUMO sim döngüsünden olabildiğince
ayrık. Böylece ileride CARLA + YOLO entegrasyonunda SADECE bu dosyadaki
algılama/aktüatör metotları değişir; sim döngüsüne dokunulmaz.

Mimari katmanlar (CARLA'ya taşırken sadece ① ve ③ değişir):
  ① ALGILAMA   : _closest_emergency / emergency_on_edge
                  SUMO  → traci.lane.getLastStepVehicleIDs + getTypeID
                  CARLA → kamera frame + YOLO "ambulance" + ROI eşleme
  ② KARAR      : state machine (yellow→allred→green→return_yellow)
                  — DEĞİŞMEZ, taşınabilir çekirdek
  ③ AKTÜATÖR   : _set_state / _restore_program
                  SUMO  → traci.trafficlight.setRedYellowGreenState
                  CARLA → traffic_light.set_state / freeze
                  Gerçek → NTCIP preemption komutu

Ayrıca Eren'in istediği iki metrik burada toplanır:
  • Acil araç bu kavşaktan ne kadar adımda (≈sn) çıktı  (transit_steps)
  • Kaç araç ne kadar bekledi  (vehicles_waited / total_wait_steps)
Tamamlanan her olay kaydı pop_completed_sessions() ile dışarı verilir;
run_sumo_live.py bunları backend'e POST eder, dashboard gösterir.
"""

from __future__ import annotations


class EmergencyPreemptionController:
    """
    Her kavşak için bağımsız preemption durum makinesi + metrik toplayıcı.
    SUMO sim döngüsünden ayrıktır; sadece kendisine verilen `traci`
    modülü üzerinden konuşur.
    """

    # Aşamalar: None → "yellow" → "allred" → "green" → "return_yellow" → None

    def __init__(
        self,
        traci_mod,
        tls_ids,
        log,
        *,
        yellow_steps: int = 3,
        all_red_steps: int = 2,
        max_emergency_green: int = 60,
        stuck_speed: float = 0.1,
        min_green_through: int = 10,
    ):
        self.traci = traci_mod
        self.tls_ids = list(tls_ids)
        self.log = log

        self.YELLOW_STEPS = yellow_steps
        self.ALL_RED_STEPS = all_red_steps
        self.MAX_EMERGENCY_GREEN = max_emergency_green
        self.STUCK_SPEED = stuck_speed
        self.MIN_GREEN_THROUGH = min_green_through

        # Durum makinesi state'leri (eskiden run_sumo_live.main içindeydi)
        self.stage = {t: None for t in self.tls_ids}
        self.edge = {}            # tls_id → ambulansın geldiği kenar id
        self.countdown = {}       # tls_id → mevcut aşamada kalan adım
        self.stuck_cnt = {t: 0 for t in self.tls_ids}

        # Metrik: aktif oturum + tamamlanmış kayıtlar
        self._session = {}        # tls_id → aktif metrik sözlüğü
        self._completed = []      # tamamlanan olay kayıtları (dışarı verilir)

    # ── Dışarıdan sorgu ───────────────────────────────────────────────────────

    def is_active(self, tls_id: str) -> bool:
        """Bu kavşak şu an preemption altında mı? (AI/fallback atlamalı)"""
        return self.stage.get(tls_id) is not None

    def pop_completed_sessions(self) -> list:
        """Tamamlanan metrik kayıtlarını döndür ve tamponu temizle."""
        out = self._completed
        self._completed = []
        return out

    # ── ① ALGILAMA katmanı (CARLA'da burası YOLO ile değişir) ─────────────────

    def _closest_emergency(self, tls_id: str):
        """
        Kavşağa yaklaşan şeritlerde 'emergency' tipi araçlardan kavşağa
        EN YAKIN olanın (kenar_id, arac_id) ikilisini döndürür. Mesafe-
        tabanlı öncelik: birden fazla ambulans çakışırsa en yakın önce.
        Yoksa (None, None). Rota bilinmek zorunda değil — kamera gibi.

        CARLA/YOLO karşılığı: kamera frame → YOLO → 'ambulance' bbox →
        hangi lane ROI → bbox alanı (büyük = yakın) ile öncelik.
        """
        seen = set()
        try:
            links = self.traci.trafficlight.getControlledLinks(tls_id)
        except Exception:
            return None, None

        best_edge = None
        best_veh = None
        best_dist = float("inf")

        for link in links:
            if not link:
                continue
            from_lane = link[0][0]
            if from_lane in seen:
                continue
            seen.add(from_lane)

            try:
                vehicles = self.traci.lane.getLastStepVehicleIDs(from_lane)
            except Exception:
                continue
            if not vehicles:
                continue

            try:
                lane_len = self.traci.lane.getLength(from_lane)
            except Exception:
                lane_len = 0.0

            for veh_id in vehicles:
                try:
                    if self.traci.vehicle.getTypeID(veh_id) != "emergency":
                        continue
                    pos = self.traci.vehicle.getLanePosition(veh_id)
                except Exception:
                    continue
                remaining = lane_len - pos     # kavşağa kalan mesafe
                if remaining < best_dist:
                    best_dist = remaining
                    best_edge = from_lane.rsplit("_", 1)[0]
                    best_veh = veh_id

        return best_edge, best_veh

    def _emergency_on_edge(self, tls_id: str, edge: str) -> bool:
        """Belirtilen yaklaşım kenarında şu an 'emergency' araç var mı?"""
        if not edge:
            return False
        seen = set()
        try:
            links = self.traci.trafficlight.getControlledLinks(tls_id)
        except Exception:
            return False
        for link in links:
            if not link:
                continue
            from_lane = link[0][0]
            if from_lane in seen:
                continue
            seen.add(from_lane)
            if from_lane.rsplit("_", 1)[0] != edge:
                continue
            try:
                for veh_id in self.traci.lane.getLastStepVehicleIDs(from_lane):
                    if self.traci.vehicle.getTypeID(veh_id) == "emergency":
                        return True
            except Exception:
                continue
        return False

    def _emergency_speed(self, tls_id: str) -> float:
        """Kavşağa yaklaşan acil aracın anlık hızı. Bulunamazsa -1."""
        seen = set()
        try:
            links = self.traci.trafficlight.getControlledLinks(tls_id)
        except Exception:
            return -1.0
        for link in links:
            if not link:
                continue
            from_lane = link[0][0]
            if from_lane in seen:
                continue
            seen.add(from_lane)
            try:
                for veh_id in self.traci.lane.getLastStepVehicleIDs(from_lane):
                    if self.traci.vehicle.getTypeID(veh_id) == "emergency":
                        return float(self.traci.vehicle.getSpeed(veh_id))
            except Exception:
                continue
        return -1.0

    def _held_vehicle_ids(self, tls_id: str, served_edge: str):
        """
        Preemption sırasında KIRMIZIDA bekletilen (served_edge dışındaki
        tüm yaklaşımlardaki) araçların [(arac_id, hiz)] listesi. Metrik
        'kaç araba ne kadar bekledi' için kullanılır.
        """
        out = []
        seen = set()
        try:
            links = self.traci.trafficlight.getControlledLinks(tls_id)
        except Exception:
            return out
        for link in links:
            if not link:
                continue
            from_lane = link[0][0]
            if from_lane in seen:
                continue
            seen.add(from_lane)
            if from_lane.rsplit("_", 1)[0] == served_edge:
                continue   # ambulansa açık kenar — bekleyen sayılmaz
            try:
                for veh_id in self.traci.lane.getLastStepVehicleIDs(from_lane):
                    if self.traci.vehicle.getTypeID(veh_id) == "emergency":
                        continue
                    out.append((veh_id, float(self.traci.vehicle.getSpeed(veh_id))))
            except Exception:
                continue
        return out

    # ── ③ AKTÜATÖR katmanı (CARLA'da burası set_state/freeze ile değişir) ──────

    def _build_state(self, tls_id: str, approach_edge: str, on_char: str) -> str:
        """
        approach_edge'den çıkan TÜM bağlantılara on_char ('G'/'y'), geri
        kalan her şeye 'r'. Ambulans nereye dönerse dönsün o kenar açık,
        diğer her yön kapalı — rota tahmini gerekmez.
        """
        try:
            links = self.traci.trafficlight.getControlledLinks(tls_id)
        except Exception:
            return ""
        out = []
        for link in links:
            if link and link[0][0].rsplit("_", 1)[0] == approach_edge:
                out.append(on_char)
            else:
                out.append("r")
        return "".join(out)

    def _set_state(self, tls_id: str, state: str):
        try:
            if state:
                self.traci.trafficlight.setRedYellowGreenState(tls_id, state)
        except Exception:
            pass

    def _set_all_red(self, tls_id: str):
        try:
            n = len(self.traci.trafficlight.getRedYellowGreenState(tls_id))
            self.traci.trafficlight.setRedYellowGreenState(tls_id, "r" * n)
        except Exception:
            pass

    def _restore_program(self, tls_id: str):
        """
        KRİTİK: setRedYellowGreenState ışığı tek-fazlı 'online' programa
        aldığı için orijinal statik programa (id='0') geri dönmezsek
        AI'nın setPhase çağrıları işe yaramaz ve kavşak donar.
        """
        try:
            self.traci.trafficlight.setProgram(tls_id, "0")
        except Exception:
            pass

    # ── İnsan-okur etiketler (dashboard'da J0/-E7 yerine K1/Kuzey) ─────────────

    def _junction_name(self, tls_id: str) -> str:
        """SUMO tls id → dashboard kavşak adı (K1..K5). Sıralı index+1."""
        try:
            return "K" + str(self.tls_ids.index(tls_id) + 1)
        except ValueError:
            return tls_id

    def _approach_dir(self, tls_id: str, edge: str) -> str:
        """
        Ambulansın geldiği yaklaşım kenarını kavşağa göre pusula yönüne
        çevirir: Kuzey/Güney/Doğu/Batı. Kenarın şerit geometrisi ile
        kavşak konumu karşılaştırılır (run_sumo_live'daki sınıflandırma
        ile aynı mantık). Çözülemezse ham kenar id'sine düşer.
        """
        if not edge:
            return "?"
        try:
            jx, jy = self.traci.junction.getPosition(tls_id)
            shape = self.traci.lane.getShape(f"{edge}_0")
            if not shape:
                return edge
            x0, y0 = shape[0]
            dx, dy = x0 - jx, y0 - jy
            if abs(dx) > abs(dy):
                return "Batı" if dx < 0 else "Doğu"
            return "Güney" if dy < 0 else "Kuzey"
        except Exception:
            return edge

    # ── Metrik yardımcıları ───────────────────────────────────────────────────

    def _open_session(self, tls_id: str, edge: str, veh_id, step: int):
        self._session[tls_id] = {
            "tls_id": tls_id,
            "junction_name": self._junction_name(tls_id),
            "ambulance_id": veh_id if veh_id is not None else "?",
            "approach_edge": edge,
            "approach_dir": self._approach_dir(tls_id, edge),
            "start_step": step,
            "waited_ids": set(),
            "wait_step_sum": 0,
        }

    def _accumulate_wait(self, tls_id: str, step: int):
        """Her preemption adımında kırmızıda bekleyen araçları say."""
        s = self._session.get(tls_id)
        if not s:
            return
        served = self.edge.get(tls_id, "")
        for veh_id, spd in self._held_vehicle_ids(tls_id, served):
            if spd < self.STUCK_SPEED:
                s["waited_ids"].add(veh_id)
                s["wait_step_sum"] += 1

    def _close_session(self, tls_id: str, step: int, result: str):
        s = self._session.pop(tls_id, None)
        if not s:
            return
        n_waited = len(s["waited_ids"])
        record = {
            "tls_id": s["tls_id"],
            "junction_name": s["junction_name"],
            "ambulance_id": s["ambulance_id"],
            "approach_edge": s["approach_edge"],
            "approach_dir": s["approach_dir"],
            "start_step": s["start_step"],
            "end_step": step,
            "transit_steps": step - s["start_step"],
            "vehicles_waited": n_waited,
            "total_wait_steps": s["wait_step_sum"],
            "avg_wait_steps": round(s["wait_step_sum"] / n_waited, 1) if n_waited else 0.0,
            "result": result,
        }
        self._completed.append(record)
        self.log.info(
            f"  [EMERGENCY METRIC] {record['junction_name']} "
            f"({record['approach_dir']} yonu) arac={record['ambulance_id']} "
            f"gecis={record['transit_steps']} adim (≈sn) | "
            f"bekleyen={record['vehicles_waited']} arac, "
            f"toplam_bekleme={record['total_wait_steps']} adim, "
            f"ort={record['avg_wait_steps']} | sonuc={result}"
        )

    # ── ② KARAR katmanı: state machine (taşınabilir çekirdek) ─────────────────

    def update(self, tls_id, step, *,
               yellow_remaining, pending_targets, last_phase_change_step):
        """
        Bir kavşak için durum makinesini bir adım ilerletir.
        run_sumo_live.py her adımda her tls için bunu çağırır.

        yellow_remaining / pending_targets / last_phase_change_step:
        sim döngüsünün paylaşılan sözlükleri — preemption başlarken
        bekleyen sarı geçişi iptal etmek ve bitince AI'nın gecikmesiz
        devralması için gereklidir (referansla mutate edilir).
        """
        stage = self.stage[tls_id]

        # Preemption aktifse her adım bekleyen araçları say (metrik)
        if stage is not None:
            self._accumulate_wait(tls_id, step)

        if stage is None:
            edge, veh_id = self._closest_emergency(tls_id)
            if edge is not None:
                self.edge[tls_id] = edge
                self._open_session(tls_id, edge, veh_id, step)
                # Bekleyen sarı geçişini iptal et, kendi sekansımızı başlat
                yellow_remaining.pop(tls_id, None)
                pending_targets.pop(tls_id, None)
                try:
                    cur = int(self.traci.trafficlight.getPhase(tls_id))
                    cur_green = cur if cur % 2 == 0 else cur - 1
                    self.traci.trafficlight.setPhase(tls_id, cur_green + 1)
                except Exception:
                    pass
                self.stage[tls_id] = "yellow"
                self.countdown[tls_id] = self.YELLOW_STEPS
                self.log.info(
                    f"  [EMERGENCY] step={step} {tls_id} — ambulans "
                    f"'{edge}' yolunda tespit edildi, sarıya geçiliyor"
                )

        elif stage == "yellow":
            self.countdown[tls_id] -= 1
            if self.countdown[tls_id] <= 0:
                self._set_all_red(tls_id)
                self.stage[tls_id] = "allred"
                self.countdown[tls_id] = self.ALL_RED_STEPS
                self.log.info(
                    f"  [EMERGENCY] step={step} {tls_id} — "
                    f"tüm yönler kırmızı, kavşak boşaltılıyor"
                )

        elif stage == "allred":
            self.countdown[tls_id] -= 1
            if self.countdown[tls_id] <= 0:
                self._set_state(
                    tls_id, self._build_state(tls_id, self.edge[tls_id], "G")
                )
                self.stage[tls_id] = "green"
                last_phase_change_step[tls_id] = step
                self.log.info(
                    f"  [EMERGENCY GREEN] step={step} {tls_id} — "
                    f"'{self.edge[tls_id]}' yolunun tüm çıkışları yeşil, "
                    f"araç geçiyor"
                )

        elif stage == "green":
            green_dur = step - last_phase_change_step.get(tls_id, step)
            edge, _ = self._closest_emergency(tls_id)
            served = self.edge.get(tls_id, "")
            still_on_served = self._emergency_on_edge(tls_id, served)

            if edge is None:
                # Hiçbir yaklaşımda acil araç yok → normal moda dön
                self._set_state(tls_id, self._build_state(tls_id, served, "y"))
                self._close_session(tls_id, step, "cleared")
                self.stage[tls_id] = "return_yellow"
                self.countdown[tls_id] = self.YELLOW_STEPS
                self.log.info(
                    f"  [EMERGENCY CLEAR] step={step} {tls_id} — "
                    f"araç geçti, normal moda dönülüyor"
                )

            elif (not still_on_served) and edge != served:
                # Hizmet edilen araç geçti ama BAŞKA yaklaşımda ikinci
                # ambulans var. Önceki aracın metriğini kapat, yenisini aç.
                self._close_session(tls_id, step, "cleared")
                new_edge, new_veh = self._closest_emergency(tls_id)
                if new_edge is None:
                    new_edge, new_veh = edge, None
                self.edge[tls_id] = new_edge
                self.stuck_cnt[tls_id] = 0
                self._open_session(tls_id, new_edge, new_veh, step)
                self._set_all_red(tls_id)
                self.stage[tls_id] = "allred"
                self.countdown[tls_id] = self.ALL_RED_STEPS
                self.log.info(
                    f"  [EMERGENCY NEXT] step={step} {tls_id} — önceki "
                    f"araç geçti, '{new_edge}' yolundaki ikinci ambulansa "
                    f"geçiliyor"
                )

            elif green_dur > self.MAX_EMERGENCY_GREEN:
                # Zaman aşımı — araç sıkıştı, kavşağı serbest bırak
                self._set_state(
                    tls_id, self._build_state(tls_id, served, "y")
                )
                self._close_session(tls_id, step, "timeout")
                self.stage[tls_id] = "return_yellow"
                self.countdown[tls_id] = self.YELLOW_STEPS
                self.log.info(
                    f"  [EMERGENCY TIMEOUT] step={step} {tls_id} — "
                    f"{green_dur} adım geçti, araç sıkıştı, AI geri alındı"
                )

            else:
                # Yeşili her adım yeniden uygula (static program üzerine
                # yazmasın), araca dokunma
                self._set_state(tls_id, self._build_state(tls_id, served, "G"))
                spd = self._emergency_speed(tls_id)
                if 0.0 <= spd < self.STUCK_SPEED:
                    self.stuck_cnt[tls_id] += 1
                    if self.stuck_cnt[tls_id] % 10 == 1:
                        self.log.info(
                            f"  [EMERGENCY WAIT] step={step} {tls_id} — "
                            f"araç önündeki trafiği bekliyor "
                            f"(spd={spd:.2f} m/s, "
                            f"stuck={self.stuck_cnt[tls_id]} adım)"
                        )
                else:
                    self.stuck_cnt[tls_id] = 0

        elif stage == "return_yellow":
            self.countdown[tls_id] -= 1
            if self.countdown[tls_id] <= 0:
                self._restore_program(tls_id)
                # min_green'i hemen sağ kabul et ki AI gecikmeden devralsın
                last_phase_change_step[tls_id] = step - self.MIN_GREEN_THROUGH
                self.stage[tls_id] = None
                self.edge.pop(tls_id, None)
                self.countdown.pop(tls_id, None)
                self.stuck_cnt[tls_id] = 0
                self.log.info(
                    f"  [EMERGENCY END] step={step} {tls_id} — orijinal "
                    f"program geri yüklendi, AI kontrolü geri alındı"
                )

        return self.stage[tls_id] is not None
