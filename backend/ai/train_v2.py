"""
TraFix — shared SUMO training infrastructure
=============================================
Provides `TrainConfig` and `SumoEnvironment` (the TraCI wrapper) plus the
model↔SUMO phase maps. These are imported by every v6 training/eval script
(stage1/2/3, finetune, eval). The legacy v2 training loop (`train()`) and the
`CoordinatedPPOAgent` model it drove were removed — only v6 is trained now.

Requires SUMO installed with SUMO_HOME set, and torch / torch-geometric.
"""

import os
import sys
import json
import time
import math
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# ── SUMO TraCI ──
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    # Varsayılan Windows / Linux yolları
    for candidate in [
        "C:\\Program Files (x86)\\Eclipse\\Sumo\\tools",
        "C:\\Program Files\\Eclipse\\Sumo\\tools",
        "/usr/share/sumo/tools",
        "/usr/local/share/sumo/tools",
    ]:
        if os.path.isdir(candidate):
            sys.path.append(candidate)
            break

try:
    import traci
    import sumolib
except ImportError:
    raise ImportError(
        "SUMO TraCI bulunamadı. SUMO_HOME ortam değişkenini ayarlayın:\n"
        "  set SUMO_HOME=C:\\Program Files (x86)\\Eclipse\\Sumo"
    )

# ── Dinamik Trafik Talebi ──
# sumo/generate_demand.py her episode'da farklı yoğunlukta trafik üretir.
# Gridlock eğitimini önlemek için import edilir; bulunamazsa static dosya kullanılır.
try:
    from sumo.generate_demand import generate_dynamic_demand as _gen_demand
    _HAS_DYNAMIC_DEMAND = True
except ImportError:
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sumo"))
        from generate_demand import generate_dynamic_demand as _gen_demand
        _HAS_DYNAMIC_DEMAND = True
    except ImportError:
        _HAS_DYNAMIC_DEMAND = False

# ── Model import ──
try:
    from backend.ai.trafix_v2 import (
        parse_sumo_observations,
        compute_reward,
        compute_gae,
        RewardWeights,
        NUM_NODE_FEATURES,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from trafix_v2 import (
        parse_sumo_observations,
        compute_reward,
        compute_gae,
        RewardWeights,
        NUM_NODE_FEATURES,
    )


# ══════════════════════════════════════════════════
#  Eğitim Konfigürasyonu
# ══════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """Tüm hiperparametreler tek yerde."""

    # ── Dosya Yolları ──
    sumo_cfg: str = "sumo/training.sumocfg"
    net_file: str = "sumo/map.net.xml"
    output_dir: str = "training_outputs"
    checkpoint_path: str = "coordinated_agent_weights.pth"

    # ── Eğitim ──
    episodes: int = 500
    max_steps_per_episode: int = 3600      # 1 saat simülasyon (1 adım = 1 sn)
    decision_interval: int = 10            # her 10 sn'de bir karar
    warmup_steps: int = 50                 # simülasyon ısınma adımı
    ppo_epochs: int = 4                    # her rollout için PPO güncelleme sayısı
    rollout_length: int = 64               # adım toplama uzunluğu

    # ── Model ──
    hidden_dim: int = 128
    num_actions: int = 4
    num_heads: int = 4

    # ── Optimizasyon ──
    lr: float = 3e-4
    lr_min: float = 1e-5
    eps: float = 1e-5
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # ── PPO ──
    clip_eps: float = 0.2
    entropy_coef: float = 0.005
    entropy_coef_min: float = 0.001
    entropy_decay: float = 0.9998
    value_coef: float = 0.25

    # ── Ödül ──
    reward_weights: RewardWeights = field(default_factory=RewardWeights)

    # ── Kayıt ──
    save_interval: int = 25                # her N episode'da checkpoint
    log_interval: int = 5                  # her N episode'da detaylı log
    eval_interval: int = 50                # her N episode'da değerlendirme

    # ── SUMO ──
    gui: bool = False
    sumo_step_length: float = 1.0
    seed: int = 42

    # ── Resume ──
    resume: bool = False


# ══════════════════════════════════════════════════
#  SUMO Ortam Arayüzü
# ══════════════════════════════════════════════════

MODEL_TO_SUMO_GREEN = {0: 0, 1: 2, 2: 4, 3: 6, 4: 8, 5: 10}
SUMO_TO_MODEL_PHASE = {0:0,1:0, 2:1,3:1, 4:2,5:2, 6:3,7:3, 8:4,9:4, 10:5,11:5}
LANE_TYPE = {0: "right", 1: "through", 2: "left"}


class SumoEnvironment:
    """
    SUMO simülasyonunu TraCI ile yönetir.
    Her adımda kavşak gözlemlerini toplar ve faz değişikliklerini uygular.
    """

    YELLOW_STEPS = 3   # 3-second yellow transition (matches run_sumo_live.py)

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.tls_ids: List[str] = []       # trafik ışığı ID'leri
        self.num_nodes = 0
        self._step_count = 0
        self._episode_count = 0
        self._pending_target: Dict[str, int] = {}        # tls_id → target green phase
        self._yellow_steps_remaining: Dict[str, int] = {}  # tls_id → steps until green
        self._phase_held_since: Dict[str, int] = {}      # tls_id → step when current green started

    # ── SUMO Başlat / Kapat ──────────────────────

    def start(self, episode: int = 0):
        """Yeni SUMO oturumu başlat."""
        # ── Dosya varlık kontrolü ──
        cfg_path = os.path.abspath(self.cfg.sumo_cfg)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"\n{'=' * 60}\n"
                f"  SUMO konfigürasyon dosyası bulunamadı!\n"
                f"  Aranan yol : {cfg_path}\n"
                f"  Çalışma dizini: {os.getcwd()}\n"
                f"{'=' * 60}\n"
                f"  Çözüm seçenekleri:\n"
                f"  1. Doğru klasörden çalıştırın:\n"
                f"       cd <proje_kök_dizini>\n"
                f"       python -m backend.ai.train_v2\n"
                f"  2. Yolu açıkça belirtin:\n"
                f"       python train_v2.py --sumo-cfg sumo/training.sumocfg\n"
                f"{'=' * 60}"
            )

        sumo_binary = "sumo-gui" if self.cfg.gui else "sumo"

        sumo_cmd = [
            sumo_binary,
            "-c", cfg_path,
            "--step-length", str(self.cfg.sumo_step_length),
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1",   # disable teleportation — forces model to actually serve all lanes
            "--no-warnings", "true",
            "--random",
            "--seed", str(self.cfg.seed + episode),
        ]

        logging.info(f"  SUMO başlatılıyor: {cfg_path}")
        traci.start(sumo_cmd)
        self._step_count = 0
        self._episode_count = episode
        self._pending_target = {}
        self._yellow_steps_remaining = {}
        self._phase_held_since = {}

        # Trafik ışığı ID'lerini al
        self.tls_ids = sorted(traci.trafficlight.getIDList())
        self.num_nodes = len(self.tls_ids)

        if self.num_nodes == 0:
            raise RuntimeError(
                "SUMO ağında trafik ışığı bulunamadı! "
                "map.net.xml dosyasında <tlLogic> tanımlı olduğundan emin olun."
            )

        # Isınma — simülasyona araç girsin
        for _ in range(self.cfg.warmup_steps):
            traci.simulationStep()
            self._step_count += 1

        # Initialise duration tracker after warmup so elapsed starts at 0
        self._phase_held_since = {tls_id: self._step_count for tls_id in self.tls_ids}

    def close(self):
        """SUMO oturumunu kapat."""
        try:
            traci.close()
        except Exception:
            pass

    @property
    def is_running(self) -> bool:
        """Simülasyon hâlâ çalışıyor mu."""
        try:
            return traci.simulation.getMinExpectedNumber() > 0
        except Exception:
            return False

    # ── Gözlem Toplama ────────────────────────────

    def get_observations(self) -> List[Dict]:
        """
        Collect per-lane vehicle counts for all 12 lanes per junction.
        Output format matches trafix_v2.parse_sumo_observations (20-dim).
        """
        observations = []

        for idx, tls_id in enumerate(self.tls_ids):
            jx, jy = traci.junction.getPosition(tls_id)

            counts = {
                "north_left": 0, "north_through": 0, "north_right": 0,
                "south_left": 0, "south_through": 0, "south_right": 0,
                "east_left":  0, "east_through":  0, "east_right":  0,
                "west_left":  0, "west_through":  0, "west_right":  0,
            }

            controlled_links = traci.trafficlight.getControlledLinks(tls_id)
            seen_lanes = set()

            for link in controlled_links:
                if not link:
                    continue
                from_lane = link[0][0]
                if from_lane in seen_lanes:
                    continue
                seen_lanes.add(from_lane)

                edge_id = from_lane.rsplit("_", 1)[0]
                lane_idx = int(from_lane.rsplit("_", 1)[1])
                lane_type = LANE_TYPE.get(lane_idx, "through")
                direction = self._classify_edge_direction(edge_id, jx, jy)

                if direction and lane_type:
                    key = f"{direction}_{lane_type}"
                    if key in counts:
                        counts[key] += traci.lane.getLastStepVehicleNumber(from_lane)

            # Phase info
            sumo_phase = traci.trafficlight.getPhase(tls_id)
            model_phase = SUMO_TO_MODEL_PHASE.get(sumo_phase, 0)

            # Manual duration tracking — immune to setPhase timer resets
            elapsed = float(
                self._step_count - self._phase_held_since.get(tls_id, self._step_count)
            )

            total_queue = sum(counts.values())

            observations.append({
                "intersection_id": idx,
                **counts,
                "queue_length": min(total_queue * 1.5, 200.0),
                "current_phase": model_phase,
                "phase_duration": elapsed,
            })

        return observations

    def _classify_edge_direction(self, edge_id: str, jx: float, jy: float) -> str:
        """Classify edge as north/south/east/west based on lane shape geometry."""
        try:
            shape = traci.lane.getShape(f"{edge_id}_0")
            if not shape:
                return ""
            x0, y0 = shape[0]
            dx, dy = x0 - jx, y0 - jy
            if abs(dx) > abs(dy):
                return "west" if dx < 0 else "east"
            else:
                return "south" if dy < 0 else "north"
        except Exception:
            return ""

    # ── Aksiyon Uygulama ─────────────────────────

    def apply_actions(self, actions: torch.Tensor):
        """
        Records desired target green phase per TLS and starts yellow transition.

        Model action (0-5) → target green SUMO phase via MODEL_TO_SUMO_GREEN dict.
        Yellow transitions are never interrupted once started.
        """
        for i, tls_id in enumerate(self.tls_ids):
            if self._yellow_steps_remaining.get(tls_id, 0) > 0:
                continue

            model_action = int(actions[i].item()) % 6
            target_sumo_phase = MODEL_TO_SUMO_GREEN[model_action]
            current_sumo_phase = traci.trafficlight.getPhase(tls_id)

            if target_sumo_phase == current_sumo_phase:
                traci.trafficlight.setPhase(tls_id, current_sumo_phase)
                continue

            self._pending_target[tls_id] = target_sumo_phase
            self._yellow_steps_remaining[tls_id] = self.YELLOW_STEPS

            # Yellow is always green+1 (even → odd)
            yellow_phase = (
                current_sumo_phase + 1
                if current_sumo_phase % 2 == 0
                else current_sumo_phase
            )
            traci.trafficlight.setPhase(tls_id, yellow_phase)

    def _advance_transitions(self):
        """Decrement yellow timers; switch to target green when timer reaches zero."""
        for tls_id in self.tls_ids:
            remaining = self._yellow_steps_remaining.get(tls_id, 0)
            if remaining <= 0:
                continue
            remaining -= 1
            self._yellow_steps_remaining[tls_id] = remaining
            if remaining == 0:
                target = self._pending_target.pop(tls_id, None)
                if target is not None:
                    traci.trafficlight.setPhase(tls_id, target)
                    self._phase_held_since[tls_id] = self._step_count

    # ── Simülasyon Adımı ──────────────────────────

    def step(self, actions: Optional[torch.Tensor] = None) -> Tuple[List[Dict], bool]:
        """
        Aksiyonları uygula → N adım simüle et → yeni gözlem döndür.

        Returns:
            observations: güncel kavşak gözlemleri
            done: simülasyon bitti mi
        """
        if actions is not None:
            self.apply_actions(actions)

        # decision_interval kadar simülasyon adımı at
        for _ in range(self.cfg.decision_interval):
            if not self.is_running:
                return self.get_observations(), True
            self._advance_transitions()
            traci.simulationStep()
            self._step_count += 1

        done = (
            not self.is_running
            or self._step_count >= self.cfg.max_steps_per_episode
        )
        return self.get_observations(), done

    # ── Metrikleri Topla ──────────────────────────

    def get_metrics(self) -> Dict[str, float]:
        """Simülasyondan performans metrikleri toplar."""
        try:
            vehicles = traci.vehicle.getIDList()
            if not vehicles:
                return {
                    "avg_speed": 0, "avg_waiting": 0,
                    "total_vehicles": 0, "total_halting": 0,
                }

            speeds = [traci.vehicle.getSpeed(v) for v in vehicles]
            waiting = [traci.vehicle.getWaitingTime(v) for v in vehicles]
            halting = sum(1 for s in speeds if s < 0.1)

            return {
                "avg_speed": sum(speeds) / len(speeds),
                "avg_waiting": sum(waiting) / len(waiting),
                "total_vehicles": len(vehicles),
                "total_halting": halting,
            }
        except Exception:
            return {
                "avg_speed": 0, "avg_waiting": 0,
                "total_vehicles": 0, "total_halting": 0,
            }
