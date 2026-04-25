"""
TraFix v2 — SUMO Entegrasyonlu PPO Eğitim Scripti
====================================================
TraCI üzerinden SUMO simülasyonuna bağlanır, CoordinatedPPOAgent'ı
eğitir ve ağırlıkları kaydeder.

Kullanım:
  cd x:\\trafix\\proje
  .\\venv\\Scripts\\activate

  # Varsayılan eğitim (500 episode)
  python train_v2.py

  # Özel ayarlarla
  python train_v2.py --episodes 1000 --lr 1e-4 --gui

  # Önceki checkpoint'tan devam
  python train_v2.py --resume --checkpoint coordinated_agent_weights.pth

Gereksinimler:
  pip install torch torch-geometric
  SUMO kurulu ve SUMO_HOME ortam değişkeni tanımlı olmalı
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
        CoordinatedPPOAgent,
        parse_sumo_observations,
        compute_reward,
        compute_gae,
        train_step,
        RewardWeights,
        NUM_NODE_FEATURES,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from trafix_v2 import (
        CoordinatedPPOAgent,
        parse_sumo_observations,
        compute_reward,
        compute_gae,
        train_step,
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

class SumoEnvironment:
    """
    SUMO simülasyonunu TraCI ile yönetir.
    Her adımda kavşak gözlemlerini toplar ve faz değişikliklerini uygular.
    """

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.tls_ids: List[str] = []       # trafik ışığı ID'leri
        self.num_nodes = 0
        self._step_count = 0
        self._episode_count = 0

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
            "-c", cfg_path,    # mutlak yol kullan
            "--step-length", str(self.cfg.sumo_step_length),
            "--waiting-time-memory", "1000",
            "--no-warnings", "true",
            "--random",
            "--seed", str(self.cfg.seed + episode),
        ]

        logging.info(f"  SUMO başlatılıyor: {cfg_path}")
        traci.start(sumo_cmd)
        self._step_count = 0
        self._episode_count = episode

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
        Her kavşak için SUMO'dan gözlem toplar.
        Çıktı formatı: trafix_v2.parse_sumo_observations ile uyumlu.
        """
        observations = []

        for idx, tls_id in enumerate(self.tls_ids):
            obs = self._get_single_intersection_obs(idx, tls_id)
            observations.append(obs)

        return observations

    def _get_single_intersection_obs(self, idx: int, tls_id: str) -> Dict:
        """Tek bir kavşağın gözlemini toplar."""
        # Kontrollü şeritler (kavşağa giren)
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        unique_lanes = list(dict.fromkeys(controlled_lanes))  # sırayı koru, tekrarı kaldır

        # Yön tahmini: şerit ID'sine veya indeksine göre gruplama
        # SUMO'da şerit isimleri genellikle "edgeId_laneIndex" formatında
        direction_counts = {"north": 0, "south": 0, "east": 0, "west": 0}
        total_queue = 0.0

        for i, lane in enumerate(unique_lanes):
            vehicle_count = traci.lane.getLastStepVehicleNumber(lane)
            queue = traci.lane.getLastStepHaltingNumber(lane) * \
                    max(traci.lane.getLastStepLength(lane), 7.5)  # ortalama araç boyu

            # Yön ataması — şerit indeksine göre 4'e böl
            direction_idx = i % 4
            dir_name = ["north", "south", "east", "west"][direction_idx]

            # Daha akıllı yön tahmini: şerit açısından
            try:
                edge_id = traci.lane.getEdgeID(lane)
                shape = traci.lane.getShape(lane)
                if len(shape) >= 2:
                    dx = shape[-1][0] - shape[0][0]
                    dy = shape[-1][1] - shape[0][1]
                    angle = math.degrees(math.atan2(dy, dx)) % 360

                    if 45 <= angle < 135:
                        dir_name = "north"
                    elif 135 <= angle < 225:
                        dir_name = "west"
                    elif 225 <= angle < 315:
                        dir_name = "south"
                    else:
                        dir_name = "east"
            except Exception:
                pass  # fallback: indeks tabanlı

            direction_counts[dir_name] += vehicle_count
            total_queue += queue

        # Mevcut faz — convert SUMO 8-phase (0–7) to model space (0–3)
        # SUMO phases: 0=N-green, 1=N-yellow, 2=E-green, 3=E-yellow,
        #              4=S-green, 5=S-yellow, 6=W-green, 7=W-yellow
        # Model actions: 0=N, 1=E, 2=S, 3=W  (maps to SUMO phase = action * 2)
        # Yellow phases (odd) map to the preceding green: sumo_phase // 2
        current_sumo_phase = traci.trafficlight.getPhase(tls_id)
        model_phase = current_sumo_phase // 2   # 0→0, 1→0, 2→1, 3→1, 4→2, 5→2, 6→3, 7→3

        # Faz süresi — bu faz ne zamandır aktif
        try:
            phase_duration = traci.trafficlight.getPhaseDuration(tls_id)
            next_switch = traci.trafficlight.getNextSwitch(tls_id)
            sim_time = traci.simulation.getTime()
            elapsed = phase_duration - max(0, next_switch - sim_time)
            phase_duration_val = max(0.0, elapsed)
        except Exception:
            phase_duration_val = 0.0

        return {
            "intersection_id": idx,
            "north_count": direction_counts["north"],
            "south_count": direction_counts["south"],
            "east_count": direction_counts["east"],
            "west_count": direction_counts["west"],
            "queue_length": total_queue,
            "current_phase": model_phase,   # 0–3 model-action space
            "phase_duration": phase_duration_val,
        }

    # ── Aksiyon Uygulama ─────────────────────────

    def apply_actions(self, actions: torch.Tensor):
        """
        Maps model actions to SUMO 8-phase scheme and applies them.

        Model action → SUMO phase mapping:
          0 (N-only) → SUMO phase 0   (North green)
          1 (E-only) → SUMO phase 2   (East green)
          2 (S-only) → SUMO phase 4   (South green)
          3 (W-only) → SUMO phase 6   (West green)
          formula: sumo_phase = model_action * 2

        Safety: if currently in a green phase (even index), transition through
        the corresponding yellow (current_green + 1) before switching approach.
        """
        for i, tls_id in enumerate(self.tls_ids):
            model_action      = int(actions[i].item()) % 4
            target_sumo_phase = model_action * 2          # 0, 2, 4, or 6
            current_sumo_phase = traci.trafficlight.getPhase(tls_id)

            if target_sumo_phase == current_sumo_phase:
                continue

            if current_sumo_phase % 2 == 0:
                # In a green phase — go through its yellow before switching
                traci.trafficlight.setPhase(tls_id, current_sumo_phase + 1)
            else:
                # In a yellow phase — safe to set target green directly
                traci.trafficlight.setPhase(tls_id, target_sumo_phase)

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


# ══════════════════════════════════════════════════
#  Graf Topolojisi
# ══════════════════════════════════════════════════

def build_edge_index(num_nodes: int, net_file: str = None) -> torch.Tensor:
    """
    Kavşak bağlantı grafını oluşturur.

    Önce net dosyasından otomatik çıkarmayı dener,
    başarısız olursa varsayılan 5-kavşak topolojisini kullanır.
    """
    # ── Otomatik: sumolib ile net dosyasından çıkar ──
    if net_file and os.path.exists(net_file):
        try:
            net = sumolib.net.readNet(net_file)
            tls_nodes = sorted(
                [n for n in net.getNodes() if n.getType() == "traffic_light"],
                key=lambda n: n.getID(),
            )

            if len(tls_nodes) >= 2:
                node_ids = [n.getID() for n in tls_nodes]
                id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

                edges_src, edges_dst = [], []
                for edge in net.getEdges():
                    src = edge.getFromNode().getID()
                    dst = edge.getToNode().getID()
                    if src in id_to_idx and dst in id_to_idx:
                        s, d = id_to_idx[src], id_to_idx[dst]
                        if s != d:
                            edges_src.append(s)
                            edges_dst.append(d)

                if edges_src:
                    # Çift yönlü yap
                    all_src = edges_src + edges_dst
                    all_dst = edges_dst + edges_src

                    # Tekrarları kaldır
                    seen = set()
                    final_src, final_dst = [], []
                    for s, d in zip(all_src, all_dst):
                        if (s, d) not in seen:
                            seen.add((s, d))
                            final_src.append(s)
                            final_dst.append(d)

                    logging.info(
                        f"Graf topolojisi net dosyasından çıkarıldı: "
                        f"{len(tls_nodes)} kavşak, {len(final_src)} kenar"
                    )
                    return torch.tensor([final_src, final_dst], dtype=torch.long)

        except Exception as e:
            logging.warning(f"Net dosyasından graf çıkarılamadı: {e}")

    # ── Varsayılan: 5 kavşaklı asimetrik ağ ──
    #   0 — 1 — 2
    #       |   |
    #       3 — 4
    logging.info("Varsayılan 5-kavşak topolojisi kullanılıyor")
    return torch.tensor([
        [0, 1, 1, 2, 1, 3, 2, 4, 3, 4],
        [1, 0, 2, 1, 3, 1, 4, 2, 4, 3],
    ], dtype=torch.long)


# ══════════════════════════════════════════════════
#  Rollout Buffer
# ══════════════════════════════════════════════════

class RolloutBuffer:
    """Bir rollout boyunca deneyimleri biriktirir."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.observations: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []   # each (N,) per-node
        self.values: List[torch.Tensor] = []    # each scalar

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
    ):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value.detach())

    def __len__(self):
        return len(self.rewards)

    def to_dict(self, edge_index: torch.Tensor, next_value: torch.Tensor) -> Dict:
        return {
            "observations": self.observations,
            "edge_index": edge_index,
            "actions": self.actions,
            "log_probs": self.log_probs,
            "rewards": self.rewards,
            "values": self.values,
            "next_value": next_value,
        }


# ══════════════════════════════════════════════════
#  Eğitim Loglayıcı
# ══════════════════════════════════════════════════

class TrainingLogger:
    """Eğitim metriklerini loglar ve CSV'ye kaydeder."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # CSV log dosyası
        self.csv_path = self.output_dir / "training_log.csv"
        self.json_path = self.output_dir / "training_history.json"
        self.history: List[Dict] = []

        # CSV başlık
        with open(self.csv_path, "w") as f:
            f.write(
                "episode,reward_mean,reward_std,policy_loss,value_loss,"
                "entropy,avg_speed,avg_waiting,total_vehicles,lr,"
                "entropy_coef,timestamp\n"
            )

        # Python logger
        log_path = self.output_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path, encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ],
        )

    def log_episode(self, episode: int, data: Dict):
        """Bir episode'un metriklerini logla."""
        self.history.append({"episode": episode, **data})

        with open(self.csv_path, "a") as f:
            f.write(
                f"{episode},{data.get('reward_mean', 0):.6f},"
                f"{data.get('reward_std', 0):.6f},"
                f"{data.get('policy_loss', 0):.6f},"
                f"{data.get('value_loss', 0):.6f},"
                f"{data.get('entropy', 0):.6f},"
                f"{data.get('avg_speed', 0):.4f},"
                f"{data.get('avg_waiting', 0):.4f},"
                f"{data.get('total_vehicles', 0)},"
                f"{data.get('lr', 0):.8f},"
                f"{data.get('entropy_coef', 0):.6f},"
                f"{datetime.now().isoformat()}\n"
            )

    def save_history(self):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════
#  Learning Rate Scheduler
# ══════════════════════════════════════════════════

class CosineWarmupScheduler:
    """Warmup + cosine decay LR scheduler."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_episodes: int,
        total_episodes: int,
        lr_min: float,
    ):
        self.optimizer = optimizer
        self.warmup_episodes = warmup_episodes
        self.total_episodes = total_episodes
        self.lr_min = lr_min
        self.base_lr = optimizer.param_groups[0]["lr"]

    def step(self, episode: int):
        if episode < self.warmup_episodes:
            # Lineer warmup
            lr = self.base_lr * (episode + 1) / self.warmup_episodes
        else:
            # Cosine decay
            progress = (episode - self.warmup_episodes) / max(
                1, self.total_episodes - self.warmup_episodes
            )
            lr = self.lr_min + 0.5 * (self.base_lr - self.lr_min) * (
                1 + math.cos(math.pi * progress)
            )

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ══════════════════════════════════════════════════
#  Ana Eğitim Döngüsü
# ══════════════════════════════════════════════════

def train(cfg: TrainConfig):
    """Ana eğitim fonksiyonu."""

    # ── Çıktı dizini ──
    logger = TrainingLogger(cfg.output_dir)
    logging.info("=" * 60)
    logging.info("  TraFix v2 — Eğitim Başlıyor")
    logging.info("=" * 60)
    logging.info(f"  Konfigürasyon: {json.dumps(asdict(cfg), indent=2, default=str)}")

    # ── Reproducibility ──
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"  Cihaz: {device}")

    # ── SUMO ortamı ──
    env = SumoEnvironment(cfg)

    # Kavşak sayısını öğrenmek için bir kez başlat-kapa
    env.start(episode=0)
    num_nodes = env.num_nodes
    env.close()
    logging.info(f"  Kavşak sayısı: {num_nodes}")

    # ── Graf topolojisi ──
    edge_index = build_edge_index(num_nodes, cfg.net_file).to(device)

    # ── Ajan ──
    agent = CoordinatedPPOAgent(
        num_node_features=NUM_NODE_FEATURES,
        hidden_dim=cfg.hidden_dim,
        num_actions=cfg.num_actions,
        num_heads=cfg.num_heads,
        entropy_coef=cfg.entropy_coef,
        value_coef=cfg.value_coef,
        clip_eps=cfg.clip_eps,
        max_grad_norm=cfg.max_grad_norm,
    ).to(device)

    # ── Checkpoint'tan devam ──
    start_episode = 0
    if cfg.resume and os.path.exists(cfg.checkpoint_path):
        try:
            checkpoint = torch.load(cfg.checkpoint_path, map_location=device, weights_only=True)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                agent.load_state_dict(checkpoint["model_state_dict"])
                start_episode = checkpoint.get("episode", 0)
                logging.info(f"  Checkpoint yüklendi: episode {start_episode}")
            else:
                agent.load_state_dict(checkpoint)
                logging.info("  Ağırlıklar yüklendi (eski format)")
        except RuntimeError as e:
            logging.warning(
                f"  Checkpoint uyumsuz (v1 ağırlıkları?), sıfırdan başlanıyor: {e}"
            )

    # ── Optimizer & Scheduler ──
    optimizer = optim.Adam(agent.parameters(), lr=cfg.lr, eps=cfg.eps)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_episodes=min(20, cfg.episodes // 10),
        total_episodes=cfg.episodes,
        lr_min=cfg.lr_min,
    )

    # ── Eğitim takibi ──
    best_reward = float("-inf")
    reward_history = deque(maxlen=50)    # son 50 episode'un ortalaması
    no_improve_count = 0
    current_entropy_coef = cfg.entropy_coef

    logging.info(f"  Başlangıç episode: {start_episode}")
    logging.info(f"  Toplam episode: {cfg.episodes}")
    logging.info(f"  Parametre sayısı: {sum(p.numel() for p in agent.parameters()):,}")
    logging.info("=" * 60)

    # ════════════════════════════════════════
    #  Episode Döngüsü
    # ════════════════════════════════════════

    for episode in range(start_episode, cfg.episodes):
        episode_start = time.time()

        # ── Entropi katsayısı azaltma ──
        current_entropy_coef = max(
            cfg.entropy_coef_min,
            cfg.entropy_coef * (cfg.entropy_decay ** episode),
        )
        agent.entropy_coef = current_entropy_coef

        # ── LR güncelle ──
        current_lr = scheduler.step(episode)

        # ── Dinamik trafik talebi üret (her episode farklı yoğunluk) ──
        if _HAS_DYNAMIC_DEMAND:
            try:
                _gen_demand()
            except Exception as _e:
                logging.warning(f"  Dinamik talep üretilemedi, önceki dosya kullanılıyor: {_e}")

        # ── SUMO başlat ──
        try:
            env.start(episode=episode)
        except FileNotFoundError as e:
            # Dosya bulunamadı → düzeltilemez, eğitimi durdur
            logging.error(str(e))
            sys.exit(1)
        except Exception as e:
            logging.error(f"Episode {episode}: SUMO başlatılamadı: {e}")
            consecutive_failures = getattr(train, '_failures', 0) + 1
            train._failures = consecutive_failures
            if consecutive_failures >= 3:
                logging.error("3 ardışık başarısızlık — eğitim durduruluyor.")
                sys.exit(1)
            continue
        else:
            train._failures = 0  # başarılı başlatma, sayacı sıfırla

        # ── Başlangıç durumu ──
        agent.train()
        buffer = RolloutBuffer()

        obs = env.get_observations()
        prev_obs = None
        prev_actions = None

        episode_rewards = []
        episode_metrics = []
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        update_count = 0
        step = 0

        # ════════════════════════════════
        #  Adım Döngüsü
        # ════════════════════════════════

        done = False
        while not done:
            # ── Gözlemi tensöre çevir (fixed-scale normalisation) ──
            x = parse_sumo_observations(obs, device)

            # ── Aksiyon seç ──
            actions, log_probs, value = agent.select_actions(x, edge_index)

            # ── Ortama uygula ──
            next_obs, done = env.step(actions)

            # ── Ödül hesapla — (N,) per-node tensor ──
            reward = compute_reward(
                current_obs=next_obs,
                previous_obs=prev_obs,
                previous_actions=prev_actions,
                current_actions=actions,
                weights=cfg.reward_weights,
            ).to(device)

            episode_rewards.append(reward.mean().item())

            # ── Buffer'a ekle ──
            buffer.add(x, actions, log_probs, reward, value)

            # ── PPO güncelleme (buffer dolduğunda) ──
            if len(buffer) >= cfg.rollout_length:
                with torch.no_grad():
                    next_x = parse_sumo_observations(next_obs, device)
                    _, next_value = agent(next_x, edge_index)

                rollout = buffer.to_dict(edge_index, next_value.detach())

                agent.train()
                losses = train_step(agent, optimizer, rollout, cfg.ppo_epochs)

                total_policy_loss += losses["policy"]
                total_value_loss  += losses["value"]
                total_entropy     += losses["entropy"]
                update_count += 1

                buffer.clear()

            # ── Metrikleri topla ──
            if step % 10 == 0:
                metrics = env.get_metrics()
                episode_metrics.append(metrics)

            # ── Sonraki adıma hazırlan ──
            prev_obs     = obs
            prev_actions = actions
            obs  = next_obs
            step += 1

        # ── Kalan buffer'ı da eğit ──
        if len(buffer) > 1:
            with torch.no_grad():
                x = parse_sumo_observations(obs, device)
                _, next_value = agent(x, edge_index)

            rollout = buffer.to_dict(edge_index, next_value.detach())
            agent.train()
            losses = train_step(agent, optimizer, rollout, cfg.ppo_epochs)
            total_policy_loss += losses["policy"]
            total_value_loss  += losses["value"]
            total_entropy     += losses["entropy"]
            update_count += 1

        # ── SUMO kapat ──
        env.close()

        # ════════════════════════════════
        #  Episode İstatistikleri
        # ════════════════════════════════

        episode_time = time.time() - episode_start
        mean_reward = sum(episode_rewards) / max(len(episode_rewards), 1)
        std_reward = (
            (sum((r - mean_reward) ** 2 for r in episode_rewards)
             / max(len(episode_rewards), 1)) ** 0.5
        )

        # SUMO metrikleri ortalaması
        avg_metrics = {}
        if episode_metrics:
            for key in episode_metrics[0]:
                vals = [m[key] for m in episode_metrics]
                avg_metrics[key] = sum(vals) / len(vals)
        else:
            avg_metrics = {"avg_speed": 0, "avg_waiting": 0,
                          "total_vehicles": 0, "total_halting": 0}

        reward_history.append(mean_reward)
        rolling_avg = sum(reward_history) / len(reward_history)

        # Loglama
        log_data = {
            "reward_mean": mean_reward,
            "reward_std": std_reward,
            "policy_loss": total_policy_loss / max(update_count, 1),
            "value_loss": total_value_loss / max(update_count, 1),
            "entropy": total_entropy / max(update_count, 1),
            "lr": current_lr,
            "entropy_coef": current_entropy_coef,
            **avg_metrics,
        }
        logger.log_episode(episode, log_data)

        # Konsol çıktısı
        if episode % cfg.log_interval == 0:
            logging.info(
                f"EP {episode:>4d}/{cfg.episodes} | "
                f"R={mean_reward:+.4f} (avg50={rolling_avg:+.4f}) | "
                f"π={log_data['policy_loss']:.4f} V={log_data['value_loss']:.4f} "
                f"H={log_data['entropy']:.4f} | "
                f"spd={avg_metrics.get('avg_speed', 0):.1f} "
                f"wait={avg_metrics.get('avg_waiting', 0):.1f} | "
                f"LR={current_lr:.2e} ε_H={current_entropy_coef:.4f} | "
                f"{episode_time:.1f}s"
            )

        # ── En iyi model kaydet ──
        if mean_reward > best_reward:
            best_reward = mean_reward
            no_improve_count = 0
            _save_checkpoint(
                agent, optimizer, episode, best_reward,
                os.path.join(cfg.output_dir, "best_model.pth"),
            )
        else:
            no_improve_count += 1

        # ── Periyodik checkpoint ──
        if (episode + 1) % cfg.save_interval == 0:
            path = os.path.join(cfg.output_dir, f"checkpoint_ep{episode + 1}.pth")
            _save_checkpoint(agent, optimizer, episode, best_reward, path)

            # Ana ağırlık dosyası (her zaman güncel)
            _save_checkpoint(
                agent, optimizer, episode, best_reward, cfg.checkpoint_path,
            )
            logging.info(f"  → Checkpoint kaydedildi: episode {episode + 1}")

    # ════════════════════════════════════════
    #  Eğitim Sonu
    # ════════════════════════════════════════

    # Son kayıt
    _save_checkpoint(agent, optimizer, cfg.episodes - 1, best_reward, cfg.checkpoint_path)
    _save_checkpoint(
        agent, optimizer, cfg.episodes - 1, best_reward,
        os.path.join(cfg.output_dir, "final_model.pth"),
    )
    logger.save_history()

    logging.info("=" * 60)
    logging.info("  EĞİTİM TAMAMLANDI")
    logging.info(f"  En iyi ödül     : {best_reward:.6f}")
    logging.info(f"  Son 50 ortalama : {rolling_avg:.6f}")
    logging.info(f"  Checkpoint      : {cfg.checkpoint_path}")
    logging.info(f"  Log dizini      : {cfg.output_dir}")
    logging.info("=" * 60)


# ══════════════════════════════════════════════════
#  Yardımcılar
# ══════════════════════════════════════════════════

def _save_checkpoint(
    agent: CoordinatedPPOAgent,
    optimizer: optim.Optimizer,
    episode: int,
    best_reward: float,
    path: str,
):
    """Model + optimizer + meta bilgi kaydet."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode": episode,
            "best_reward": best_reward,
            "config": {
                "hidden_dim": agent.hidden_dim,
                "num_actions": agent.num_actions,
                "entropy_coef": agent.entropy_coef,
            },
        },
        path,
    )


# ══════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="TraFix v2 — SUMO PPO Eğitim",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dosya yolları
    parser.add_argument("--sumo-cfg", default="sumo/training.sumocfg",
                        help="SUMO konfigürasyon dosyası")
    parser.add_argument("--net-file", default="sumo/map.net.xml",
                        help="SUMO ağ dosyası")
    parser.add_argument("--output-dir", default="training_outputs",
                        help="Çıktı dizini")
    parser.add_argument("--checkpoint", default="coordinated_agent_weights.pth",
                        help="Checkpoint dosya adı")

    # Eğitim
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=3600,
                        help="Episode başına maks simülasyon adımı")
    parser.add_argument("--decision-interval", type=int, default=10,
                        help="Karar aralığı (simülasyon adımı)")
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=4)

    # Model
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)

    # Optimizasyon
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--value-coef", type=float, default=0.25)

    # SUMO
    parser.add_argument("--gui", action="store_true",
                        help="SUMO GUI ile çalıştır")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=5,
                        help="Her N episode'da detaylı log yaz")

    # Resume
    parser.add_argument("--resume", action="store_true",
                        help="Önceki checkpoint'tan devam et")

    args = parser.parse_args()

    return TrainConfig(
        sumo_cfg=args.sumo_cfg,
        net_file=args.net_file,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        decision_interval=args.decision_interval,
        rollout_length=args.rollout_length,
        ppo_epochs=args.ppo_epochs,
        hidden_dim=args.hidden_dim,
        num_actions=args.num_actions,
        num_heads=args.num_heads,
        lr=args.lr,
        gamma=args.gamma,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        gui=args.gui,
        seed=args.seed,
        resume=args.resume,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)