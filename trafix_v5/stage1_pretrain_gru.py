"""
TraFix v5 — Stage 1: GRU Temporal Encoder Pretraining
=======================================================
Supervised next-step prediction to pretrain the GRU temporal encoder.

Inputs : sliding window of T=10 consecutive junction observations
Target : observation vector at timestep T+1
Loss   : MSE between predicted and actual next-step features
Saves  : checkpoints/stage1_gru.pt  (temporal_enc state dict only)

Kullanım:
  python stage1_pretrain_gru.py
  python stage1_pretrain_gru.py --episodes 500 --lr 5e-4 --gui
"""

import os
import sys
import math
import time
import logging
import argparse
from pathlib import Path
from collections import deque
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ── SUMO TraCI ──
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
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
except ImportError:
    raise ImportError(
        "SUMO TraCI bulunamadı. SUMO_HOME ortam değişkenini ayarlayın:\n"
        "  export SUMO_HOME=/usr/share/sumo"
    )

# ── Path setup ──
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

from trafix_v5 import TraFixV5, NUM_JUNCTIONS
from scenario_generator import ScenarioGenerator, ScenarioEnvironment

try:
    from backend.ai.train_v2 import TrainConfig
    from backend.ai.trafix_v2 import parse_sumo_observations, NUM_NODE_FEATURES
except ImportError:
    try:
        from train_v2 import TrainConfig
        from trafix_v2 import parse_sumo_observations, NUM_NODE_FEATURES
    except ImportError:
        sys.path.insert(0, str(_PROJECT_ROOT / "backend" / "ai"))
        from train_v2 import TrainConfig
        from trafix_v2 import parse_sumo_observations, NUM_NODE_FEATURES


# ══════════════════════════════════════════════════
#  Sabitler
# ══════════════════════════════════════════════════

OBS_DIM = NUM_NODE_FEATURES          # 10: counts/30, queue/150, phase one-hot (4), duration/120
NUM_PHASES = 4
T_WINDOW = 10                        # GRU geçmiş penceresi

# Only predict the physically-meaningful traffic state features (indices 0-4).
# Phase one-hot (5-8) and duration (9) are driven by random actions → pure noise
# for the pretraining task and cause the GRU to default to predicting the mean.
TRAFFIC_FEAT_DIM = 5                 # [north/30, south/30, east/30, west/30, queue/150]

CHECKPOINTS_DIR = _SCRIPT_DIR / "checkpoints"
STAGE1_CHECKPOINT = CHECKPOINTS_DIR / "stage1_gru.pt"
DEFAULT_SUMO_CFG = str(_PROJECT_ROOT / "sumo" / "training.sumocfg")
DEFAULT_NET_FILE = str(_PROJECT_ROOT / "sumo" / "map.net.xml")


# ══════════════════════════════════════════════════
#  Yardımcı: minimal SumoEnvironment konfigürasyonu
# ══════════════════════════════════════════════════

def make_env_config(sumo_cfg: str, gui: bool, seed: int, decision_interval: int) -> TrainConfig:
    cfg = TrainConfig()
    cfg.sumo_cfg = sumo_cfg
    cfg.gui = gui
    cfg.seed = seed
    cfg.decision_interval = decision_interval
    cfg.warmup_steps = 30
    cfg.max_steps_per_episode = 1800
    cfg.num_actions = NUM_PHASES
    return cfg


# ══════════════════════════════════════════════════
#  Stage 1 Eğitim Döngüsü
# ══════════════════════════════════════════════════

def train(args: argparse.Namespace):
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    logs_dir = _SCRIPT_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(str(logs_dir / "stage1_pretrain.log"), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("=" * 60)
    logging.info("  TraFix v5 — Stage 1: GRU Pretraining")
    logging.info("=" * 60)
    logging.info(f"  obs_dim={OBS_DIM}, T_window={T_WINDOW}, episodes={args.episodes}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"  Device: {device}")

    # ── Model ──
    model = TraFixV5(obs_dim=OBS_DIM, num_phases=NUM_PHASES).to(device)

    # Prediction head: GRU hidden → next-step traffic state (counts + queue only)
    pred_head = nn.Linear(model.hidden_dim, TRAFFIC_FEAT_DIM).to(device)

    trainable = list(model.temporal_enc.parameters()) + list(pred_head.parameters())
    optimizer = optim.Adam(trainable, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-5
    )

    env_cfg = make_env_config(args.sumo_cfg, args.gui, args.seed, args.decision_interval)
    env = ScenarioEnvironment(env_cfg)

    generator = ScenarioGenerator(
        net_file=args.net_file,
        output_dir=str(_SCRIPT_DIR / "scenarios"),
        seed=42,
    )

    best_loss = math.inf

    for episode in range(args.episodes):
        scenario_type, route_file = generator.sample(episode)
        env.set_route_file(route_file)

        episode_start = time.time()
        episode_losses = []

        try:
            env.start(episode=episode)
            num_nodes = env.num_nodes

            # First observation after warmup
            obs_list = env.get_observations()
            x = parse_sumo_observations(obs_list, device=device)  # [J, obs_dim]

            # Initialize window by repeating first observation
            window: deque = deque([x.detach()] * T_WINDOW, maxlen=T_WINDOW)

            done = False
            while not done:
                # Random actions to generate diverse observations
                random_actions = torch.randint(0, NUM_PHASES, (num_nodes,))
                next_obs_list, done = env.step(random_actions)
                x_next = parse_sumo_observations(next_obs_list, device=device)  # [J, obs_dim]

                # Stack window → [1, T, J, obs_dim]
                window_tensor = torch.stack(list(window)).unsqueeze(0)       # [1, T, J, obs_dim]
                # Target: only traffic-state features — counts + queue, all in [0,1]
                target = x_next[:, :TRAFFIC_FEAT_DIM].unsqueeze(0)          # [1, J, 5]

                # Forward: temporal encoder only
                h = model.temporal_enc(window_tensor)  # [1, J, hidden_dim]
                pred = pred_head(h)                    # [1, J, TRAFFIC_FEAT_DIM]

                loss = F.mse_loss(pred, target)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=0.5)
                optimizer.step()

                episode_losses.append(loss.item())
                window.append(x_next.detach())

        finally:
            env.close()

        mean_loss = sum(episode_losses) / max(len(episode_losses), 1)
        elapsed = time.time() - episode_start

        scheduler.step(mean_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.temporal_enc.state_dict(), str(STAGE1_CHECKPOINT))

        if (episode + 1) % 10 == 0:
            logging.info(
                f"  Ep {episode + 1:>3d}/{args.episodes} | "
                f"Loss: {mean_loss:.6f} | Best: {best_loss:.6f} | "
                f"LR: {current_lr:.2e} | "
                f"Steps: {len(episode_losses):>4d} | {elapsed:.1f}s | "
                f"{generator.last_summary}"
            )

    logging.info(f"  Best checkpoint saved → {STAGE1_CHECKPOINT}")
    print("Stage 1 complete. GRU saved.")


# ══════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TraFix v5 Stage 1 — GRU temporal encoder pretraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sumo-cfg", default=DEFAULT_SUMO_CFG,
                        help="SUMO konfigürasyon dosyası")
    parser.add_argument("--net-file", default=DEFAULT_NET_FILE,
                        help="SUMO .net.xml dosyası (scenario generator için)")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decision-interval", type=int, default=10,
                        help="SUMO adım aralığı (saniye)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true", help="SUMO GUI ile çalıştır")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
