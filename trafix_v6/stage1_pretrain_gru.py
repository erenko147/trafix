"""
TraFix v6 — Stage 1: GRU Temporal Encoder Pretraining
=======================================================
Supervised next-step prediction to pretrain the GRU temporal encoder.

v6 changes vs v5:
  OBS_DIM = 20          (was 10)
  TRAFFIC_FEAT_DIM = 12 (was 5) — predict all 12 per-lane counts
  Prediction target: x_next[:, :12] (first 12 features)

Inputs : sliding window of T=10 consecutive junction observations [J, 20]
Target : first 12 features at timestep T+1 (normalised per-lane counts)
Loss   : MSE
Saves  : checkpoints/stage1_gru.pt (temporal_enc state dict only)

Usage:
  python trafix_v6/stage1_pretrain_gru.py
  python trafix_v6/stage1_pretrain_gru.py --episodes 300 --lr 1e-3
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
    raise ImportError("SUMO TraCI not found. Set SUMO_HOME.")

# ── Path setup ──
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

from trafix_v6.trafix_v6 import TraFixV6, NUM_JUNCTIONS
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
#  Constants
# ══════════════════════════════════════════════════

OBS_DIM = NUM_NODE_FEATURES          # 20
NUM_PHASES = 6
T_WINDOW = 10

# Predict all 12 per-lane counts (indices 0-11 in the 20-dim obs)
TRAFFIC_FEAT_DIM = 12

CHECKPOINTS_DIR = _SCRIPT_DIR / "checkpoints"
STAGE1_CHECKPOINT = CHECKPOINTS_DIR / "stage1_gru.pt"
DEFAULT_SUMO_CFG = str(_PROJECT_ROOT / "sumo" / "training.sumocfg")
DEFAULT_NET_FILE = str(_PROJECT_ROOT / "sumo" / "map.net.xml")


def make_env_config(sumo_cfg, gui, seed, decision_interval):
    cfg = TrainConfig()
    cfg.sumo_cfg = sumo_cfg
    cfg.gui = gui
    cfg.seed = seed
    cfg.decision_interval = decision_interval
    cfg.warmup_steps = 30
    cfg.max_steps_per_episode = 1800
    cfg.num_actions = NUM_PHASES
    return cfg


def train(args):
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
    logging.info("  TraFix v6 — Stage 1: GRU Pretraining")
    logging.info("=" * 60)
    logging.info(f"  obs_dim={OBS_DIM}, traffic_feat_dim={TRAFFIC_FEAT_DIM}, "
                 f"T_window={T_WINDOW}, episodes={args.episodes}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"  Device: {device}")

    # ── Model ──
    model = TraFixV6(obs_dim=OBS_DIM, num_phases=NUM_PHASES).to(device)

    # Prediction head: GRU hidden → next-step 12 per-lane counts
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

            obs_list = env.get_observations()
            x = parse_sumo_observations(obs_list, device=device)  # [J, 20]
            window: deque = deque([x.detach()] * T_WINDOW, maxlen=T_WINDOW)

            done = False
            while not done:
                random_actions = torch.randint(0, NUM_PHASES, (num_nodes,))
                next_obs_list, done = env.step(random_actions)
                x_next = parse_sumo_observations(next_obs_list, device=device)

                window_tensor = torch.stack(list(window)).unsqueeze(0)  # [1, T, J, 20]
                # Target: first 12 features (per-lane counts, already normalised)
                target = x_next[:, :TRAFFIC_FEAT_DIM].unsqueeze(0)     # [1, J, 12]

                h = model.temporal_enc(window_tensor)  # [1, J, hidden_dim]
                pred = pred_head(h)                    # [1, J, 12]

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
                f"LR: {current_lr:.2e} | Steps: {len(episode_losses):>4d} | "
                f"{elapsed:.1f}s | {generator.last_summary}"
            )

    logging.info(f"  Best checkpoint saved → {STAGE1_CHECKPOINT}")
    print("Stage 1 complete. GRU saved.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="TraFix v6 Stage 1 — GRU temporal encoder pretraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sumo-cfg", default=DEFAULT_SUMO_CFG)
    parser.add_argument("--net-file", default=DEFAULT_NET_FILE)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decision-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
