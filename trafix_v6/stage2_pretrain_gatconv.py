"""
TraFix v6 — Stage 2: GATConv + MLP Trunk Pretraining
======================================================
Loads frozen GRU from Stage 1, then trains GATConv + trunk with auxiliary
neighbor total-queue prediction.

v6 changes vs v5:
  OBS_DIM = 20      (was 10)
  QUEUE_FEAT_IDX = 12  (index 12 = total queue / 200)

Auxiliary task: predict neighbor total queue at T+1.
Loss:    variance-normalised MSE
Saves:   checkpoints/stage2_gatconv.pt
         checkpoints/stage2_trunk.pt

Usage:
  python trafix_v6/stage2_pretrain_gatconv.py
  python trafix_v6/stage2_pretrain_gatconv.py --episodes 400 --offpeak-episodes 200
"""

import os
import sys
import math
import time
import logging
import argparse
from pathlib import Path
from collections import deque
from typing import List

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
from scenario_generator import ScenarioGenerator, ScenarioEnvironment, ScenarioType

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

OBS_DIM = NUM_NODE_FEATURES   # 20
NUM_PHASES = 6
T_WINDOW = 30
QUEUE_FEAT_IDX = 12           # total queue / 200.0 (index 12 in 20-dim obs)

# Real map topology neighbor lists (from sumo/map.net.xml)
# Layout:  J0—J1, J0—J2, J1—J3, J2—J3, J2—J4
TOPOLOGY_NEIGHBORS: List[List[int]] = [[1, 2], [0, 3], [0, 3, 4], [1, 2], [2]]
MAX_NEIGHBORS = max(len(nb) for nb in TOPOLOGY_NEIGHBORS)

CHECKPOINTS_DIR = _SCRIPT_DIR / "checkpoints"
STAGE1_CHECKPOINT = CHECKPOINTS_DIR / "stage1_gru.pt"
STAGE2_GATCONV_CHECKPOINT = CHECKPOINTS_DIR / "stage2_gatconv.pt"
STAGE2_TRUNK_CHECKPOINT = CHECKPOINTS_DIR / "stage2_trunk.pt"
DEFAULT_SUMO_CFG = str(_PROJECT_ROOT / "sumo" / "training.sumocfg")
DEFAULT_NET_FILE = str(_PROJECT_ROOT / "sumo" / "map.net.xml")


def _nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 0.02) -> torch.Tensor:
    """Variance-normalised MSE — keeps loss scale stable across traffic intensities."""
    var = target.var(correction=0).clamp(min=eps)
    return F.mse_loss(pred, target) / var


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
    if not STAGE1_CHECKPOINT.exists():
        sys.exit(f"Stage 1 checkpoint not found: {STAGE1_CHECKPOINT}\nRun stage1_pretrain_gru.py first.")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    logs_dir = _SCRIPT_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(str(logs_dir / "stage2_pretrain.log"), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("=" * 60)
    logging.info("  TraFix v6 — Stage 2: GATConv + Trunk Pretraining")
    logging.info("=" * 60)
    logging.info(f"  obs_dim={OBS_DIM}, QUEUE_FEAT_IDX={QUEUE_FEAT_IDX}, "
                 f"T_window={T_WINDOW}, episodes={args.episodes}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"  Device: {device}")

    model = TraFixV6(obs_dim=OBS_DIM, num_phases=NUM_PHASES).to(device)

    # Load and freeze GRU from Stage 1
    temporal_state = torch.load(str(STAGE1_CHECKPOINT), map_location=device, weights_only=True)
    model.temporal_enc.load_state_dict(temporal_state)
    for param in model.temporal_enc.parameters():
        param.requires_grad = False
    logging.info(f"  GRU loaded from {STAGE1_CHECKPOINT} and frozen.")

    # Temporary neighbor queue prediction heads (Sigmoid-bounded [0,1])
    pred_heads = nn.ModuleList([
        nn.Sequential(
            nn.Linear(model.trunk_out, len(TOPOLOGY_NEIGHBORS[j])),
            nn.Sigmoid(),
        )
        for j in range(NUM_JUNCTIONS)
    ]).to(device)

    trainable_params = (
        list(model.graph_enc.parameters())
        + list(model.trunk.parameters())
        + list(pred_heads.parameters())
    )
    optimizer = optim.Adam(trainable_params, lr=args.lr)
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
    logging.info(
        f"  Curriculum: OFFPEAK-only for first {args.offpeak_episodes} episodes, "
        f"then standard curriculum mix."
    )

    for episode in range(args.episodes):
        if episode < args.offpeak_episodes:
            route_file = generator.generate(ScenarioType.OFFPEAK, episode)
            scenario_type = ScenarioType.OFFPEAK
        else:
            scenario_type, route_file = generator.sample(episode)

        env.set_route_file(route_file)
        episode_start = time.time()
        episode_losses = []

        try:
            env.start(episode=episode)
            num_nodes = env.num_nodes

            obs_list = env.get_observations()
            x = parse_sumo_observations(obs_list, device=device)
            window: deque = deque([x.detach()] * T_WINDOW, maxlen=T_WINDOW)

            done = False
            while not done:
                random_actions = torch.randint(0, NUM_PHASES, (num_nodes,))
                next_obs_list, done = env.step(random_actions)
                x_next = parse_sumo_observations(next_obs_list, device=device)

                window_tensor = torch.stack(list(window)).unsqueeze(0)

                with torch.no_grad():
                    gru_out = model.temporal_enc(window_tensor)  # [1, J, hidden_dim]

                gru_flat = gru_out.squeeze(0)  # [J, hidden_dim]
                g = model.graph_enc(gru_flat, model.edge_index)  # [J, gat_out]
                t = model.trunk(g)  # [J, trunk_out]

                loss = torch.tensor(0.0, device=device)
                for j in range(NUM_JUNCTIONS):
                    pred = pred_heads[j](t[j])  # [len(neighbors_j)]
                    target = torch.tensor(
                        [x_next[nb, QUEUE_FEAT_IDX].item() for nb in TOPOLOGY_NEIGHBORS[j]],
                        dtype=torch.float32, device=device,
                    )
                    loss = loss + _nmse(pred, target)
                loss = loss / NUM_JUNCTIONS

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
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
            torch.save(model.graph_enc.state_dict(), str(STAGE2_GATCONV_CHECKPOINT))
            torch.save(model.trunk.state_dict(), str(STAGE2_TRUNK_CHECKPOINT))

        if (episode + 1) % 10 == 0:
            phase_tag = "OFFPEAK-only" if episode < args.offpeak_episodes else "curriculum"
            logging.info(
                f"  Ep {episode + 1:>3d}/{args.episodes} [{phase_tag}] | "
                f"Loss: {mean_loss:.6f} | Best: {best_loss:.6f} | "
                f"LR: {current_lr:.2e} | Steps: {len(episode_losses):>4d} | "
                f"{elapsed:.1f}s | {generator.last_summary}"
            )

    logging.info(f"  Best GATConv saved → {STAGE2_GATCONV_CHECKPOINT}")
    logging.info(f"  Best Trunk   saved → {STAGE2_TRUNK_CHECKPOINT}")
    print("Stage 2 complete. GATConv and trunk saved.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="TraFix v6 Stage 2 — GATConv + trunk pretraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sumo-cfg", default=DEFAULT_SUMO_CFG)
    parser.add_argument("--net-file", default=DEFAULT_NET_FILE)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--offpeak-episodes", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--decision-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
