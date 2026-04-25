"""
TraFix v5 — Stage 2: GATConv + MLP Trunk Pretraining
======================================================
Loads frozen GRU from Stage 1, then trains GATConv + trunk with an
auxiliary neighbor queue-length prediction task.

Auxiliary task : from trunk embedding of junction i, predict the
                 queue_length of each of i's neighbors at T+1.
                 Chain topology 0-1-2-3-4 → neighbor counts: [1,2,2,2,1]
Loss           : masked MSE (only over valid neighbors)
Saves          : checkpoints/stage2_gatconv.pt
                 checkpoints/stage2_trunk.pt

Kullanım:
  python stage2_pretrain_gatconv.py
  python stage2_pretrain_gatconv.py --episodes 300 --lr 5e-4
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
T_WINDOW = 10
QUEUE_FEAT_IDX = 4                   # index of queue_length in FEATURE_ORDER

# Chain 0-1-2-3-4 bidirectional neighbor lists
CHAIN_NEIGHBORS: List[List[int]] = [[1], [0, 2], [1, 3], [2, 4], [3]]
MAX_NEIGHBORS = max(len(nb) for nb in CHAIN_NEIGHBORS)  # 2

CHECKPOINTS_DIR = _SCRIPT_DIR / "checkpoints"
STAGE1_CHECKPOINT = CHECKPOINTS_DIR / "stage1_gru.pt"
STAGE2_GATCONV_CHECKPOINT = CHECKPOINTS_DIR / "stage2_gatconv.pt"
STAGE2_TRUNK_CHECKPOINT = CHECKPOINTS_DIR / "stage2_trunk.pt"
DEFAULT_SUMO_CFG = str(_PROJECT_ROOT / "sumo" / "training.sumocfg")
DEFAULT_NET_FILE = str(_PROJECT_ROOT / "sumo" / "map.net.xml")


# ══════════════════════════════════════════════════
#  Komşu kuyruk tahmini yardımcıları
# ══════════════════════════════════════════════════

def build_neighbor_targets(obs_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    obs_tensor: [J, obs_dim]
    Returns:
        targets: [J, MAX_NEIGHBORS]  — neighbor queue lengths (0 for padding)
        mask:    [J, MAX_NEIGHBORS]  — 1.0 for valid neighbors, 0.0 for padding
    """
    J = obs_tensor.shape[0]
    targets = torch.zeros(J, MAX_NEIGHBORS, device=obs_tensor.device)
    mask = torch.zeros(J, MAX_NEIGHBORS, device=obs_tensor.device)

    for i, neighbors in enumerate(CHAIN_NEIGHBORS):
        for k, nb in enumerate(neighbors):
            targets[i, k] = obs_tensor[nb, QUEUE_FEAT_IDX]
            mask[i, k] = 1.0

    return targets, mask


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
#  Stage 2 Eğitim Döngüsü
# ══════════════════════════════════════════════════

def train(args: argparse.Namespace):
    # ── Prerequisite check ──
    if not STAGE1_CHECKPOINT.exists():
        sys.exit(
            "Stage 1 checkpoint not found. Run stage1_pretrain_gru.py first.\n"
            f"  Expected: {STAGE1_CHECKPOINT}"
        )

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
    logging.info("  TraFix v5 — Stage 2: GATConv + Trunk Pretraining")
    logging.info("=" * 60)
    logging.info(f"  obs_dim={OBS_DIM}, T_window={T_WINDOW}, episodes={args.episodes}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"  Device: {device}")

    # ── Model ──
    model = TraFixV5(obs_dim=OBS_DIM, num_phases=NUM_PHASES).to(device)

    # Load and freeze GRU (temporal encoder)
    temporal_state = torch.load(str(STAGE1_CHECKPOINT), map_location=device, weights_only=True)
    model.temporal_enc.load_state_dict(temporal_state)
    for param in model.temporal_enc.parameters():
        param.requires_grad = False
    logging.info(f"  GRU loaded from {STAGE1_CHECKPOINT} and frozen.")

    # Temporary neighbor queue prediction heads (one per junction, not added to TraFixV5)
    pred_heads = nn.ModuleList([
        nn.Linear(model.trunk_out, len(CHAIN_NEIGHBORS[j]))
        for j in range(NUM_JUNCTIONS)
    ]).to(device)

    trainable_params = (
        list(model.graph_enc.parameters())
        + list(model.trunk.parameters())
        + list(pred_heads.parameters())
    )
    optimizer = optim.Adam(trainable_params, lr=args.lr)

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
            x = parse_sumo_observations(obs_list, device=device)  # [J, obs_dim]
            window: deque = deque([x.detach()] * T_WINDOW, maxlen=T_WINDOW)

            done = False
            while not done:
                random_actions = torch.randint(0, NUM_PHASES, (num_nodes,))
                next_obs_list, done = env.step(random_actions)
                x_next = parse_sumo_observations(next_obs_list, device=device)  # [J, obs_dim]

                # Stack window → [1, T, J, obs_dim]
                window_tensor = torch.stack(list(window)).unsqueeze(0)

                # GRU (frozen) → [1, J, hidden_dim]
                with torch.no_grad():
                    gru_out = model.temporal_enc(window_tensor)  # [1, J, hidden_dim]

                # Reshape for GAT: [J, hidden_dim]
                gru_flat = gru_out.squeeze(0)  # [J, hidden_dim]

                # GATConv → [J, gat_out]
                g = model.graph_enc(gru_flat, model.edge_index)  # [J, gat_out]

                # Trunk → [J, trunk_out]
                t = model.trunk(g)  # [J, trunk_out]

                # Predict neighbor queue lengths for each junction
                loss = torch.tensor(0.0, device=device)
                for j in range(NUM_JUNCTIONS):
                    pred = pred_heads[j](t[j])               # [len(neighbors_j)]
                    target = torch.tensor(
                        [x_next[nb, QUEUE_FEAT_IDX].item() for nb in CHAIN_NEIGHBORS[j]],
                        dtype=torch.float32,
                        device=device,
                    )
                    loss = loss + F.mse_loss(pred, target)

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

        if mean_loss < best_loss:
            best_loss = mean_loss

        if (episode + 1) % 10 == 0:
            logging.info(
                f"  Ep {episode + 1:>3d}/{args.episodes} | "
                f"Loss: {mean_loss:.6f} | Best: {best_loss:.6f} | "
                f"Steps: {len(episode_losses):>4d} | {elapsed:.1f}s | "
                f"{generator.last_summary}"
            )

    # ── Save GATConv and trunk (not GRU, not actor/critic) ──
    torch.save(model.graph_enc.state_dict(), str(STAGE2_GATCONV_CHECKPOINT))
    torch.save(model.trunk.state_dict(), str(STAGE2_TRUNK_CHECKPOINT))
    logging.info(f"  GATConv saved → {STAGE2_GATCONV_CHECKPOINT}")
    logging.info(f"  Trunk   saved → {STAGE2_TRUNK_CHECKPOINT}")
    print("Stage 2 complete. GATConv and trunk saved.")


# ══════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TraFix v5 Stage 2 — GATConv + trunk pretraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sumo-cfg", default=DEFAULT_SUMO_CFG)
    parser.add_argument("--net-file", default=DEFAULT_NET_FILE)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--decision-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
