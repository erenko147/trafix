"""
TraFix v5 — Stage 3: Full PPO Training with Pretrained Weights
===============================================================
Loads Stage 1 (GRU) and Stage 2 (GATConv, trunk) checkpoints, then runs
full PPO with differential learning rates protecting pretrained components.

Differential LRs:
  GRU       1e-4  (protected)
  GATConv   2e-4  (semi-protected)
  Trunk     3e-4
  Actor     3e-4
  Critic    3e-4

Reward : compute_reward() from trafix_v2.py (pressure + queue + throughput +
          fairness + phase stability + wait penalty + green wave)

Saves  : checkpoints/stage3_ep{N}.pt  every 100 episodes
         checkpoints/trafix_v5_final.pt  at end

Kullanım:
  python stage3_train_ppo.py
  python stage3_train_ppo.py --episodes 1000 --gui
"""

import os
import sys
import math
import time
import logging
import argparse
import random
from pathlib import Path
from collections import deque
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

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
    from backend.ai.trafix_v2 import (
        parse_sumo_observations,
        compute_reward,
        compute_gae,
        NUM_NODE_FEATURES,
        RewardWeights,
    )
except ImportError:
    try:
        from train_v2 import TrainConfig
        from trafix_v2 import (
            parse_sumo_observations,
            compute_reward,
            compute_gae,
            NUM_NODE_FEATURES,
            RewardWeights,
        )
    except ImportError:
        sys.path.insert(0, str(_PROJECT_ROOT / "backend" / "ai"))
        from train_v2 import TrainConfig
        from trafix_v2 import (
            parse_sumo_observations,
            compute_reward,
            compute_gae,
            NUM_NODE_FEATURES,
            RewardWeights,
        )


# ══════════════════════════════════════════════════
#  Sabitler
# ══════════════════════════════════════════════════

OBS_DIM = NUM_NODE_FEATURES         # 10 after one-hot phase refactor
NUM_PHASES = 4
T_WINDOW = 10                       # GRU geçmiş penceresi

CHECKPOINTS_DIR = _SCRIPT_DIR / "checkpoints"
STAGE1_CHECKPOINT = CHECKPOINTS_DIR / "stage1_gru.pt"
STAGE2_GATCONV_CHECKPOINT = CHECKPOINTS_DIR / "stage2_gatconv.pt"
STAGE2_TRUNK_CHECKPOINT = CHECKPOINTS_DIR / "stage2_trunk.pt"
FINAL_CHECKPOINT = CHECKPOINTS_DIR / "trafix_v5_final.pt"
DEFAULT_SUMO_CFG = str(_PROJECT_ROOT / "sumo" / "training.sumocfg")
DEFAULT_NET_FILE = str(_PROJECT_ROOT / "sumo" / "map.net.xml")


# ══════════════════════════════════════════════════
#  Rollout Buffer
# ══════════════════════════════════════════════════

class RolloutBuffer:
    """Stores (obs_window, action, log_prob, reward, value) tuples."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.obs_windows: List[torch.Tensor] = []   # each: [T, J, obs_dim]
        self.actions:     List[torch.Tensor] = []   # each: [J]
        self.log_probs:   List[torch.Tensor] = []   # each: [J]
        self.rewards:     List[torch.Tensor] = []   # each: (N,) per-node tensor
        self.values:      List[torch.Tensor] = []   # each: scalar tensor

    def add(
        self,
        obs_window: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
    ):
        self.obs_windows.append(obs_window)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value.detach())

    def __len__(self) -> int:
        return len(self.rewards)


# ══════════════════════════════════════════════════
#  PPO Güncelleme
# ══════════════════════════════════════════════════

def ppo_update(
    model: TraFixV5,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    next_value: torch.Tensor,
    clip_eps: float,
    gamma: float,
    gae_lambda: float,
    entropy_coef: float,
    value_loss_coef: float,
    ppo_epochs: int,
    minibatch_size: int,
    device: torch.device,
    max_log_ratio: float = 2.0,
    value_clip_eps: float = 0.2,
    target_kl: float = 0.015,
) -> Dict[str, float]:
    """Runs ppo_epochs passes with ratio clamping, value clipping, KL early stop."""

    # ── GAE ──
    rewards_t = [r.to(device) for r in buffer.rewards]
    values_t  = [v.to(device) for v in buffer.values]
    next_val  = next_value.to(device).detach()

    advantages, returns = compute_gae(rewards_t, values_t, next_val, gamma, gae_lambda)
    advantages = advantages.to(device)
    returns = returns.to(device)

    # Stack rollout tensors for batched evaluation
    obs_batch      = torch.stack(buffer.obs_windows).to(device)         # [N, T, J, obs_dim]
    actions_batch  = torch.stack(buffer.actions).to(device)             # [N, J]
    old_lp_batch   = torch.stack(buffer.log_probs).detach().to(device)  # [N, J]
    old_val_batch  = torch.cat(buffer.values).to(device)                # [N]

    N = obs_batch.shape[0]
    all_indices = list(range(N))

    metrics = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "total": 0.0}
    update_count = 0
    kl_exceeded = False

    for _ in range(ppo_epochs):
        if kl_exceeded:
            break
        random.shuffle(all_indices)
        for start in range(0, N, minibatch_size):
            idx = all_indices[start : start + minibatch_size]
            if not idx:
                continue

            mb_obs     = obs_batch[idx]          # [mb, T, J, obs_dim]
            mb_acts    = actions_batch[idx]      # [mb, J]
            mb_old     = old_lp_batch[idx]       # [mb, J]
            mb_adv     = advantages[idx]         # [mb, J]
            mb_ret     = returns[idx]            # [mb, J]
            mb_old_val = old_val_batch[idx]      # [mb]

            # Evaluate under current policy
            new_lp, ent, value = model.evaluate_actions(mb_obs, mb_acts)
            # new_lp: [mb, J], ent: [mb, J], value: [mb, 1]

            # Clamp log-ratio before exp to prevent ratio explosion
            log_ratio = (new_lp - mb_old).clamp(-max_log_ratio, max_log_ratio)
            ratio = torch.exp(log_ratio)                         # [mb, J]

            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Clipped value objective (PPO-style)
            v_new = value.squeeze(-1)                            # [mb]
            ret_target = mb_ret.mean(dim=-1)                     # [mb]
            v_clipped = mb_old_val + (v_new - mb_old_val).clamp(-value_clip_eps, value_clip_eps)
            vf_loss1 = (v_new - ret_target).pow(2)
            vf_loss2 = (v_clipped - ret_target).pow(2)
            value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

            entropy_loss = ent.mean()

            total_loss = (
                policy_loss
                + value_loss_coef * value_loss
                - entropy_coef * entropy_loss
            )

            # Skip minibatch if loss is NaN or Inf
            if not torch.isfinite(total_loss):
                continue

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            metrics["policy"]  += policy_loss.item()
            metrics["value"]   += value_loss.item()
            metrics["entropy"] += entropy_loss.item()
            metrics["total"]   += total_loss.item()
            update_count += 1

            # KL early stopping
            with torch.no_grad():
                approx_kl = 0.5 * log_ratio.pow(2).mean().item()
            if approx_kl > target_kl:
                kl_exceeded = True
                break

    denom = max(update_count, 1)
    return {k: v / denom for k, v in metrics.items()}


# ══════════════════════════════════════════════════
#  Checkpoint kayıt
# ══════════════════════════════════════════════════

def save_checkpoint(model: TraFixV5, optimizer: optim.Optimizer, episode: int, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode": episode,
        },
        str(path),
    )


# ══════════════════════════════════════════════════
#  Ana Eğitim Döngüsü
# ══════════════════════════════════════════════════

def train(args: argparse.Namespace):
    # ── Prerequisite checks ──
    missing = []
    if not STAGE1_CHECKPOINT.exists():
        missing.append(f"  stage1_gru.pt missing: {STAGE1_CHECKPOINT}")
    if not STAGE2_GATCONV_CHECKPOINT.exists():
        missing.append(f"  stage2_gatconv.pt missing: {STAGE2_GATCONV_CHECKPOINT}")
    if not STAGE2_TRUNK_CHECKPOINT.exists():
        missing.append(f"  stage2_trunk.pt missing: {STAGE2_TRUNK_CHECKPOINT}")
    if missing:
        sys.exit(
            "Stage 1 and Stage 2 checkpoints not found. "
            "Run stage1_pretrain_gru.py and stage2_pretrain_gatconv.py first.\n"
            + "\n".join(missing)
        )

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    logs_dir = _SCRIPT_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(str(logs_dir / "stage3_ppo.log"), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("=" * 60)
    logging.info("  TraFix v5 — Stage 3: Full PPO Training")
    logging.info("=" * 60)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"  Device: {device}")

    # ── Model ──
    model = TraFixV5(obs_dim=OBS_DIM, num_phases=NUM_PHASES).to(device)

    # Load pretrained weights in order: GRU → GATConv → trunk
    model.temporal_enc.load_state_dict(
        torch.load(str(STAGE1_CHECKPOINT), map_location=device, weights_only=True)
    )
    model.graph_enc.load_state_dict(
        torch.load(str(STAGE2_GATCONV_CHECKPOINT), map_location=device, weights_only=True)
    )
    model.trunk.load_state_dict(
        torch.load(str(STAGE2_TRUNK_CHECKPOINT), map_location=device, weights_only=True)
    )
    logging.info("  Pretrained weights loaded: GRU, GATConv, trunk")
    logging.info(f"  Actor heads and critic head: randomly initialized")
    logging.info(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Differential learning rates ──
    optimizer = optim.Adam(
        [
            {"params": model.temporal_enc.parameters(), "lr": 1e-4},
            {"params": model.graph_enc.parameters(),    "lr": 2e-4},
            {"params": model.trunk.parameters(),        "lr": args.lr},
            {"params": model.actor_heads.parameters(),  "lr": args.lr},
            {"params": model.critic_head.parameters(),  "lr": args.lr},
        ]
    )
    # Store base LRs for cosine decay
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    # ── Optional encoder freeze (warm start) ──
    if args.freeze_episodes > 0:
        for p in model.temporal_enc.parameters():
            p.requires_grad_(False)
        for p in model.graph_enc.parameters():
            p.requires_grad_(False)
        logging.info(f"  Pretrained encoders frozen for first {args.freeze_episodes} episodes")

    # ── SUMO environment ──
    env_cfg = TrainConfig()
    env_cfg.sumo_cfg = args.sumo_cfg
    env_cfg.gui = args.gui
    env_cfg.seed = args.seed
    env_cfg.decision_interval = args.decision_interval
    env_cfg.warmup_steps = 50
    env_cfg.max_steps_per_episode = args.max_steps
    env_cfg.num_actions = NUM_PHASES
    env_cfg.rollout_length = args.minibatch_size
    env = ScenarioEnvironment(env_cfg)

    # Stage 3 uses unseeded generator for maximum generalisation
    generator = ScenarioGenerator(
        net_file=args.net_file,
        output_dir=str(_SCRIPT_DIR / "scenarios"),
        seed=None,
    )

    reward_history = deque(maxlen=50)
    best_reward = -math.inf

    for episode in range(args.episodes):
        # ── Unfreeze pretrained encoders after warm-up ──
        if args.freeze_episodes > 0 and episode == args.freeze_episodes:
            for p in model.temporal_enc.parameters():
                p.requires_grad_(True)
            for p in model.graph_enc.parameters():
                p.requires_grad_(True)
            logging.info(f"  Episode {episode}: Pretrained encoders unfrozen")

        # ── Cosine LR decay ──
        progress = episode / max(args.episodes - 1, 1)
        lr_scale = (args.lr_min / args.lr
                    + 0.5 * (1.0 - args.lr_min / args.lr)
                    * (1.0 + math.cos(math.pi * progress)))
        for pg, base_lr in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = base_lr * lr_scale
        current_lr = optimizer.param_groups[-1]["lr"]  # trunk/actor/critic LR

        scenario_type, route_file = generator.sample(episode)
        env.set_route_file(route_file)

        episode_start = time.time()
        episode_rewards = []
        episode_queues: List[float] = []
        episode_waits:  List[float] = []
        episode_entropies: List[torch.Tensor] = []

        model.train()

        try:
            env.start(episode=episode)
            num_nodes = env.num_nodes

            obs_list = env.get_observations()
            x = parse_sumo_observations(obs_list, device=device)  # [J, obs_dim]
            window: deque = deque([x.detach()] * T_WINDOW, maxlen=T_WINDOW)

            buffer = RolloutBuffer()
            prev_obs = None
            prev_actions = None
            done = False

            ppo_metrics: Dict[str, float] = {"policy": 0.0, "value": 0.0,
                                              "entropy": 0.0, "total": 0.0}
            update_count = 0

            while not done:
                # ── Build observation window ──
                window_tensor = torch.stack(list(window))           # [T, J, obs_dim]
                obs_input = window_tensor.unsqueeze(0).to(device)   # [1, T, J, obs_dim]

                # ── Select actions ──
                with torch.no_grad():
                    actions, log_probs, value = model.get_action(obs_input)
                    # actions: [1, J], log_probs: [1, J], value: [1, 1]

                actions_1d = actions.squeeze(0)   # [J] for env

                # ── Step environment ──
                next_obs_list, done = env.step(actions_1d)
                x_next = parse_sumo_observations(next_obs_list, device=device)

                # ── Reward ──
                reward = compute_reward(
                    current_obs=next_obs_list,
                    previous_obs=prev_obs,
                    previous_actions=prev_actions,
                    current_actions=actions_1d,
                ).to(device)

                episode_rewards.append(reward.mean().item())

                # Track queue and wait metrics from obs
                q_mean = float(x_next[:, 4].mean())  # queue_length col
                w_mean = float(x_next[:, 9].mean())  # phase_duration col index 9 after one-hot
                episode_queues.append(q_mean)
                episode_waits.append(w_mean)

                # ── Buffer ──
                buffer.add(
                    obs_window=window_tensor.detach(),
                    action=actions_1d.detach(),
                    log_prob=log_probs.squeeze(0).detach(),  # [J]
                    reward=reward.detach(),
                    value=value.squeeze(0).detach(),         # [1]
                )

                # ── PPO update when buffer is full ──
                if len(buffer) >= args.rollout_length:
                    next_window = torch.stack(
                        list(window)[-(T_WINDOW - 1):] + [x_next.detach()]
                    ).unsqueeze(0).to(device)

                    with torch.no_grad():
                        _, _, next_val = model.get_action(next_window)

                    step_metrics = ppo_update(
                        model=model,
                        optimizer=optimizer,
                        buffer=buffer,
                        next_value=next_val.squeeze(0),
                        clip_eps=args.clip_eps,
                        gamma=args.gamma,
                        gae_lambda=args.gae_lambda,
                        entropy_coef=args.entropy_coef,
                        value_loss_coef=args.value_loss_coef,
                        ppo_epochs=args.ppo_epochs,
                        minibatch_size=args.minibatch_size,
                        device=device,
                        max_log_ratio=args.max_log_ratio,
                        value_clip_eps=args.value_clip_eps,
                        target_kl=args.target_kl,
                    )
                    for k in ppo_metrics:
                        ppo_metrics[k] += step_metrics[k]
                    update_count += 1
                    buffer.clear()

                # Track per-junction entropy
                with torch.no_grad():
                    _, ent, _ = model.evaluate_actions(obs_input, actions)
                episode_entropies.append(ent.squeeze(0).detach())  # [J]

                prev_obs = next_obs_list
                prev_actions = actions_1d
                window.append(x_next.detach())

            # ── Flush remaining buffer ──
            if len(buffer) > 1:
                next_window = torch.stack(
                    list(window)[-(T_WINDOW - 1):] + [x_next.detach()]
                ).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, _, next_val = model.get_action(next_window)

                step_metrics = ppo_update(
                    model=model,
                    optimizer=optimizer,
                    buffer=buffer,
                    next_value=next_val.squeeze(0),
                    clip_eps=args.clip_eps,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    entropy_coef=args.entropy_coef,
                    value_loss_coef=args.value_loss_coef,
                    ppo_epochs=args.ppo_epochs,
                    minibatch_size=args.minibatch_size,
                    device=device,
                    max_log_ratio=args.max_log_ratio,
                    value_clip_eps=args.value_clip_eps,
                    target_kl=args.target_kl,
                )
                for k in ppo_metrics:
                    ppo_metrics[k] += step_metrics[k]
                update_count += 1

        finally:
            env.close()

        # ── Episode statistics ──
        mean_reward = sum(episode_rewards) / max(len(episode_rewards), 1)
        mean_queue  = sum(episode_queues)  / max(len(episode_queues), 1)
        mean_wait   = sum(episode_waits)   / max(len(episode_waits), 1)
        elapsed     = time.time() - episode_start

        # Per-junction entropy (mean over timesteps per junction)
        if episode_entropies:
            entropy_per_j = torch.stack(episode_entropies).mean(0)  # [J]
        else:
            entropy_per_j = torch.zeros(NUM_JUNCTIONS)

        reward_history.append(mean_reward)
        rolling_avg = sum(reward_history) / len(reward_history)

        if mean_reward > best_reward:
            best_reward = mean_reward

        if (episode + 1) % 10 == 0:
            denom = max(update_count, 1)
            ent_str = " ".join(f"J{j}={entropy_per_j[j].item():.3f}"
                                for j in range(NUM_JUNCTIONS))
            logging.info(
                f"  Ep {episode + 1:>4d}/{args.episodes} | "
                f"scenario={scenario_type.value} | "
                f"R={mean_reward:+.4f} (avg50={rolling_avg:+.4f}) | "
                f"Q={mean_queue:.3f} W={mean_wait:.3f} | "
                f"π={ppo_metrics['policy']/denom:.4f} "
                f"V={ppo_metrics['value']/denom:.4f} | "
                f"H=[{ent_str}] | LR={current_lr:.2e} | {elapsed:.1f}s | "
                f"{generator.last_summary}"
            )

        # ── Periodic checkpoint ──
        if (episode + 1) % 100 == 0:
            ckpt_path = CHECKPOINTS_DIR / f"stage3_ep{episode + 1}.pt"
            save_checkpoint(model, optimizer, episode, ckpt_path)
            logging.info(f"  → Checkpoint saved: {ckpt_path}")

    # ── Final checkpoint ──
    save_checkpoint(model, optimizer, args.episodes - 1, FINAL_CHECKPOINT)
    logging.info(f"  Final model saved → {FINAL_CHECKPOINT}")
    logging.info(f"  Best episode reward: {best_reward:.6f}")
    print("Stage 3 complete. Final model saved.")


# ══════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TraFix v5 Stage 3 — full PPO training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sumo-cfg", default=DEFAULT_SUMO_CFG)
    parser.add_argument("--net-file", default=DEFAULT_NET_FILE)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--decision-interval", type=int, default=10)
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Base LR for trunk, actor heads, critic head")
    parser.add_argument("--lr-min", type=float, default=1e-5,
                        help="Minimum LR for cosine decay")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--max-log-ratio", type=float, default=2.0,
                        help="Clamp log(new/old) to [-X, X] before exp")
    parser.add_argument("--value-clip-eps", type=float, default=0.2,
                        help="Epsilon for clipped value objective")
    parser.add_argument("--target-kl", type=float, default=0.015,
                        help="Approx KL threshold for early stopping PPO epochs")
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--value-loss-coef", type=float, default=0.25)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--freeze-episodes", type=int, default=0,
                        help="Freeze pretrained encoders for this many episodes (0=no freeze)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
