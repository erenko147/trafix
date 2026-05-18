"""
TraFix v6 — Stage 3: Full PPO Training with Pretrained Weights
===============================================================
Loads Stage 1 (GRU) and Stage 2 (GATConv, trunk) checkpoints, then runs
full PPO with differential learning rates.

v6 changes vs v5:
  NUM_PHASES = 6
  OBS_DIM = 20
  Governor: num_phases=6, pressure_thresh=0.35, freeze_episodes=100
  entropy_coef = 0.01 (was 0.005 — prevent premature collapse over 6 phases)
  episodes = 2000

Saves:
  checkpoints/stage3_ep{N}.pt  every 100 episodes
  checkpoints/trafix_v6_final.pt  at end

Usage:
  python trafix_v6/stage3_train_ppo.py
  python trafix_v6/stage3_train_ppo.py --episodes 2000 --gui
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
    raise ImportError("SUMO TraCI not found. Set SUMO_HOME.")

# ── Path setup ──
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

from trafix_v6.trafix_v6 import TraFixV6, NUM_JUNCTIONS
from scenario_generator import ScenarioGenerator, ScenarioEnvironment
from rule_governor import RuleGovernor, sample_governed, evaluate_governed

try:
    from backend.ai.train_v2 import TrainConfig
    from backend.ai.trafix_v2 import (
        parse_sumo_observations, compute_reward, compute_gae,
        NUM_NODE_FEATURES, RewardWeights,
    )
except ImportError:
    try:
        from train_v2 import TrainConfig
        from trafix_v2 import (
            parse_sumo_observations, compute_reward, compute_gae,
            NUM_NODE_FEATURES, RewardWeights,
        )
    except ImportError:
        sys.path.insert(0, str(_PROJECT_ROOT / "backend" / "ai"))
        from train_v2 import TrainConfig
        from trafix_v2 import (
            parse_sumo_observations, compute_reward, compute_gae,
            NUM_NODE_FEATURES, RewardWeights,
        )


# ══════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════

OBS_DIM = NUM_NODE_FEATURES   # 20
NUM_PHASES = 6
T_WINDOW = 30

CHECKPOINTS_DIR = _SCRIPT_DIR / "checkpoints"
STAGE1_CHECKPOINT = CHECKPOINTS_DIR / "stage1_gru.pt"
STAGE2_GATCONV_CHECKPOINT = CHECKPOINTS_DIR / "stage2_gatconv.pt"
STAGE2_TRUNK_CHECKPOINT = CHECKPOINTS_DIR / "stage2_trunk.pt"
FINAL_CHECKPOINT = CHECKPOINTS_DIR / "trafix_v6_final.pt"
DEFAULT_SUMO_CFG = str(_PROJECT_ROOT / "sumo" / "training.sumocfg")
DEFAULT_NET_FILE = str(_PROJECT_ROOT / "sumo" / "map.net.xml")


# ══════════════════════════════════════════════════
#  Rollout Buffer
# ══════════════════════════════════════════════════

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs_windows: List[torch.Tensor] = []
        self.actions:     List[torch.Tensor] = []
        self.log_probs:   List[torch.Tensor] = []
        self.rewards:     List[torch.Tensor] = []
        self.values:      List[torch.Tensor] = []

    def add(self, obs_window, action, log_prob, reward, value):
        self.obs_windows.append(obs_window)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value.detach())

    def __len__(self):
        return len(self.rewards)


# ══════════════════════════════════════════════════
#  PPO Update
# ══════════════════════════════════════════════════

def ppo_update(
    model, optimizer, buffer, next_value,
    clip_eps, gamma, gae_lambda, entropy_coef, value_loss_coef,
    ppo_epochs, minibatch_size, device,
    max_log_ratio=2.0, value_clip_eps=0.2, target_kl=0.015,
    governor=None,
) -> Dict[str, float]:
    rewards_t = [r.to(device) for r in buffer.rewards]
    values_t  = [v.to(device) for v in buffer.values]
    next_val  = next_value.to(device).detach()

    advantages, returns = compute_gae(rewards_t, values_t, next_val, gamma, gae_lambda)
    advantages = advantages.to(device)
    returns = returns.to(device)

    obs_batch     = torch.stack(buffer.obs_windows).to(device)
    actions_batch = torch.stack(buffer.actions).to(device)
    old_lp_batch  = torch.stack(buffer.log_probs).detach().to(device)
    old_val_batch = torch.stack(buffer.values).to(device)   # [T, J]

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

            mb_obs     = obs_batch[idx]
            mb_acts    = actions_batch[idx]
            mb_old     = old_lp_batch[idx]
            mb_adv     = advantages[idx]
            mb_ret     = returns[idx]
            mb_old_val = old_val_batch[idx]

            if governor is not None:
                logits_list, value = model.forward(mb_obs)
                obs_last_batch = mb_obs[:, -1, :, :]
                masked_logits = governor.apply_stateless_batch(logits_list, obs_last_batch)
                new_lp, ent = evaluate_governed(masked_logits, mb_acts)
            else:
                new_lp, ent, value = model.evaluate_actions(mb_obs, mb_acts)

            log_ratio = (new_lp - mb_old).clamp(-max_log_ratio, max_log_ratio)
            ratio = torch.exp(log_ratio)

            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            v_new = value                  # [batch, J]
            ret_target = mb_ret            # [batch, J]
            v_clipped = mb_old_val + (v_new - mb_old_val).clamp(-value_clip_eps, value_clip_eps)
            vf_loss1 = (v_new - ret_target).pow(2)
            vf_loss2 = (v_clipped - ret_target).pow(2)
            value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

            entropy_loss = ent.mean()
            total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_loss

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

            with torch.no_grad():
                approx_kl = 0.5 * log_ratio.pow(2).mean().item()
            if approx_kl > target_kl:
                kl_exceeded = True
                break

    denom = max(update_count, 1)
    return {k: v / denom for k, v in metrics.items()}


def save_checkpoint(model, optimizer, episode, path, best_reward=-math.inf):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode": episode,
            "best_reward": best_reward,
        },
        str(path),
    )


# ══════════════════════════════════════════════════
#  Training Loop
# ══════════════════════════════════════════════════

def train(args):
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            sys.exit(f"Resume checkpoint not found: {resume_path}")
    else:
        missing = []
        for label, path in [
            ("stage1_gru.pt", STAGE1_CHECKPOINT),
            ("stage2_gatconv.pt", STAGE2_GATCONV_CHECKPOINT),
            ("stage2_trunk.pt", STAGE2_TRUNK_CHECKPOINT),
        ]:
            if not path.exists():
                missing.append(f"  {label}: {path}")
        if missing:
            sys.exit(
                "Stage 1 and Stage 2 checkpoints required. "
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
    logging.info("  TraFix v6 — Stage 3: Full PPO Training")
    logging.info(f"  NUM_PHASES={NUM_PHASES}, OBS_DIM={OBS_DIM}")
    logging.info("=" * 60)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"  Device: {device}")

    model = TraFixV6(obs_dim=OBS_DIM, num_phases=NUM_PHASES).to(device)

    optimizer = optim.Adam([
        {"params": model.temporal_enc.parameters(), "lr": 1e-4},
        {"params": model.graph_enc.parameters(),    "lr": 2e-4},
        {"params": model.trunk.parameters(),         "lr": args.lr},
        {"params": model.actor_heads.parameters(),   "lr": args.lr},
        {"params": model.local_critics.parameters(), "lr": args.lr},
        {"params": model.global_critic.parameters(), "lr": args.lr},
    ])
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    start_episode = 0
    best_reward = -math.inf

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_episode = ckpt["episode"] + 1
        best_reward = ckpt.get("best_reward", -math.inf)
        logging.info(f"  Resumed from {args.resume} (ep {ckpt['episode']}), continuing from {start_episode}")
    else:
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

    logging.info(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

    # Freeze pretrained encoders during warm start
    if args.freeze_episodes > 0 and start_episode < args.freeze_episodes:
        for p in model.temporal_enc.parameters():
            p.requires_grad_(False)
        for p in model.graph_enc.parameters():
            p.requires_grad_(False)
        logging.info(f"  Encoders frozen until episode {args.freeze_episodes}")

    env_cfg = TrainConfig()
    env_cfg.sumo_cfg = args.sumo_cfg
    env_cfg.gui = args.gui
    env_cfg.seed = args.seed
    env_cfg.decision_interval = args.decision_interval
    env_cfg.warmup_steps = 50
    env_cfg.max_steps_per_episode = args.max_steps
    env_cfg.num_actions = NUM_PHASES
    env_cfg.rollout_length = args.rollout_length
    env = ScenarioEnvironment(env_cfg)

    generator = ScenarioGenerator(
        net_file=args.net_file,
        output_dir=str(_SCRIPT_DIR / "scenarios"),
        seed=None,
    )

    # Rule governor for 6 phases (Section 15 spec)
    governor = RuleGovernor(
        num_junctions=NUM_JUNCTIONS,
        num_phases=6,
        min_green_s=10.0,
        max_green_s=90.0,
        flicker_window=2,
        flicker_penalty=3.0,
        pressure_boost=1.0,
        pressure_thresh=0.35,
    )

    reward_history = deque(maxlen=50)

    for episode in range(start_episode, args.episodes):
        # Unfreeze after warm-up
        if args.freeze_episodes > 0 and episode == args.freeze_episodes:
            for p in model.temporal_enc.parameters():
                p.requires_grad_(True)
            for p in model.graph_enc.parameters():
                p.requires_grad_(True)
            logging.info(f"  Episode {episode}: Encoders unfrozen")

        # Cosine LR decay
        progress = episode / max(args.episodes - 1, 1)
        lr_scale = (args.lr_min / args.lr
                    + 0.5 * (1.0 - args.lr_min / args.lr)
                    * (1.0 + math.cos(math.pi * progress)))
        for pg, base_lr in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = base_lr * lr_scale
        current_lr = optimizer.param_groups[-1]["lr"]

        scenario_type, route_file = generator.sample(episode)
        env.set_route_file(route_file)

        episode_start = time.time()
        episode_rewards = []
        episode_queues: List[float] = []
        episode_waits:  List[float] = []
        episode_entropies: List[torch.Tensor] = []

        model.train()
        governor.reset()

        try:
            env.start(episode=episode)
            num_nodes = env.num_nodes

            obs_list = env.get_observations()
            x = parse_sumo_observations(obs_list, device=device)
            window: deque = deque([x.detach()] * T_WINDOW, maxlen=T_WINDOW)

            buffer = RolloutBuffer()
            prev_obs = None
            prev_actions = None
            done = False

            ppo_metrics = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "total": 0.0}
            update_count = 0

            while not done:
                window_tensor = torch.stack(list(window))
                obs_input = window_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    logits_list, value = model.forward(obs_input)
                    obs_last = obs_input[0, -1]
                    masked_full = governor.apply(logits_list, obs_last)
                    actions, log_probs = sample_governed(masked_full)
                    _, ent = evaluate_governed(masked_full, actions)

                actions_1d = actions.squeeze(0)
                governor.update_state(actions_1d)

                next_obs_list, done = env.step(actions_1d)
                x_next = parse_sumo_observations(next_obs_list, device=device)

                reward = compute_reward(
                    current_obs=next_obs_list,
                    previous_obs=prev_obs,
                    previous_actions=prev_actions,
                    current_actions=actions_1d,
                ).to(device)

                episode_rewards.append(reward.mean().item())

                # Index 12 = queue, index 19 = duration
                q_mean = float(x_next[:, 12].mean())
                w_mean = float(x_next[:, 19].mean())
                episode_queues.append(q_mean)
                episode_waits.append(w_mean)

                buffer.add(
                    obs_window=window_tensor.detach(),
                    action=actions_1d.detach(),
                    log_prob=log_probs.squeeze(0).detach(),
                    reward=reward.detach(),
                    value=value.squeeze(0).detach(),
                )

                if len(buffer) >= args.rollout_length:
                    next_window = torch.stack(
                        list(window)[-(T_WINDOW - 1):] + [x_next.detach()]
                    ).unsqueeze(0).to(device)
                    with torch.no_grad():
                        _, next_val = model.forward(next_window)

                    step_metrics = ppo_update(
                        model=model, optimizer=optimizer, buffer=buffer,
                        next_value=next_val.squeeze(0),
                        clip_eps=args.clip_eps, gamma=args.gamma,
                        gae_lambda=args.gae_lambda,
                        entropy_coef=args.entropy_coef,
                        value_loss_coef=args.value_loss_coef,
                        ppo_epochs=args.ppo_epochs,
                        minibatch_size=args.minibatch_size,
                        device=device, target_kl=args.target_kl,
                        governor=governor,
                    )
                    for k in ppo_metrics:
                        ppo_metrics[k] += step_metrics[k]
                    update_count += 1
                    buffer.clear()

                episode_entropies.append(ent.squeeze(0).detach())

                prev_obs = next_obs_list
                prev_actions = actions_1d
                window.append(x_next.detach())

            # Flush remaining buffer — compute proper bootstrap value from the
            # last observed state (window already contains x_next as its last
            # frame). Using zeros is wrong for truncated episodes (max_steps).
            if len(buffer) >= 1:
                last_window = torch.stack(list(window)).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, next_val = model.forward(last_window)
                next_val = next_val.squeeze(0).detach()
                step_metrics = ppo_update(
                    model=model, optimizer=optimizer, buffer=buffer,
                    next_value=next_val,
                    clip_eps=args.clip_eps, gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    entropy_coef=args.entropy_coef,
                    value_loss_coef=args.value_loss_coef,
                    ppo_epochs=args.ppo_epochs,
                    minibatch_size=args.minibatch_size,
                    device=device, target_kl=args.target_kl,
                    governor=governor,
                )
                for k in ppo_metrics:
                    ppo_metrics[k] += step_metrics[k]
                update_count += 1

        finally:
            env.close()

        mean_reward = sum(episode_rewards) / max(len(episode_rewards), 1)
        mean_queue  = sum(episode_queues)  / max(len(episode_queues),  1)
        mean_wait   = sum(episode_waits)   / max(len(episode_waits),   1)
        elapsed     = time.time() - episode_start

        if episode_entropies:
            entropy_per_j = torch.stack(episode_entropies).mean(0)
        else:
            entropy_per_j = torch.zeros(NUM_JUNCTIONS)

        reward_history.append(mean_reward)
        rolling_avg = sum(reward_history) / len(reward_history)

        if mean_reward > best_reward:
            best_reward = mean_reward

        if (episode + 1) % 10 == 0:
            denom = max(update_count, 1)
            ent_str = " ".join(f"J{j}={entropy_per_j[j].item():.3f}" for j in range(NUM_JUNCTIONS))
            logging.info(
                f"  Ep {episode + 1:>4d}/{args.episodes} | "
                f"scenario={scenario_type.value} | "
                f"R={mean_reward:+.4f} (avg50={rolling_avg:+.4f}) | "
                f"Q={mean_queue:.3f} W={mean_wait:.3f} | "
                f"π={ppo_metrics['policy']/denom:.4f} V={ppo_metrics['value']/denom:.4f} | "
                f"H=[{ent_str}] | LR={current_lr:.2e} | {elapsed:.1f}s | "
                f"{generator.last_summary}"
            )

        if (episode + 1) % 100 == 0:
            ckpt_path = CHECKPOINTS_DIR / f"stage3_ep{episode + 1}.pt"
            save_checkpoint(model, optimizer, episode, ckpt_path, best_reward)
            logging.info(f"  → Checkpoint saved: {ckpt_path}")

    save_checkpoint(model, optimizer, args.episodes - 1, FINAL_CHECKPOINT, best_reward)
    logging.info(f"  Final model saved → {FINAL_CHECKPOINT}")
    logging.info(f"  Best episode reward: {best_reward:.6f}")
    print("Stage 3 complete. Final model saved.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="TraFix v6 Stage 3 — full PPO training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sumo-cfg", default=DEFAULT_SUMO_CFG)
    parser.add_argument("--net-file", default=DEFAULT_NET_FILE)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--decision-interval", type=int, default=10)
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-loss-coef", type=float, default=0.25)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--freeze-episodes", type=int, default=100)
    parser.add_argument("--max-log-ratio", type=float, default=2.0)
    parser.add_argument("--value-clip-eps", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
