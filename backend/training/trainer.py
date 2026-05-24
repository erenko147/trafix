"""
Main PPO training loop.

train(cfg) runs the full episode loop; _save_checkpoint() handles persistence.
Entry point: scripts/train.py (or `python -m backend.training.trainer`).
"""

import json
import logging
import os
import random
import sys
import time
from collections import deque
from dataclasses import asdict
from typing import Optional

import torch
import torch.optim as optim

from backend.ai.agent import CoordinatedPPOAgent
from backend.ai.observation import parse_sumo_observations, NUM_NODE_FEATURES
from backend.ai.reward import compute_reward, train_step
from backend.training.buffer import RolloutBuffer
from backend.training.config import TrainConfig
from backend.training.environment import SumoEnvironment, build_edge_index
from backend.training.logger import TrainingLogger
from backend.training.scheduler import CosineWarmupScheduler

# ── Dynamic demand (optional) ─────────────────────────────────────────────────

try:
    from sumo.generate_demand import generate_dynamic_demand as _gen_demand
    _HAS_DYNAMIC_DEMAND = True
except ImportError:
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sumo"))
        from generate_demand import generate_dynamic_demand as _gen_demand
        _HAS_DYNAMIC_DEMAND = True
    except ImportError:
        _HAS_DYNAMIC_DEMAND = False


def _save_checkpoint(
    agent:       CoordinatedPPOAgent,
    optimizer:   optim.Optimizer,
    episode:     int,
    best_reward: float,
    path:        str,
):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict":     agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode":              episode,
            "best_reward":          best_reward,
            "config": {
                "hidden_dim":    agent.hidden_dim,
                "num_actions":   agent.num_actions,
                "entropy_coef":  agent.entropy_coef,
            },
        },
        path,
    )


def train(cfg: TrainConfig):
    logger = TrainingLogger(cfg.output_dir)
    logging.info("=" * 60)
    logging.info("  TraFix — Training Start")
    logging.info("=" * 60)
    logging.info(f"  Config: {json.dumps(asdict(cfg), indent=2, default=str)}")

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"  Device: {device}")

    env = SumoEnvironment(cfg)
    env.start(episode=0)
    num_nodes = env.num_nodes
    env.close()
    logging.info(f"  Junctions: {num_nodes}")

    edge_index = build_edge_index(num_nodes, cfg.net_file).to(device)

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

    start_episode = 0
    if cfg.resume and os.path.exists(cfg.checkpoint_path):
        try:
            ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=True)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                agent.load_state_dict(ckpt["model_state_dict"])
                start_episode = ckpt.get("episode", 0)
                logging.info(f"  Resumed from episode {start_episode}")
            else:
                agent.load_state_dict(ckpt)
                logging.info("  Weights loaded (legacy format)")
        except RuntimeError as e:
            logging.warning(f"  Checkpoint mismatch — starting fresh: {e}")

    optimizer = optim.Adam(agent.parameters(), lr=cfg.lr, eps=cfg.eps)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_episodes=min(20, cfg.episodes // 10),
        total_episodes=cfg.episodes,
        lr_min=cfg.lr_min,
    )

    best_reward    = float("-inf")
    reward_history = deque(maxlen=50)
    no_improve_count    = 0
    current_entropy_coef = cfg.entropy_coef

    logging.info(f"  Start episode  : {start_episode}")
    logging.info(f"  Total episodes : {cfg.episodes}")
    logging.info(f"  Parameters     : {sum(p.numel() for p in agent.parameters()):,}")
    logging.info("=" * 60)

    for episode in range(start_episode, cfg.episodes):
        episode_start = time.time()

        current_entropy_coef = max(
            cfg.entropy_coef_min,
            cfg.entropy_coef * (cfg.entropy_decay ** episode),
        )
        agent.entropy_coef = current_entropy_coef
        current_lr         = scheduler.step(episode)

        if _HAS_DYNAMIC_DEMAND:
            try:
                _gen_demand()
            except Exception as _e:
                logging.warning(f"  Dynamic demand failed, using previous file: {_e}")

        try:
            env.start(episode=episode)
        except FileNotFoundError as e:
            logging.error(str(e))
            sys.exit(1)
        except Exception as e:
            logging.error(f"Episode {episode}: SUMO failed to start: {e}")
            consecutive_failures = getattr(train, "_failures", 0) + 1
            train._failures = consecutive_failures
            if consecutive_failures >= 3:
                logging.error("3 consecutive failures — stopping training.")
                sys.exit(1)
            continue
        else:
            train._failures = 0

        agent.train()
        buffer = RolloutBuffer()

        obs          = env.get_observations()
        prev_obs     = None
        prev_actions = None

        episode_rewards  = []
        episode_metrics  = []
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        update_count      = 0
        step              = 0
        done              = False

        while not done:
            x                   = parse_sumo_observations(obs, device)
            actions, log_probs, value = agent.select_actions(x, edge_index)
            next_obs, done      = env.step(actions)

            reward = compute_reward(
                current_obs=next_obs,
                previous_obs=prev_obs,
                previous_actions=prev_actions,
                current_actions=actions,
                weights=cfg.reward_weights,
            ).to(device)

            episode_rewards.append(reward.mean().item())
            buffer.add(x, actions, log_probs, reward, value)

            if len(buffer) >= cfg.rollout_length:
                with torch.no_grad():
                    next_x        = parse_sumo_observations(next_obs, device)
                    _, next_value = agent(next_x, edge_index)

                rollout = buffer.to_dict(edge_index, next_value.detach())
                agent.train()
                losses = train_step(agent, optimizer, rollout, cfg.ppo_epochs)

                total_policy_loss += losses["policy"]
                total_value_loss  += losses["value"]
                total_entropy     += losses["entropy"]
                update_count      += 1
                buffer.clear()

            if step % 10 == 0:
                episode_metrics.append(env.get_metrics())

            prev_obs     = obs
            prev_actions = actions
            obs          = next_obs
            step         += 1

        if len(buffer) > 1:
            with torch.no_grad():
                x             = parse_sumo_observations(obs, device)
                _, next_value = agent(x, edge_index)
            rollout = buffer.to_dict(edge_index, next_value.detach())
            agent.train()
            losses = train_step(agent, optimizer, rollout, cfg.ppo_epochs)
            total_policy_loss += losses["policy"]
            total_value_loss  += losses["value"]
            total_entropy     += losses["entropy"]
            update_count      += 1

        env.close()

        episode_time = time.time() - episode_start
        mean_reward  = sum(episode_rewards) / max(len(episode_rewards), 1)
        std_reward   = (
            sum((r - mean_reward) ** 2 for r in episode_rewards)
            / max(len(episode_rewards), 1)
        ) ** 0.5

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

        log_data = {
            "reward_mean":  mean_reward,
            "reward_std":   std_reward,
            "policy_loss":  total_policy_loss / max(update_count, 1),
            "value_loss":   total_value_loss  / max(update_count, 1),
            "entropy":      total_entropy     / max(update_count, 1),
            "lr":           current_lr,
            "entropy_coef": current_entropy_coef,
            **avg_metrics,
        }
        logger.log_episode(episode, log_data)

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

        if mean_reward > best_reward:
            best_reward      = mean_reward
            no_improve_count = 0
            _save_checkpoint(
                agent, optimizer, episode, best_reward,
                os.path.join(cfg.output_dir, "best_model.pth"),
            )
        else:
            no_improve_count += 1

        if (episode + 1) % cfg.save_interval == 0:
            path = os.path.join(cfg.output_dir, f"checkpoint_ep{episode + 1}.pth")
            _save_checkpoint(agent, optimizer, episode, best_reward, path)
            _save_checkpoint(agent, optimizer, episode, best_reward, cfg.checkpoint_path)
            logging.info(f"  → Checkpoint saved: episode {episode + 1}")

    _save_checkpoint(agent, optimizer, cfg.episodes - 1, best_reward, cfg.checkpoint_path)
    _save_checkpoint(
        agent, optimizer, cfg.episodes - 1, best_reward,
        os.path.join(cfg.output_dir, "final_model.pth"),
    )
    logger.save_history()

    logging.info("=" * 60)
    logging.info("  TRAINING COMPLETE")
    logging.info(f"  Best reward     : {best_reward:.6f}")
    logging.info(f"  Last-50 average : {rolling_avg:.6f}")
    logging.info(f"  Checkpoint      : {cfg.checkpoint_path}")
    logging.info(f"  Log directory   : {cfg.output_dir}")
    logging.info("=" * 60)
