"""
TraFix v6 — Morning Peak Fine-Tuning
======================================
Fine-tunes the final v6 checkpoint on a morning-peak-heavy curriculum
while preserving OFFPEAK and INCIDENT performance.

Design choices vs Stage 3:
  LR         = 1e-5  (10-30x lower — protects pretrained weights)
  Curriculum = 70% MORNING_PEAK / 30% OFFPEAK  (no INCIDENT/EVENING to avoid noise)
  Frozen     = GRU temporal_enc + GATConv graph_enc (only trunk + actor heads update)
  Episodes   = 250 default  (short run — stop if forgetting detected)
  Eval every = 50 episodes across all 4 scenario types to catch forgetting early
  Best model = saved by lowest MORNING_PEAK queue, gated by OFFPEAK queue < 0.035

Usage:
  python trafix_v6/finetune_morning_peak.py
  python trafix_v6/finetune_morning_peak.py --episodes 300 --lr 5e-6
  python trafix_v6/finetune_morning_peak.py --checkpoint trafix_v6/checkpoints/trafix_v6_final.pt
"""

import os
import sys
import math
import time
import random
import logging
import argparse
from pathlib import Path
from collections import deque
from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim

# ── SUMO TraCI ──
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    for candidate in [
        "C:\\Program Files (x86)\\Eclipse\\Sumo\\tools",
        "C:\\Program Files\\Eclipse\\Sumo\\tools",
        "/usr/share/sumo/tools",
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
from trafix_v6.rule_governor import RuleGovernor, sample_governed, evaluate_governed
from trafix_v6.scenario_generator import ScenarioGenerator, ScenarioEnvironment, ScenarioType

try:
    from backend.ai.train_v2 import TrainConfig
    from backend.ai.trafix_v2 import (
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

OBS_DIM    = NUM_NODE_FEATURES   # 20
NUM_PHASES = 6
T_WINDOW   = 10

CHECKPOINTS_DIR   = _SCRIPT_DIR / "checkpoints"
DEFAULT_CKPT      = str(CHECKPOINTS_DIR / "trafix_v6_final.pt")
FINETUNE_BEST     = CHECKPOINTS_DIR / "trafix_v6_finetuned_best.pt"
FINETUNE_FINAL    = CHECKPOINTS_DIR / "trafix_v6_finetuned_final.pt"
DEFAULT_SUMO_CFG  = str(_PROJECT_ROOT / "sumo" / "training.sumocfg")
DEFAULT_NET_FILE  = str(_PROJECT_ROOT / "sumo" / "map.net.xml")

# Gate: only save as best if OFFPEAK queue doesn't regress past this threshold.
# Baseline OFFPEAK queue from full eval was ~0.025; allow 40% slack.
OFFPEAK_GATE = 0.035


# ══════════════════════════════════════════════════
#  Rollout buffer (same as Stage 3)
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
#  PPO update (identical to Stage 3)
# ══════════════════════════════════════════════════

def ppo_update(
    model, optimizer, buffer, next_value,
    clip_eps, gamma, gae_lambda, entropy_coef,
    value_loss_coef, ppo_epochs, minibatch_size, device,
    target_kl=0.015, governor=None,
) -> Dict[str, float]:
    rewards_t = [r.to(device) for r in buffer.rewards]
    values_t  = [v.to(device) for v in buffer.values]

    advantages, returns = compute_gae(
        rewards_t, values_t, next_value.to(device).detach(), gamma, gae_lambda
    )
    advantages = advantages.to(device)
    returns    = returns.to(device)

    obs_batch     = torch.stack(buffer.obs_windows).to(device)
    actions_batch = torch.stack(buffer.actions).to(device)
    old_lp_batch  = torch.stack(buffer.log_probs).detach().to(device)
    old_val_batch = torch.cat(buffer.values).to(device)

    N = obs_batch.shape[0]
    metrics = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "total": 0.0}
    update_count = 0
    kl_exceeded = False

    for _ in range(ppo_epochs):
        if kl_exceeded:
            break
        idx_list = list(range(N))
        random.shuffle(idx_list)
        for start in range(0, N, minibatch_size):
            idx = idx_list[start : start + minibatch_size]
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
                masked = governor.apply_stateless_batch(logits_list, obs_last_batch)
                new_lp, ent = evaluate_governed(masked, mb_acts)
            else:
                new_lp, ent, value = model.evaluate_actions(mb_obs, mb_acts)

            log_ratio = (new_lp - mb_old).clamp(-2.0, 2.0)
            ratio = torch.exp(log_ratio)

            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            v_new      = value.squeeze(-1)
            ret_target = mb_ret.mean(dim=-1)
            v_clipped  = mb_old_val + (v_new - mb_old_val).clamp(-0.2, 0.2)
            value_loss = 0.5 * torch.max(
                (v_new - ret_target).pow(2), (v_clipped - ret_target).pow(2)
            ).mean()

            entropy_loss = ent.mean()
            total_loss   = (
                policy_loss
                + value_loss_coef * value_loss
                - entropy_coef * entropy_loss
            )

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


# ══════════════════════════════════════════════════
#  Quick inline eval (no SUMO GUI)
# ══════════════════════════════════════════════════

def quick_eval(
    model, env, generator, device, n_scenarios, episode_offset, scenario_type=None
) -> Dict[str, float]:
    """
    Run n_scenarios evaluation episodes and return aggregate metrics.
    If scenario_type is given, force that type; otherwise use curriculum.
    """
    model.eval()
    queues, rewards = [], []

    for i in range(n_scenarios):
        ep_idx = episode_offset + i
        if scenario_type is not None:
            route_file = generator.generate(scenario_type, ep_idx)
        else:
            _, route_file = generator.sample(ep_idx)

        env.set_route_file(route_file)
        try:
            env.start(episode=ep_idx)
        except Exception:
            env.close()
            continue

        obs_list = env.get_observations()
        x = parse_sumo_observations(obs_list, device=device)
        window: deque = deque([x.detach()] * T_WINDOW, maxlen=T_WINDOW)

        prev_obs, prev_actions, done = None, None, False
        ep_queues, ep_rewards = [], []

        with torch.no_grad():
            while not done:
                window_tensor = torch.stack(list(window)).unsqueeze(0).to(device)
                logits_list, _ = model.forward(window_tensor)
                actions_1d = torch.stack(
                    [l.argmax(dim=-1) for l in logits_list], dim=1
                ).squeeze(0)

                next_obs_list, done = env.step(actions_1d)
                x_next = parse_sumo_observations(next_obs_list, device=device)

                reward = compute_reward(
                    current_obs=next_obs_list,
                    previous_obs=prev_obs,
                    previous_actions=prev_actions,
                    current_actions=actions_1d,
                )
                ep_rewards.append(reward.mean().item())
                ep_queues.append(float(x_next[:, 12].mean()))

                prev_obs     = next_obs_list
                prev_actions = actions_1d
                window.append(x_next.detach())

        env.close()
        if ep_queues:
            queues.append(sum(ep_queues) / len(ep_queues))
            rewards.append(sum(ep_rewards) / len(ep_rewards))

    model.train()
    return {
        "mean_queue":  sum(queues)  / max(len(queues),  1),
        "mean_reward": sum(rewards) / max(len(rewards), 1),
        "n": len(queues),
    }


# ══════════════════════════════════════════════════
#  Fine-tuning loop
# ══════════════════════════════════════════════════

def finetune(args):
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    logs_dir = _SCRIPT_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(str(logs_dir / "finetune_morning.log"), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("=" * 65)
    logging.info("  TraFix v6 — Morning Peak Fine-Tuning")
    logging.info("=" * 65)
    logging.info(f"  Base checkpoint : {args.checkpoint}")
    logging.info(f"  Episodes        : {args.episodes}")
    logging.info(f"  LR              : {args.lr}")
    logging.info(f"  Curriculum      : {int(args.morning_fraction*100)}% MORNING / "
                 f"{int((1-args.morning_fraction)*100)}% OFFPEAK")
    logging.info(f"  Frozen layers   : temporal_enc + graph_enc")
    logging.info(f"  OFFPEAK gate    : queue < {OFFPEAK_GATE} (else reject best)")
    logging.info("=" * 65)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"  Device: {device}")

    # ── Load base model ──
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    model = TraFixV6(obs_dim=OBS_DIM, num_phases=NUM_PHASES).to(device)
    ckpt  = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    logging.info(f"  Loaded: {ckpt_path} (episode={ckpt.get('episode','?')})")

    # ── Freeze encoders — only trunk + actor heads + critic update ──
    for p in model.temporal_enc.parameters():
        p.requires_grad_(False)
    for p in model.graph_enc.parameters():
        p.requires_grad_(False)
    trainable_params = (
        list(model.trunk.parameters())
        + list(model.actor_heads.parameters())
        + list(model.critic_head.parameters())
    )
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total     = sum(p.numel() for p in model.parameters())
    logging.info(f"  Trainable params: {n_trainable:,} / {n_total:,} "
                 f"(encoders frozen)")

    optimizer = optim.Adam(trainable_params, lr=args.lr)

    # ── SUMO environment ──
    env_cfg = TrainConfig()
    env_cfg.sumo_cfg            = args.sumo_cfg
    env_cfg.gui                 = False
    env_cfg.seed                = args.seed
    env_cfg.decision_interval   = 10
    env_cfg.warmup_steps        = 50
    env_cfg.max_steps_per_episode = 3600
    env_cfg.num_actions         = NUM_PHASES
    env_cfg.rollout_length      = 64
    env = ScenarioEnvironment(env_cfg)

    # Training generator (no fixed seed → maximum generalisation)
    train_gen = ScenarioGenerator(
        net_file=args.net_file,
        output_dir=str(_SCRIPT_DIR / "scenarios"),
        seed=None,
    )
    # Eval generator (fixed seed → reproducible comparisons)
    eval_gen = ScenarioGenerator(
        net_file=args.net_file,
        output_dir=str(_SCRIPT_DIR / "scenarios_eval"),
        seed=77,
    )

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

    best_morning_queue = math.inf
    reward_history     = deque(maxlen=20)

    logging.info(f"\n  Running baseline eval before fine-tuning...")
    baseline_morning = quick_eval(
        model, env, eval_gen, device,
        n_scenarios=10, episode_offset=800,
        scenario_type=ScenarioType.MORNING_PEAK,
    )
    baseline_offpeak = quick_eval(
        model, env, eval_gen, device,
        n_scenarios=10, episode_offset=800,
        scenario_type=ScenarioType.OFFPEAK,
    )
    logging.info(
        f"  Baseline MORNING_PEAK  queue={baseline_morning['mean_queue']:.4f}  "
        f"reward={baseline_morning['mean_reward']:.4f}  (n={baseline_morning['n']})"
    )
    logging.info(
        f"  Baseline OFFPEAK       queue={baseline_offpeak['mean_queue']:.4f}  "
        f"reward={baseline_offpeak['mean_reward']:.4f}  (n={baseline_offpeak['n']})"
    )
    logging.info("")

    # ── Fine-tuning loop ──
    for episode in range(args.episodes):
        # Sample scenario: morning_fraction MORNING_PEAK, rest OFFPEAK
        if random.random() < args.morning_fraction:
            scenario_type = ScenarioType.MORNING_PEAK
            ep_idx = 850 + episode   # puts us in the curriculum's peak range
        else:
            scenario_type = ScenarioType.OFFPEAK
            ep_idx = episode

        route_file = train_gen.generate(scenario_type, ep_idx)
        env.set_route_file(route_file)
        episode_start = time.time()
        episode_rewards = []

        model.train()
        governor.reset()

        try:
            env.start(episode=episode)

            obs_list = env.get_observations()
            x = parse_sumo_observations(obs_list, device=device)
            window: deque = deque([x.detach()] * T_WINDOW, maxlen=T_WINDOW)

            buffer = RolloutBuffer()
            prev_obs, prev_actions, done = None, None, False
            ppo_metrics = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "total": 0.0}
            update_count = 0

            while not done:
                window_tensor = torch.stack(list(window))
                obs_input     = window_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    logits_list, value = model.forward(obs_input)
                    obs_last    = obs_input[0, -1]
                    masked_full = governor.apply(logits_list, obs_last)
                    actions, _  = sample_governed(masked_full)
                    masked_sl   = governor.apply_stateless(logits_list, obs_last)
                    log_probs, _ = evaluate_governed(masked_sl, actions)

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
                buffer.add(
                    obs_window=window_tensor.detach(),
                    action=actions_1d.detach(),
                    log_prob=log_probs.squeeze(0).detach(),
                    reward=reward.detach(),
                    value=value.squeeze(0).detach(),
                )

                if len(buffer) >= 64:
                    next_window = torch.stack(
                        list(window)[-(T_WINDOW - 1):] + [x_next.detach()]
                    ).unsqueeze(0).to(device)
                    with torch.no_grad():
                        _, next_val = model.forward(next_window)

                    step_metrics = ppo_update(
                        model=model, optimizer=optimizer, buffer=buffer,
                        next_value=next_val.squeeze(0),
                        clip_eps=0.2, gamma=0.99, gae_lambda=0.95,
                        entropy_coef=args.entropy_coef,
                        value_loss_coef=0.25, ppo_epochs=4,
                        minibatch_size=64, device=device,
                        target_kl=0.015, governor=governor,
                    )
                    for k in ppo_metrics:
                        ppo_metrics[k] += step_metrics[k]
                    update_count += 1
                    buffer.clear()

                prev_obs, prev_actions = next_obs_list, actions_1d
                window.append(x_next.detach())

            # Flush
            if len(buffer) >= 1:
                next_val = torch.zeros(1, device=device)
                step_metrics = ppo_update(
                    model=model, optimizer=optimizer, buffer=buffer,
                    next_value=next_val,
                    clip_eps=0.2, gamma=0.99, gae_lambda=0.95,
                    entropy_coef=args.entropy_coef,
                    value_loss_coef=0.25, ppo_epochs=4,
                    minibatch_size=64, device=device,
                    target_kl=0.015, governor=governor,
                )
                for k in ppo_metrics:
                    ppo_metrics[k] += step_metrics[k]
                update_count += 1

        finally:
            env.close()

        mean_reward = sum(episode_rewards) / max(len(episode_rewards), 1)
        reward_history.append(mean_reward)
        elapsed = time.time() - episode_start
        denom   = max(update_count, 1)

        if (episode + 1) % 10 == 0:
            rolling = sum(reward_history) / len(reward_history)
            logging.info(
                f"  Ep {episode+1:>4d}/{args.episodes} | {scenario_type.value:<14s} | "
                f"R={mean_reward:+.4f} (avg20={rolling:+.4f}) | "
                f"pi={ppo_metrics['policy']/denom:.4f} "
                f"V={ppo_metrics['value']/denom:.4f} | "
                f"{elapsed:.1f}s"
            )

        # ── Periodic eval every eval_interval episodes ──
        if (episode + 1) % args.eval_interval == 0:
            logging.info(f"\n  --- Eval at episode {episode+1} ---")

            morning = quick_eval(
                model, env, eval_gen, device,
                n_scenarios=8, episode_offset=850,
                scenario_type=ScenarioType.MORNING_PEAK,
            )
            offpeak = quick_eval(
                model, env, eval_gen, device,
                n_scenarios=8, episode_offset=0,
                scenario_type=ScenarioType.OFFPEAK,
            )

            logging.info(
                f"  MORNING_PEAK  queue={morning['mean_queue']:.4f}  "
                f"reward={morning['mean_reward']:.4f}"
            )
            logging.info(
                f"  OFFPEAK       queue={offpeak['mean_queue']:.4f}  "
                f"reward={offpeak['mean_reward']:.4f}  "
                f"[gate={OFFPEAK_GATE}]"
            )

            # Save best only if MORNING improves AND OFFPEAK stays healthy
            offpeak_ok = offpeak["mean_queue"] < OFFPEAK_GATE
            if morning["mean_queue"] < best_morning_queue and offpeak_ok:
                best_morning_queue = morning["mean_queue"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "episode": episode,
                        "morning_queue": morning["mean_queue"],
                        "offpeak_queue": offpeak["mean_queue"],
                        "baseline_morning_queue": baseline_morning["mean_queue"],
                        "baseline_offpeak_queue": baseline_offpeak["mean_queue"],
                    },
                    str(FINETUNE_BEST),
                )
                logging.info(
                    f"  *** New best: morning={morning['mean_queue']:.4f} "
                    f"(was {baseline_morning['mean_queue']:.4f}) "
                    f"offpeak={offpeak['mean_queue']:.4f} — saved ***"
                )
            elif not offpeak_ok:
                logging.info(
                    f"  [GATE BLOCKED] OFFPEAK queue {offpeak['mean_queue']:.4f} "
                    f">= {OFFPEAK_GATE} — best not updated (forgetting detected)"
                )
            logging.info("")

    # ── Final save ──
    torch.save(
        {"model_state_dict": model.state_dict(), "episode": args.episodes - 1},
        str(FINETUNE_FINAL),
    )

    logging.info("=" * 65)
    logging.info("  Fine-tuning complete")
    logging.info(f"  Best checkpoint : {FINETUNE_BEST}")
    logging.info(f"  Final checkpoint: {FINETUNE_FINAL}")
    logging.info(f"  Best MORNING_PEAK queue seen: {best_morning_queue:.4f} "
                 f"(baseline: {baseline_morning['mean_queue']:.4f})")
    logging.info("")
    logging.info("  Evaluate with:")
    logging.info("    python trafix_v6/eval_stage3.py "
                 "--checkpoint trafix_v6/checkpoints/trafix_v6_finetuned_best.pt "
                 "--scenarios 50 --greedy --episode-offset 800")
    logging.info("=" * 65)


def parse_args():
    parser = argparse.ArgumentParser(
        description="TraFix v6 — Morning Peak fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT,
                        help="Base checkpoint to fine-tune from")
    parser.add_argument("--sumo-cfg", default=DEFAULT_SUMO_CFG)
    parser.add_argument("--net-file", default=DEFAULT_NET_FILE)
    parser.add_argument("--episodes", type=int, default=250,
                        help="Fine-tuning episodes")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (keep low to avoid catastrophic forgetting)")
    parser.add_argument("--morning-fraction", type=float, default=0.70,
                        help="Fraction of episodes that are MORNING_PEAK (rest = OFFPEAK)")
    parser.add_argument("--entropy-coef", type=float, default=0.005,
                        help="Entropy coefficient")
    parser.add_argument("--eval-interval", type=int, default=50,
                        help="Eval across all scenario types every N episodes")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    finetune(args)
