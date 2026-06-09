"""
TraFix v6 — Argmax-policy fine-tune (Steps 1-3 of the argmax-fix plan)
=====================================================================
Fixes the greedy (argmax) phase-collapse by fine-tuning the production checkpoint
with:

  • Step 1 (reward): the NEW per-movement anti-starvation reward in
    backend/ai/trafix_v2.compute_reward (imported below — no extra code here).
  • Step 2 (this file):
      (a) ENTROPY ANNEALING — entropy_coef decays from --entropy-start toward
          --entropy-end over training so the policy sharpens to a well-defined
          argmax mode instead of staying diffuse/sampled.
      (b) GREEDY (ARGMAX) CHECKPOINT SELECTION — "best" is chosen by an argmax
          rollout, NOT by sampled reward. The greedy eval uses governor.apply()
          but DELIBERATELY OMITS the live-runner starvation overrides, so it
          measures exactly the raw policy that tests/diagnostics/
          argmax_phase_distribution.py measures and that the diagnostic must pass.
  • Step 3 (curriculum): a LOW-DEMAND-HEAVY scenario mix (lots of OFFPEAK) so the
    greedy policy is actually shaped where pressure/queue/throughput go flat —
    while still showing MORNING/EVENING peaks so high-traffic is not regressed.

Coordination: this is the SINGLE reconciled retrain for Steps 1-3. It does NOT
overwrite trafix_v6_final.pt — it writes new checkpoints:
    checkpoints/trafix_v6_argmax_best.pt    (best by greedy selection)
    checkpoints/trafix_v6_argmax_final.pt   (last episode)
Promotion to trafix_v6_final.pt (old → trafix_v6_final_prev.pt) is a separate,
manual step done ONLY after the diagnostic + high-traffic checks pass (Step 4).

The training governor here is IDENTICAL to the production governor
(backend/main.py::load_model) so the policy acts under the constraints it will
ship with.

Usage:
  .venv/bin/python trafix_v6/finetune_argmax.py                  # full run
  .venv/bin/python trafix_v6/finetune_argmax.py --episodes 400
  .venv/bin/python trafix_v6/finetune_argmax.py --smoke          # 2-ep code check
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
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# ── SUMO TraCI ──
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    for candidate in ["/usr/share/sumo/tools", "/usr/local/share/sumo/tools"]:
        if os.path.isdir(candidate):
            sys.path.append(candidate)
            os.environ.setdefault("SUMO_HOME", os.path.dirname(candidate))
            break

try:
    import traci  # noqa: F401
except ImportError:
    raise ImportError("SUMO TraCI not found. Set SUMO_HOME.")

# ── Path setup ──
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

from trafix_v6.trafix_v6 import TraFixV6, NUM_JUNCTIONS, NUM_PHASES as _NP
from trafix_v6.rule_governor import RuleGovernor, sample_governed, evaluate_governed
from trafix_v6.scenario_generator import (
    ScenarioGenerator, ScenarioEnvironment, ScenarioType,
)

try:
    from backend.ai.train_v2 import TrainConfig
    from backend.ai.trafix_v2 import (
        parse_sumo_observations, compute_reward, compute_gae, NUM_NODE_FEATURES,
    )
except ImportError:
    sys.path.insert(0, str(_PROJECT_ROOT / "backend" / "ai"))
    from train_v2 import TrainConfig
    from trafix_v2 import (
        parse_sumo_observations, compute_reward, compute_gae, NUM_NODE_FEATURES,
    )

# Reuse the Stage-3 PPO update verbatim (governor-consistent, KL early-stop).
from trafix_v6.stage3_train_ppo import ppo_update, RolloutBuffer

OBS_DIM = NUM_NODE_FEATURES   # 20
NUM_PHASES = _NP              # 6
T_WINDOW = 30

CHECKPOINTS_DIR = _SCRIPT_DIR / "checkpoints"
DEFAULT_CKPT   = str(CHECKPOINTS_DIR / "trafix_v6_final.pt")
ARGMAX_BEST    = CHECKPOINTS_DIR / "trafix_v6_argmax_best.pt"
ARGMAX_FINAL   = CHECKPOINTS_DIR / "trafix_v6_argmax_final.pt"
DEFAULT_SUMO_CFG = str(_PROJECT_ROOT / "sumo" / "training.sumocfg")
DEFAULT_NET_FILE = str(_PROJECT_ROOT / "sumo" / "map.net.xml")

# Step 3 — low-demand-heavy training curriculum. OFFPEAK dominates so the greedy
# policy is shaped at low flow (the collapse regime), but peaks/pulses stay in the
# mix so high-traffic behaviour is not forgotten.
_TRAIN_MIX: List[Tuple[ScenarioType, float]] = [
    (ScenarioType.OFFPEAK,      0.45),
    (ScenarioType.MORNING_PEAK, 0.18),
    (ScenarioType.EVENING_PEAK, 0.15),
    (ScenarioType.PULSE,        0.12),
    (ScenarioType.INCIDENT,     0.10),
]

# Fixed greedy-eval set (reproducible). Mix low + peak so selection rewards a
# policy that is good at low demand AND does not regress high traffic.
_EVAL_SET: List[Tuple[ScenarioType, int]] = [
    (ScenarioType.OFFPEAK, 10), (ScenarioType.OFFPEAK, 11),
    (ScenarioType.OFFPEAK, 12),
    (ScenarioType.PULSE, 20),
    (ScenarioType.MORNING_PEAK, 30), (ScenarioType.MORNING_PEAK, 31),
    (ScenarioType.EVENING_PEAK, 40),
]


def _sample_scenario_type() -> ScenarioType:
    r = random.random()
    cum = 0.0
    for st, w in _TRAIN_MIX:
        cum += w
        if r <= cum:
            return st
    return _TRAIN_MIX[-1][0]


def _entropy_coef(episode: int, total: int, start: float, end: float) -> float:
    """Cosine anneal entropy_coef start → end over the run (never zero early)."""
    if total <= 1:
        return end
    progress = min(episode / (total - 1), 1.0)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def greedy_eval(model, env, generator, governor, device,
                eval_set=_EVAL_SET) -> Dict[str, float]:
    """
    GREEDY (argmax) rollout used for checkpoint selection.

    Convention (documented + must match argmax_phase_distribution.py):
      • phases = ARGMAX of governor.apply(logits, obs_last)   ← governor ON
      • NO starvation overrides                                ← raw policy
      • standard yellow transitions (handled by ScenarioEnvironment.step)

    Returns aggregate greedy metrics:
      mean_reward  — under the NEW anti-starvation reward (higher = better)
      mean_queue   — obs[:,12] mean (lower = better)
      worst_lock   — max over junctions of (max single-phase share); >0.70 ⇒ a
                     junction is collapsing. Lower = better. (Diagnostic-style.)
      peak_queue   — mean_queue restricted to MORNING/EVENING peak episodes, used
                     as a high-traffic no-regression guard.
    """
    model.eval()
    rewards, queues = [], []
    peak_queues = []
    # per-junction phase counts aggregated across all eval episodes
    phase_counts = [[0] * NUM_PHASES for _ in range(NUM_JUNCTIONS)]

    for scen_type, ep_idx in eval_set:
        route_file = generator.generate(scen_type, ep_idx)
        env.set_route_file(route_file)
        try:
            env.start(episode=ep_idx)
        except Exception:
            env.close()
            continue

        obs_list = env.get_observations()
        x = parse_sumo_observations(obs_list, device=device)
        window = deque([x.detach()] * T_WINDOW, maxlen=T_WINDOW)
        governor.reset()

        prev_obs, prev_actions, done = None, None, False
        ep_q = []
        while not done:
            window_tensor = torch.stack(list(window)).unsqueeze(0).to(device)
            logits_list, _ = model.forward(window_tensor)
            obs_last = window_tensor[0, -1]
            masked = governor.apply(logits_list, obs_last)
            actions_1d = torch.stack(
                [torch.argmax(l, dim=-1).reshape(()) for l in masked]
            )
            governor.update_state(actions_1d)

            for j in range(NUM_JUNCTIONS):
                phase_counts[j][int(actions_1d[j].item()) % NUM_PHASES] += 1

            next_obs_list, done = env.step(actions_1d)
            x_next = parse_sumo_observations(next_obs_list, device=device)

            reward = compute_reward(
                current_obs=next_obs_list, previous_obs=prev_obs,
                previous_actions=prev_actions, current_actions=actions_1d,
            )
            rewards.append(reward.mean().item())
            ep_q.append(float(x_next[:, 12].mean()))

            prev_obs, prev_actions = next_obs_list, actions_1d
            window.append(x_next.detach())

        env.close()
        if ep_q:
            q = sum(ep_q) / len(ep_q)
            queues.append(q)
            if scen_type in (ScenarioType.MORNING_PEAK, ScenarioType.EVENING_PEAK):
                peak_queues.append(q)

    model.train()

    worst_lock = 0.0
    for j in range(NUM_JUNCTIONS):
        tot = sum(phase_counts[j])
        if tot > 0:
            worst_lock = max(worst_lock, max(phase_counts[j]) / tot)

    return {
        "mean_reward": sum(rewards) / max(len(rewards), 1),
        "mean_queue": sum(queues) / max(len(queues), 1),
        "worst_lock": worst_lock,
        "peak_queue": sum(peak_queues) / max(len(peak_queues), 1),
    }


def finetune(args):
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    logs_dir = _SCRIPT_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(str(logs_dir / "finetune_argmax.log"), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("=" * 68)
    logging.info("  TraFix v6 — Argmax-policy fine-tune (Steps 1-3)")
    logging.info(f"  base ckpt   : {args.checkpoint}")
    logging.info(f"  episodes    : {args.episodes}")
    logging.info(f"  entropy     : {args.entropy_start} -> {args.entropy_end} (cosine anneal)")
    logging.info(f"  selection   : GREEDY argmax rollout (governor ON, overrides OFF)")
    logging.info(f"  curriculum  : low-demand-heavy {[ (s.value,w) for s,w in _TRAIN_MIX ]}")
    logging.info("=" * 68)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")
    model = TraFixV6(obs_dim=OBS_DIM, num_phases=NUM_PHASES).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    logging.info(f"  loaded {ckpt_path} (episode={ckpt.get('episode','?')})")

    # Freeze pretrained encoders (protect GRU/GAT prior) — train trunk + heads.
    for p in model.temporal_enc.parameters():
        p.requires_grad_(False)
    for p in model.graph_enc.parameters():
        p.requires_grad_(False)
    trainable = (list(model.trunk.parameters())
                 + list(model.actor_heads.parameters())
                 + list(model.local_critics.parameters())
                 + list(model.global_critic.parameters()))
    optimizer = optim.Adam(trainable, lr=args.lr)
    logging.info(f"  trainable params: {sum(p.numel() for p in trainable):,} "
                 f"/ {sum(p.numel() for p in model.parameters()):,} (encoders frozen)")

    env_cfg = TrainConfig()
    env_cfg.sumo_cfg = args.sumo_cfg
    env_cfg.gui = False
    env_cfg.seed = args.seed
    env_cfg.decision_interval = 10
    env_cfg.warmup_steps = 50
    env_cfg.max_steps_per_episode = 3600
    env_cfg.num_actions = NUM_PHASES
    env_cfg.rollout_length = 64
    env = ScenarioEnvironment(env_cfg)

    train_gen = ScenarioGenerator(net_file=args.net_file,
                                  output_dir=str(_SCRIPT_DIR / "scenarios"), seed=None)
    eval_gen = ScenarioGenerator(net_file=args.net_file,
                                 output_dir=str(_SCRIPT_DIR / "scenarios_eval"), seed=77)

    # Production governor (identical to backend/main.py::load_model).
    governor = RuleGovernor(
        num_junctions=NUM_JUNCTIONS, num_phases=NUM_PHASES,
        min_green_s=10.0, max_green_s=90.0,
        flicker_window=2, flicker_penalty=3.0,
        pressure_boost=1.0, pressure_thresh=args.pressure_thresh,
    )

    # Baseline greedy eval before fine-tuning.
    base = greedy_eval(model, env, eval_gen, governor, device)
    logging.info(f"  BASELINE greedy: reward={base['mean_reward']:+.4f} "
                 f"queue={base['mean_queue']:.4f} worst_lock={base['worst_lock']:.2f} "
                 f"peak_queue={base['peak_queue']:.4f}")
    peak_ceiling = base["peak_queue"] * (1.0 + args.peak_slack)
    logging.info(f"  high-traffic guard: peak_queue must stay < {peak_ceiling:.4f} "
                 f"(baseline x {1.0+args.peak_slack:.2f})")

    best_reward = -math.inf
    reward_history = deque(maxlen=20)

    for episode in range(args.episodes):
        ent_coef = _entropy_coef(episode, args.episodes,
                                 args.entropy_start, args.entropy_end)
        scen_type = _sample_scenario_type()
        route_file = train_gen.generate(scen_type, 850 + episode)
        env.set_route_file(route_file)
        t0 = time.time()

        model.train()
        governor.reset()
        ep_rewards = []
        try:
            env.start(episode=episode)
            obs_list = env.get_observations()
            x = parse_sumo_observations(obs_list, device=device)
            window = deque([x.detach()] * T_WINDOW, maxlen=T_WINDOW)
            buffer = RolloutBuffer()
            prev_obs, prev_actions, done = None, None, False

            while not done:
                window_tensor = torch.stack(list(window))
                obs_input = window_tensor.unsqueeze(0).to(device)
                with torch.no_grad():
                    logits_list, value = model.forward(obs_input)
                    masked_full = governor.apply(logits_list, obs_input[0, -1])
                    actions, log_probs = sample_governed(masked_full)
                actions_1d = actions.squeeze(0)
                governor.update_state(actions_1d)

                next_obs_list, done = env.step(actions_1d)
                x_next = parse_sumo_observations(next_obs_list, device=device)
                reward = compute_reward(
                    current_obs=next_obs_list, previous_obs=prev_obs,
                    previous_actions=prev_actions, current_actions=actions_1d,
                ).to(device)
                ep_rewards.append(reward.mean().item())

                buffer.add(obs_window=window_tensor.detach(),
                           action=actions_1d.detach(),
                           log_prob=log_probs.squeeze(0).detach(),
                           reward=reward.detach(),
                           value=value.squeeze(0).detach())

                if len(buffer) >= 64:
                    next_window = torch.stack(
                        list(window)[-(T_WINDOW - 1):] + [x_next.detach()]
                    ).unsqueeze(0).to(device)
                    with torch.no_grad():
                        _, next_val = model.forward(next_window)
                    ppo_update(model=model, optimizer=optimizer, buffer=buffer,
                               next_value=next_val.squeeze(0),
                               clip_eps=0.2, gamma=0.99, gae_lambda=0.95,
                               entropy_coef=ent_coef, value_loss_coef=0.25,
                               ppo_epochs=4, minibatch_size=64, device=device,
                               target_kl=0.015, governor=governor)
                    buffer.clear()

                prev_obs, prev_actions = next_obs_list, actions_1d
                window.append(x_next.detach())

            if len(buffer) >= 1:
                last_window = torch.stack(list(window)).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, next_val = model.forward(last_window)
                ppo_update(model=model, optimizer=optimizer, buffer=buffer,
                           next_value=next_val.squeeze(0),
                           clip_eps=0.2, gamma=0.99, gae_lambda=0.95,
                           entropy_coef=ent_coef, value_loss_coef=0.25,
                           ppo_epochs=4, minibatch_size=64, device=device,
                           target_kl=0.015, governor=governor)
        finally:
            env.close()

        reward_history.append(sum(ep_rewards) / max(len(ep_rewards), 1))
        if (episode + 1) % 10 == 0:
            logging.info(f"  Ep {episode+1:>4d}/{args.episodes} | {scen_type.value:<13s} "
                         f"| R={reward_history[-1]:+.4f} (avg20={sum(reward_history)/len(reward_history):+.4f}) "
                         f"| ent={ent_coef:.4f} | {time.time()-t0:.1f}s")

        if (episode + 1) % args.eval_interval == 0:
            g = greedy_eval(model, env, eval_gen, governor, device)
            peak_ok = g["peak_queue"] <= peak_ceiling
            logging.info(f"  --- greedy eval @ep{episode+1}: reward={g['mean_reward']:+.4f} "
                         f"queue={g['mean_queue']:.4f} worst_lock={g['worst_lock']:.2f} "
                         f"peak_queue={g['peak_queue']:.4f} {'OK' if peak_ok else 'PEAK-REGRESSED'} ---")
            if g["mean_reward"] > best_reward and peak_ok:
                best_reward = g["mean_reward"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "episode": episode,
                    "greedy_reward": g["mean_reward"],
                    "greedy_queue": g["mean_queue"],
                    "greedy_worst_lock": g["worst_lock"],
                    "greedy_peak_queue": g["peak_queue"],
                    "selection": "greedy_argmax_governor_no_overrides",
                }, str(ARGMAX_BEST))
                logging.info(f"  *** new best greedy reward={best_reward:+.4f} "
                             f"(worst_lock={g['worst_lock']:.2f}) — saved {ARGMAX_BEST.name} ***")
            elif not peak_ok:
                logging.info("  [GATE] peak_queue regressed — best not updated")

    torch.save({"model_state_dict": model.state_dict(),
                "episode": args.episodes - 1}, str(ARGMAX_FINAL))
    logging.info("=" * 68)
    logging.info(f"  done. best greedy reward={best_reward:+.4f}")
    logging.info(f"  best : {ARGMAX_BEST}")
    logging.info(f"  final: {ARGMAX_FINAL}")
    logging.info("  Validate (Step 4) before promoting:")
    logging.info(f"    .venv/bin/python tests/diagnostics/argmax_phase_distribution.py "
                 f"--checkpoint {ARGMAX_BEST} --route tests/scenarios/type1_low.rou.xml")
    logging.info(f"    .venv/bin/python tests/diagnostics/argmax_phase_distribution.py "
                 f"--checkpoint {ARGMAX_BEST} --route tests/scenarios/type1_medium.rou.xml")
    logging.info(f"    .venv/bin/python tests/diagnostics/high_traffic_reference.py "
                 f"--checkpoint {ARGMAX_BEST} --tag after")
    logging.info("=" * 68)


def parse_args():
    p = argparse.ArgumentParser(
        description="TraFix v6 — argmax-policy fine-tune (Steps 1-3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--sumo-cfg", default=DEFAULT_SUMO_CFG)
    p.add_argument("--net-file", default=DEFAULT_NET_FILE)
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-5,
                   help="LR for trunk+heads (low — encoders frozen)")
    p.add_argument("--entropy-start", type=float, default=0.01,
                   help="entropy_coef at start (keeps exploration alive)")
    p.add_argument("--entropy-end", type=float, default=0.0005,
                   help="entropy_coef target (~0 ⇒ sharp argmax mode)")
    p.add_argument("--eval-interval", type=int, default=25)
    p.add_argument("--pressure-thresh", type=float, default=0.12,
                   help="governor pressure_thresh (Step 5; must match production governor)")
    p.add_argument("--peak-slack", type=float, default=0.10,
                   help="allowed high-traffic queue regression before gating best")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="2-episode, eval-every-1 code check (no real training)")
    args = p.parse_args()
    if args.smoke:
        args.episodes = 2
        args.eval_interval = 1
    return args


if __name__ == "__main__":
    finetune(parse_args())
