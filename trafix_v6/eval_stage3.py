"""
TraFix v6 — Stage 3 Checkpoint Evaluator
==========================================
Loads a stage3 checkpoint and evaluates across SUMO scenarios.

v6 changes vs v5:
  NUM_PHASES = 6
  OBS_DIM = 20
  queue metric: obs[:, 12] (index 12 = total queue / 200)
  wait metric:  obs[:, 19] (index 19 = phase duration / 120)

Usage:
  python trafix_v6/eval_stage3.py
  python trafix_v6/eval_stage3.py --checkpoint checkpoints/stage3_ep600.pt --scenarios 10
  python trafix_v6/eval_stage3.py --checkpoint checkpoints/trafix_v6_final.pt --greedy --gui
"""

import os
import sys
import math
import argparse
import logging
from pathlib import Path
from collections import deque
from typing import List, Dict

import torch

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
    from backend.ai.trafix_v2 import parse_sumo_observations, compute_reward, NUM_NODE_FEATURES
except ImportError:
    try:
        from train_v2 import TrainConfig
        from trafix_v2 import parse_sumo_observations, compute_reward, NUM_NODE_FEATURES
    except ImportError:
        sys.path.insert(0, str(_PROJECT_ROOT / "backend" / "ai"))
        from train_v2 import TrainConfig
        from trafix_v2 import parse_sumo_observations, compute_reward, NUM_NODE_FEATURES


OBS_DIM    = NUM_NODE_FEATURES   # 20
NUM_PHASES = 6
T_WINDOW   = 10

CHECKPOINTS_DIR    = _SCRIPT_DIR / "checkpoints"
DEFAULT_CHECKPOINT = str(CHECKPOINTS_DIR / "trafix_v6_final.pt")
DEFAULT_SUMO_CFG   = str(_PROJECT_ROOT / "sumo" / "training.sumocfg")
DEFAULT_NET_FILE   = str(_PROJECT_ROOT / "sumo" / "map.net.xml")


def run_episode(model, env, device, episode_idx, greedy) -> Dict[str, float]:
    env.start(episode=episode_idx)

    obs_list = env.get_observations()
    x = parse_sumo_observations(obs_list, device=device)
    window: deque = deque([x.detach()] * T_WINDOW, maxlen=T_WINDOW)

    rewards, queues, waits = [], [], []
    steps = 0
    prev_obs = None
    prev_actions = None
    done = False

    model.eval()
    with torch.no_grad():
        while not done:
            window_tensor = torch.stack(list(window)).unsqueeze(0).to(device)

            if greedy:
                logits_list, _ = model.forward(window_tensor)
                actions_1d = torch.stack(
                    [l.argmax(dim=-1) for l in logits_list], dim=1
                ).squeeze(0)
            else:
                actions, _lp, _ = model.get_action(window_tensor)
                actions_1d = actions.squeeze(0)

            next_obs_list, done = env.step(actions_1d)
            x_next = parse_sumo_observations(next_obs_list, device=device)

            reward = compute_reward(
                current_obs=next_obs_list,
                previous_obs=prev_obs,
                previous_actions=prev_actions,
                current_actions=actions_1d,
            )

            rewards.append(reward.mean().item())
            queues.append(float(x_next[:, 12].mean()))   # queue index
            waits.append(float(x_next[:, 19].mean()))    # duration index

            prev_obs = next_obs_list
            prev_actions = actions_1d
            window.append(x_next.detach())
            steps += 1

    env.close()

    return {
        "mean_reward": sum(rewards) / max(len(rewards), 1),
        "mean_queue":  sum(queues)  / max(len(queues),  1),
        "mean_wait":   sum(waits)   / max(len(waits),   1),
        "episode_steps": steps,
    }


def evaluate(args):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"  Device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model = TraFixV6(obs_dim=OBS_DIM, num_phases=NUM_PHASES).to(device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    loaded_ep = checkpoint.get("episode", "?")
    logging.info(f"  Checkpoint loaded: {ckpt_path} (episode={loaded_ep})")
    logging.info(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    logging.info(f"  Mode: {'greedy' if args.greedy else 'stochastic'}")

    env_cfg = TrainConfig()
    env_cfg.sumo_cfg = args.sumo_cfg
    env_cfg.gui = args.gui
    env_cfg.seed = args.seed
    env_cfg.decision_interval = 10
    env_cfg.warmup_steps = 50
    env_cfg.max_steps_per_episode = args.max_steps
    env_cfg.num_actions = NUM_PHASES
    env = ScenarioEnvironment(env_cfg)

    generator = ScenarioGenerator(
        net_file=args.net_file,
        output_dir=str(_SCRIPT_DIR / "scenarios_eval"),
        seed=args.seed,
    )

    logging.info("=" * 60)
    logging.info(f"  Running {args.scenarios} evaluation scenarios")
    logging.info("=" * 60)

    all_metrics: List[Dict[str, float]] = []
    scenario_types: List[str] = []

    for i in range(args.scenarios):
        scenario_type, route_file = generator.sample(i + args.episode_offset)
        env.set_route_file(route_file)
        scenario_types.append(scenario_type.value)

        try:
            metrics = run_episode(model, env, device, episode_idx=i, greedy=args.greedy)
        except Exception as e:
            logging.warning(f"  Scenario {i} ({scenario_type.value}) failed: {e}")
            env.close()
            continue

        all_metrics.append(metrics)
        logging.info(
            f"  [{i+1:>3d}/{args.scenarios}] {scenario_type.value:<18s} | "
            f"R={metrics['mean_reward']:+.4f}  "
            f"Q={metrics['mean_queue']:.3f}  "
            f"W={metrics['mean_wait']:.3f}  "
            f"steps={metrics['episode_steps']}"
        )

    if not all_metrics:
        logging.error("No scenarios completed.")
        return

    def _mean(key):
        return sum(m[key] for m in all_metrics) / len(all_metrics)

    logging.info("=" * 60)
    logging.info("  AGGREGATE RESULTS")
    logging.info(f"  Scenarios evaluated : {len(all_metrics)}/{args.scenarios}")
    logging.info(f"  mean_reward         : {_mean('mean_reward'):+.4f}")
    logging.info(f"  mean_queue          : {_mean('mean_queue'):.4f}")
    logging.info(f"  mean_wait           : {_mean('mean_wait'):.4f}")
    logging.info(f"  mean_episode_steps  : {_mean('episode_steps'):.1f}")
    logging.info("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="TraFix v6 — checkpoint evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sumo-cfg", default=DEFAULT_SUMO_CFG)
    parser.add_argument("--net-file", default=DEFAULT_NET_FILE)
    parser.add_argument("--scenarios", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-offset", type=int, default=800,
                        help="Curriculum episode offset (>=800 gives full mixed scenarios)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
