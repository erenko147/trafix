"""
TraFix training CLI.

Usage:
  python scripts/train.py                        # 500 episodes, defaults
  python scripts/train.py --episodes 1000
  python scripts/train.py --gui
  python scripts/train.py --resume --checkpoint training_outputs/best_model.pth
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.training.config import TrainConfig
from backend.training.trainer import train


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="TraFix PPO Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--sumo-cfg",          default="sumo/training.sumocfg")
    parser.add_argument("--net-file",          default="sumo/map.net.xml")
    parser.add_argument("--output-dir",        default="training_outputs")
    parser.add_argument("--checkpoint",        default="coordinated_agent_weights.pth")
    parser.add_argument("--episodes",          type=int,   default=500)
    parser.add_argument("--max-steps",         type=int,   default=3600)
    parser.add_argument("--decision-interval", type=int,   default=10)
    parser.add_argument("--rollout-length",    type=int,   default=64)
    parser.add_argument("--ppo-epochs",        type=int,   default=4)
    parser.add_argument("--hidden-dim",        type=int,   default=128)
    parser.add_argument("--num-actions",       type=int,   default=4)
    parser.add_argument("--num-heads",         type=int,   default=4)
    parser.add_argument("--lr",                type=float, default=3e-4)
    parser.add_argument("--gamma",             type=float, default=0.99)
    parser.add_argument("--clip-eps",          type=float, default=0.2)
    parser.add_argument("--entropy-coef",      type=float, default=0.005)
    parser.add_argument("--value-coef",        type=float, default=0.25)
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--log-interval",      type=int,   default=5)
    parser.add_argument("--gui",               action="store_true")
    parser.add_argument("--resume",            action="store_true")

    args = parser.parse_args()

    return TrainConfig(
        sumo_cfg=args.sumo_cfg,
        net_file=args.net_file,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        decision_interval=args.decision_interval,
        rollout_length=args.rollout_length,
        ppo_epochs=args.ppo_epochs,
        hidden_dim=args.hidden_dim,
        num_actions=args.num_actions,
        num_heads=args.num_heads,
        lr=args.lr,
        gamma=args.gamma,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        gui=args.gui,
        seed=args.seed,
        resume=args.resume,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
