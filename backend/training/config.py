"""
Training hyperparameter configuration.

All tuneable knobs live in TrainConfig so they can be overridden from the
CLI (scripts/train.py) without touching training logic.
"""

from dataclasses import dataclass, field

from backend.ai.reward import RewardWeights


@dataclass
class TrainConfig:
    # ── File paths ──────────────────────────────────────────────────────────
    sumo_cfg:         str = "sumo/training.sumocfg"
    net_file:         str = "sumo/map.net.xml"
    output_dir:       str = "training_outputs"
    checkpoint_path:  str = "coordinated_agent_weights.pth"

    # ── Training ────────────────────────────────────────────────────────────
    episodes:              int   = 500
    max_steps_per_episode: int   = 3600
    decision_interval:     int   = 10
    warmup_steps:          int   = 50
    ppo_epochs:            int   = 4
    rollout_length:        int   = 64

    # ── Model ───────────────────────────────────────────────────────────────
    hidden_dim:  int = 128
    num_actions: int = 4
    num_heads:   int = 4

    # ── Optimisation ────────────────────────────────────────────────────────
    lr:            float = 3e-4
    lr_min:        float = 1e-5
    eps:           float = 1e-5
    max_grad_norm: float = 0.5
    gamma:         float = 0.99
    gae_lambda:    float = 0.95

    # ── PPO ─────────────────────────────────────────────────────────────────
    clip_eps:          float = 0.2
    entropy_coef:      float = 0.005
    entropy_coef_min:  float = 0.001
    entropy_decay:     float = 0.9998
    value_coef:        float = 0.25

    # ── Reward ──────────────────────────────────────────────────────────────
    reward_weights: RewardWeights = field(default_factory=RewardWeights)

    # ── Checkpointing / logging ─────────────────────────────────────────────
    save_interval: int = 25
    log_interval:  int = 5
    eval_interval: int = 50

    # ── SUMO ────────────────────────────────────────────────────────────────
    gui:              bool  = False
    sumo_step_length: float = 1.0
    seed:             int   = 42

    # ── Resume ──────────────────────────────────────────────────────────────
    resume: bool = False
