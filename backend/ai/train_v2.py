"""
Backward-compatibility shim for backend.ai.train_v2.

Training components are now split across focused modules:
  backend.training.config       — TrainConfig
  backend.training.environment  — SumoEnvironment, build_edge_index
  backend.training.buffer       — RolloutBuffer
  backend.training.scheduler    — CosineWarmupScheduler
  backend.training.logger       — TrainingLogger
  backend.training.trainer      — train(), _save_checkpoint

Entry point: scripts/train.py
"""

from backend.training.config import TrainConfig          # noqa: F401
from backend.training.environment import (               # noqa: F401
    SumoEnvironment,
    build_edge_index,
)
from backend.training.buffer import RolloutBuffer        # noqa: F401
from backend.training.scheduler import CosineWarmupScheduler  # noqa: F401
from backend.training.logger import TrainingLogger       # noqa: F401
from backend.training.trainer import train, _save_checkpoint  # noqa: F401
