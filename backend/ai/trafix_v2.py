"""
Backward-compatibility shim for backend.ai.trafix_v2.

All public symbols are now in focused sub-modules:
  backend.ai.observation  — parse_sumo_observations, NUM_NODE_FEATURES, LANE_KEYS
  backend.ai.agent        — CoordinatedPPOAgent, SpatioTemporalGNN, IntersectionCoordinator
  backend.ai.reward       — compute_reward, compute_gae, train_step, RewardWeights

Import from those modules directly in new code.
"""

from backend.ai.observation import (   # noqa: F401
    parse_sumo_observations,
    NUM_NODE_FEATURES,
    LANE_KEYS,
)
from backend.ai.agent import (         # noqa: F401
    CoordinatedPPOAgent,
    SpatioTemporalGNN,
    IntersectionCoordinator,
)
from backend.ai.reward import (        # noqa: F401
    compute_reward,
    compute_gae,
    train_step,
    RewardWeights,
)
