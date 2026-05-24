"""
Rollout buffer — accumulates experience tuples during environment interaction.
"""

from typing import Dict, List

import torch


class RolloutBuffer:
    """Stores one rollout's worth of (obs, action, log_prob, reward, value) tuples."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.observations: List[torch.Tensor] = []
        self.actions:      List[torch.Tensor] = []
        self.log_probs:    List[torch.Tensor] = []
        self.rewards:      List[torch.Tensor] = []
        self.values:       List[torch.Tensor] = []

    def add(
        self,
        obs:      torch.Tensor,
        action:   torch.Tensor,
        log_prob: torch.Tensor,
        reward:   torch.Tensor,
        value:    torch.Tensor,
    ):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value.detach())

    def __len__(self) -> int:
        return len(self.rewards)

    def to_dict(self, edge_index: torch.Tensor, next_value: torch.Tensor) -> Dict:
        return {
            "observations": self.observations,
            "edge_index":   edge_index,
            "actions":      self.actions,
            "log_probs":    self.log_probs,
            "rewards":      self.rewards,
            "values":       self.values,
            "next_value":   next_value,
        }
