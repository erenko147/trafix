"""
Learning-rate scheduler: linear warmup followed by cosine decay.
"""

import math

import torch.optim as optim


class CosineWarmupScheduler:
    """Warmup + cosine-annealing LR schedule."""

    def __init__(
        self,
        optimizer:        optim.Optimizer,
        warmup_episodes:  int,
        total_episodes:   int,
        lr_min:           float,
    ):
        self.optimizer       = optimizer
        self.warmup_episodes = warmup_episodes
        self.total_episodes  = total_episodes
        self.lr_min          = lr_min
        self.base_lr         = optimizer.param_groups[0]["lr"]

    def step(self, episode: int) -> float:
        if episode < self.warmup_episodes:
            lr = self.base_lr * (episode + 1) / self.warmup_episodes
        else:
            progress = (episode - self.warmup_episodes) / max(
                1, self.total_episodes - self.warmup_episodes
            )
            lr = self.lr_min + 0.5 * (self.base_lr - self.lr_min) * (
                1 + math.cos(math.pi * progress)
            )

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr
