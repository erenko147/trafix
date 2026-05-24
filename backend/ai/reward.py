"""
Reward computation, GAE estimation, and PPO training step.

All functions operate on per-node (N,) tensors so rewards remain
interpretable at the individual junction level.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from backend.ai.observation import LANE_KEYS, _NS_KEYS, _EW_KEYS

# ── Green-wave topology ───────────────────────────────────────────────────────

_GREEN_WAVE_EDGES: List[Tuple[int, int]] = [(0, 1), (1, 2), (1, 3), (3, 4)]
_PLATOON_THRESHOLD: int = 5


@dataclass
class RewardWeights:
    pressure:      float = -0.30
    queue:         float = -0.25
    throughput:    float =  0.25
    fairness:      float =  0.00
    phase_penalty: float = -0.08
    wait_penalty:  float = -0.05
    green_wave:    float =  0.20
    starvation:    float = -0.15


def _intersection_total(o: Dict) -> int:
    return sum(o.get(k, 0) for k in LANE_KEYS)


def _compute_green_wave(cur: List[Dict], prev: Optional[List[Dict]]) -> float:
    score = 0.0
    for (src_id, dst_id) in _GREEN_WAVE_EDGES:
        src      = cur[src_id]
        dst      = cur[dst_id]
        src_phase = src["current_phase"]
        dst_phase = dst["current_phase"]

        src_is_through = src_phase in (0, 3)
        dst_aligned    = dst_phase == src_phase

        if src_phase == 0:
            src_through_demand = src.get("north_through", 0) + src.get("south_through", 0)
        elif src_phase == 3:
            src_through_demand = src.get("east_through", 0) + src.get("west_through", 0)
        else:
            src_through_demand = 0

        has_platoon = src_through_demand >= _PLATOON_THRESHOLD

        if src_is_through and dst_aligned and has_platoon:
            base = min(src_through_demand / _PLATOON_THRESHOLD, 2.0)
            score += base
            if prev is not None:
                prev_q = prev[dst_id]["queue_length"]
                curr_q = dst["queue_length"]
                if curr_q < prev_q:
                    score += (prev_q - curr_q) / max(prev_q, 1.0)

    return score / max(len(_GREEN_WAVE_EDGES), 1)


def compute_reward(
    current_obs: List[Dict],
    previous_obs: Optional[List[Dict]],
    previous_actions: Optional[torch.Tensor],
    current_actions: torch.Tensor,
    weights: RewardWeights = RewardWeights(),
) -> torch.Tensor:
    """
    Per-intersection reward.  Returns (N,) tensor.

    Local signals : pressure, queue, throughput, fairness, phase_penalty,
                    wait_penalty, starvation.
    Global signal : green_wave bonus distributed uniformly.
    """
    cur  = sorted(current_obs,  key=lambda d: d["intersection_id"])
    prev = (
        sorted(previous_obs, key=lambda d: d["intersection_id"])
        if previous_obs is not None else None
    )

    rewards = []
    for i, o in enumerate(cur):
        pressure  = _intersection_total(o) / 60.0
        queue     = o.get("queue_length", 0.0) / 200.0

        throughput = 0.0
        if prev is not None:
            prev_total = _intersection_total(prev[i])
            cur_total  = _intersection_total(o)
            throughput = (prev_total - cur_total) / max(prev_total, 1.0)
            throughput = max(throughput, -1.0)

        lane_counts = [o.get(k, 0) for k in LANE_KEYS]
        mean_c      = sum(lane_counts) / 12.0
        var_c       = sum((c - mean_c) ** 2 for c in lane_counts) / 12.0
        fairness    = math.sqrt(var_c) / max(mean_c, 1.0)

        phase_change = 0.0
        if previous_actions is not None:
            phase_change = float(current_actions[i].item() != previous_actions[i].item())

        wait = 0.0
        if o.get("phase_duration", 0.0) > 60.0:
            wait = (o["phase_duration"] - 60.0) / 60.0

        starvation = 0.0
        phase      = int(o.get("current_phase", 0))
        dur        = o.get("phase_duration", 0.0)
        if dur > 30.0:
            ns_q      = sum(o.get(k, 0) for k in _NS_KEYS)
            ew_q      = sum(o.get(k, 0) for k in _EW_KEYS)
            total_dir = ns_q + ew_q
            if total_dir > 0:
                excess = min((dur - 30.0) / 60.0, 2.0)
                if phase in (0, 1, 2):
                    starvation = (ew_q / total_dir) * excess
                else:
                    starvation = (ns_q / total_dir) * excess

        r = (
            weights.pressure      * pressure
            + weights.queue       * queue
            + weights.throughput  * throughput
            + weights.fairness    * fairness
            + weights.phase_penalty * phase_change
            + weights.wait_penalty  * wait
            + weights.starvation    * starvation
        )
        rewards.append(r)

    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    gw            = _compute_green_wave(cur, prev)
    return reward_tensor + weights.green_wave * gw


def compute_gae(
    rewards: List[torch.Tensor],
    values: List[torch.Tensor],
    next_value: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GAE-Lambda advantage estimation.
    Returns: advantages (T, N), returns (T, N) — both normalised.
    """
    N   = rewards[0].shape[0]
    gae = torch.zeros(N, dtype=torch.float32, device=rewards[0].device)

    values_ext     = values + [next_value]
    advantages_list: List[torch.Tensor] = []

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_ext[t + 1] - values_ext[t]
        gae   = delta + gamma * lam * gae
        advantages_list.insert(0, gae.clone())

    advantages     = torch.stack(advantages_list)
    values_stacked = torch.stack([v.to(advantages.device) for v in values])
    returns        = advantages + values_stacked

    if advantages.numel() > 1 and advantages.std() > 0.01:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    if returns.numel() > 1 and returns.std() > 0.01:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return advantages, returns


def train_step(
    agent,
    optimizer: torch.optim.Optimizer,
    rollout: Dict,
    ppo_epochs: int = 4,
) -> Dict[str, float]:
    advantages, returns = compute_gae(
        rollout["rewards"], rollout["values"], rollout["next_value"],
    )

    total_metrics = {"total": 0.0, "policy": 0.0, "value": 0.0, "entropy": 0.0}
    T = len(rollout["rewards"])

    for _ in range(ppo_epochs):
        for t in range(T):
            losses = agent.compute_ppo_loss(
                x=rollout["observations"][t],
                edge_index=rollout["edge_index"],
                old_actions=rollout["actions"][t],
                old_log_probs=rollout["log_probs"][t],
                advantages=advantages[t],
                returns=returns[t],
            )
            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(agent.parameters(), agent.max_grad_norm)
            optimizer.step()
            for k in total_metrics:
                total_metrics[k] += losses[k].item()

    n = ppo_epochs * T
    return {k: v / max(n, 1) for k, v in total_metrics.items()}
