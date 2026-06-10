"""
TraFix — shared observation, reward and advantage code
======================================================
Single source of truth (used by ALL model versions and every v6 training script)
for the 20-dim SUMO observation parser, the per-junction reward function, and GAE.

  • NUM_NODE_FEATURES = 20 (12 per-lane shares + queue + 6-phase one-hot + duration)
  • parse_sumo_observations: 12 lane fields as junction-relative shares
      (count / total_12_lane_sum) — scale-invariant, fixes low-demand signal (P5)
  • compute_reward: pressure/queue/throughput/fairness/anti-starvation over 12 lanes
  • _compute_green_wave: through phases are 0 (NS) and 3 (EW) in model space

The legacy v2 model (GCN + Multi-Head-Attention `CoordinatedPPOAgent`) and its
`train_step` lived here but had no trained weights and were removed; only the
production v6 model (`trafix_v6/trafix_v6.py`) runs. This module is kept because
v6 reuses the observation/reward/GAE code below.
"""

import math
import torch
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


# ══════════════════════════════════════════════════
#  SUMO Observation Parsing
# ══════════════════════════════════════════════════

# 12 per-lane keys in order (indices 0-11)
_LANE_KEYS = [
    "north_left", "north_through", "north_right",
    "south_left", "south_through", "south_right",
    "east_left",  "east_through",  "east_right",
    "west_left",  "west_through",  "west_right",
]

_NS_KEYS = ["north_left", "north_through", "north_right",
            "south_left", "south_through", "south_right"]
_EW_KEYS = ["east_left",  "east_through",  "east_right",
            "west_left",  "west_through",  "west_right"]

# Output feature count:
#   [0-11]  12 lane relative shares  (each lane_count / total_12_lane_sum)
#   [12]    total queue / 200
#   [13-18] 6-bit phase one-hot
#   [19]    phase duration / 120
NUM_NODE_FEATURES = 20


def parse_sumo_observations(
    obs_list: List[Dict],
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Convert list of junction observation dicts to [J, 20] normalised tensor.

    Each dict must contain:
      intersection_id, north_left, north_through, north_right,
      south_left, south_through, south_right,
      east_left, east_through, east_right,
      west_left, west_through, west_right,
      queue_length, current_phase (model space 0-5), phase_duration
    """
    rows = []
    for o in sorted(obs_list, key=lambda x: x["intersection_id"]):
        row = []

        # Indices 0-11: per-lane share of total junction demand.
        # Dividing by the sum of all 12 lanes makes the features scale-invariant:
        # 1 car out of 5 and 4 cars out of 20 both read as 0.20, giving the model
        # a meaningful gradient at low demand where absolute counts are near zero.
        total_lane = sum(o.get(key, 0) for key in _LANE_KEYS)
        denom = max(total_lane, 1)
        for key in _LANE_KEYS:
            row.append(o.get(key, 0) / denom)

        # Index 12: total queue / 200
        row.append(o.get("queue_length", 0.0) / 200.0)

        # Indices 13-18: 6-bit phase one-hot
        one_hot = [0.0] * 6
        phase = int(o.get("current_phase", 0)) % 6
        one_hot[phase] = 1.0
        row.extend(one_hot)

        # Index 19: phase duration / 120, capped at 3.0 (= 6 min) so long holds
        # remain distinguishable and don't all collapse to 1.0
        row.append(min(o.get("phase_duration", 0.0) / 120.0, 3.0))

        rows.append(row)

    return torch.tensor(rows, dtype=torch.float32, device=device)


# ══════════════════════════════════════════════════
#  Reward Function — Per-Node (N,) Tensor
# ══════════════════════════════════════════════════

_GREEN_WAVE_EDGES: List[Tuple[int, int]] = [(0, 1), (1, 2), (1, 3), (3, 4)]
_PLATOON_THRESHOLD: int = 5


@dataclass
class RewardWeights:
    pressure:      float = -0.30
    queue:         float = -0.25
    throughput:    float =  0.25
    fairness:      float =  0.00     # CV-of-lanes term; kept disabled (noisy) —
                                     # anti-starvation is handled by `starvation`
    phase_penalty: float = -0.08
    wait_penalty:  float = -0.05
    green_wave:    float =  0.20
    starvation:    float = -0.20     # per-movement anti-starvation (see below)
    clear_bonus:   float =  0.06     # low-demand shaping: reward clearing any car


def _intersection_total(o: Dict) -> int:
    """Sum of all 12 per-lane vehicle counts."""
    return sum(o.get(k, 0) for k in _LANE_KEYS)


def _compute_green_wave(cur: List[Dict], prev: Optional[List[Dict]]) -> float:
    """
    Green wave bonus: reward through-phase alignment between adjacent junctions.
    Through phases in model space: 0 = NS-through, 3 = EW-through.
    """
    score = 0.0
    for (src_id, dst_id) in _GREEN_WAVE_EDGES:
        src = cur[src_id]
        dst = cur[dst_id]

        src_phase = src["current_phase"]  # model space 0-5
        dst_phase = dst["current_phase"]

        # Only through phases (0=NS, 3=EW) create meaningful green waves
        src_is_through = src_phase in (0, 3)
        dst_aligned = dst_phase == src_phase

        # Through demand: NS-through uses north/south through counts; EW uses east/west
        if src_phase == 0:
            src_through_demand = (
                src.get("north_through", 0) + src.get("south_through", 0)
            )
        elif src_phase == 3:
            src_through_demand = (
                src.get("east_through", 0) + src.get("west_through", 0)
            )
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
    Per-intersection reward. Returns (N,) tensor.

    Local signals: pressure, queue, throughput, fairness, phase_penalty,
                   wait_penalty, starvation.
    Global signal: green_wave bonus distributed uniformly.
    """
    cur  = sorted(current_obs,  key=lambda d: d["intersection_id"])
    prev = sorted(previous_obs, key=lambda d: d["intersection_id"]) \
           if previous_obs is not None else None

    rewards = []
    for i, o in enumerate(cur):
        # 1. Pressure: sum of 12 lanes / 60.0
        pressure = _intersection_total(o) / 60.0

        # 2. Queue: total queue / 200.0
        queue = o.get("queue_length", 0.0) / 200.0

        # 3. Throughput: vehicle reduction at this intersection
        throughput = 0.0
        if prev is not None:
            prev_total = _intersection_total(prev[i])
            cur_total  = _intersection_total(o)
            throughput = (prev_total - cur_total) / max(prev_total, 1.0)
            throughput = max(throughput, -1.0)

        # 4. Fairness: std of 12 per-lane counts / max(mean, 1.0)
        lane_counts = [o.get(k, 0) for k in _LANE_KEYS]
        mean_c = sum(lane_counts) / 12.0
        var_c  = sum((c - mean_c) ** 2 for c in lane_counts) / 12.0
        fairness = math.sqrt(var_c) / max(mean_c, 1.0)

        # 5. Phase stability
        phase_change = 0.0
        if previous_actions is not None:
            phase_change = float(
                current_actions[i].item() != previous_actions[i].item()
            )

        # 6. Wait penalty: fires when phase_duration > 60s
        wait = 0.0
        if o.get("phase_duration", 0.0) > 60.0:
            wait = (o["phase_duration"] - 60.0) / 60.0

        # 7. Per-movement anti-starvation (the root-cause fix for argmax collapse).
        #    Penalise the junction whenever movements that HAVE waiting vehicles are
        #    going unserved while the current phase holds. The penalty is the share
        #    of total demand sitting in the *unserved* movement groups, scaled by how
        #    long the current phase has been held (a proxy for unserved time).
        #
        #    Key properties (vs the old NS/EW-only, >30s-gated term):
        #      • ACTIVE EVEN AT LOW DEMAND — it is share-based, so it is
        #        scale-invariant and gives signal with only 1-3 cars/lane, exactly
        #        where pressure/queue/throughput go flat and argmax turns arbitrary.
        #      • Per-movement (all 6 phase groups), not just NS-vs-EW, so it teaches
        #        "serve every movement that has demand," which is what the live
        #        runner's STARVE/DIRECTION/LEFT overrides do externally.
        #      • ZERO for any group with no demand: if the current phase serves the
        #        only movement that has cars, unserved_share = 0 ⇒ no penalty. It
        #        never rewards/penalises serving empty approaches, so it does not
        #        fight throughput on quiet directions.
        phase = int(o.get("current_phase", 0)) % 6
        dur   = o.get("phase_duration", 0.0)
        group_demand = [
            o.get("north_through", 0) + o.get("south_through", 0),  # phase 0 NS-thru
            o.get("north_left", 0),                                 # phase 1 N-left
            o.get("south_left", 0),                                 # phase 2 S-left
            o.get("east_through", 0) + o.get("west_through", 0),    # phase 3 EW-thru
            o.get("east_left", 0),                                  # phase 4 E-left
            o.get("west_left", 0),                                  # phase 5 W-left
        ]
        total_dem = sum(group_demand)
        starvation = 0.0
        if total_dem > 0:
            excess = min(dur / 45.0, 2.0)
            unserved_share = (total_dem - group_demand[phase]) / total_dem
            starvation = unserved_share * excess

        # 8. Low-demand clearing shaping (Step 3): a small POSITIVE reward for
        #    actually removing vehicles from this junction, in ABSOLUTE terms (the
        #    `throughput` term above is relative to prev_total and goes noisy/flat
        #    with only a handful of cars). Capped and one-sided (only clearing is
        #    rewarded) so it stays small relative to starvation/throughput.
        clear = 0.0
        if prev is not None:
            cleared = _intersection_total(prev[i]) - _intersection_total(o)
            if cleared > 0:
                clear = min(cleared, 5.0) / 5.0

        r = (
            weights.pressure      * pressure
            + weights.queue       * queue
            + weights.throughput  * throughput
            + weights.fairness    * fairness
            + weights.phase_penalty * phase_change
            + weights.wait_penalty  * wait
            + weights.starvation    * starvation
            + weights.clear_bonus   * clear
        )
        rewards.append(r)

    reward_tensor = torch.tensor(rewards, dtype=torch.float32)

    # Green wave: global bonus added uniformly to all nodes
    gw = _compute_green_wave(cur, prev)
    reward_tensor = reward_tensor + weights.green_wave * gw

    return reward_tensor   # (N,)


# ══════════════════════════════════════════════════
#  GAE — Returns (T, N) tensors
# ══════════════════════════════════════════════════

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

    values_ext = values + [next_value]
    advantages_list: List[torch.Tensor] = []

    for t in reversed(range(len(rewards))):
        v_next = values_ext[t + 1]
        v_curr = values_ext[t]
        delta = rewards[t] + gamma * v_next - v_curr
        gae   = delta + gamma * lam * gae
        advantages_list.insert(0, gae.clone())

    advantages = torch.stack(advantages_list)  # (T, N)

    # Stack values directly — works for both scalar [1] and per-junction [J] tensors
    values_stacked = torch.stack([v.to(advantages.device) for v in values])  # [T, N]
    returns = advantages + values_stacked

    if advantages.numel() > 1 and advantages.std() > 0.01:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    if returns.numel() > 1 and returns.std() > 0.01:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return advantages, returns
