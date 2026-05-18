"""
TraFix v2 — Coordinated Multi-Intersection PPO Agent
=====================================================
GCN (spatial) + Multi-Head Attention (coordination) — GRU removed.

v6 updates:
  • NUM_NODE_FEATURES: 10 → 20 (12 per-lane counts + queue + 6-phase one-hot + duration)
  • parse_sumo_observations: 12 lane fields, normalised per-lane
  • compute_reward: pressure/queue/throughput/fairness over 12 lanes
  • _compute_green_wave: through phases are 0 (NS) and 3 (EW) in model space
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    raise ImportError(
        "torch_geometric not found. Install with:\n"
        "  pip install torch-geometric"
    )


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

# Per-lane normalisers: left/right lanes = 15 (single lane), through = 30
_NORM = {
    "north_left":    15.0, "north_through": 30.0, "north_right":  15.0,
    "south_left":    15.0, "south_through": 30.0, "south_right":  15.0,
    "east_left":     15.0, "east_through":  30.0, "east_right":   15.0,
    "west_left":     15.0, "west_through":  30.0, "west_right":   15.0,
}

# Output feature count:
#   [0-11]  12 normalised per-lane counts
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

        # Indices 0-11: per-lane counts normalised
        for key in _LANE_KEYS:
            row.append(o.get(key, 0) / _NORM[key])

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
#  Spatio GNN  (GCN — GRU removed)
# ══════════════════════════════════════════════════

class SpatioTemporalGNN(nn.Module):
    """2-layer GCN with residual + LayerNorm."""

    def __init__(self, num_node_features: int, hidden_dim: int):
        super().__init__()
        self.gcn1 = GCNConv(num_node_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gcn1(x, edge_index))
        h = h + F.relu(self.gcn2(h, edge_index))
        return self.layer_norm(h)


# ══════════════════════════════════════════════════
#  Cross-Intersection Coordination Layer
# ══════════════════════════════════════════════════

class IntersectionCoordinator(nn.Module):
    """Multi-Head Attention for cross-intersection coordination."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        x = node_features.unsqueeze(0)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x.squeeze(0)


# ══════════════════════════════════════════════════
#  Coordinated PPO Agent
# ══════════════════════════════════════════════════

class CoordinatedPPOAgent(nn.Module):
    """Flow: SUMO obs → GCN → Attention → Actor/Critic"""

    def __init__(
        self,
        num_node_features: int = NUM_NODE_FEATURES,
        hidden_dim: int = 128,
        num_actions: int = 6,
        num_heads: int = 4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.25,
        clip_eps: float = 0.2,
        max_grad_norm: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.clip_eps = clip_eps
        self.max_grad_norm = max_grad_norm

        self.st_gnn = SpatioTemporalGNN(num_node_features, hidden_dim)
        self.coordinator = IntersectionCoordinator(hidden_dim, num_heads)

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.st_gnn(x, edge_index)
        coordinated = self.coordinator(features)
        action_probs = F.softmax(self.actor(coordinated), dim=-1)
        state_value = self.critic(coordinated.mean(dim=0))
        return action_probs, state_value

    @torch.no_grad()
    def select_actions(self, x, edge_index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs, value = self.forward(x, edge_index)
        dists = Categorical(probs)
        actions = dists.sample()
        log_probs = dists.log_prob(actions)
        return actions, log_probs, value

    def compute_ppo_loss(self, x, edge_index, old_actions, old_log_probs,
                         advantages, returns) -> Dict[str, torch.Tensor]:
        probs, value = self.forward(x, edge_index)
        dists = Categorical(probs)
        new_log_probs = dists.log_prob(old_actions)
        entropy = dists.entropy().mean()
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(value.squeeze(), returns.mean())
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        return {
            "total": total_loss, "policy": policy_loss.detach(),
            "value": value_loss.detach(), "entropy": entropy.detach(),
        }


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
    fairness:      float =  0.00
    phase_penalty: float = -0.08
    wait_penalty:  float = -0.05
    green_wave:    float =  0.20
    starvation:    float = -0.15


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

        # 7. Directional starvation: penalise holding one through-direction while
        #    the other has vehicles waiting. Grows with time past min-green (30s)
        #    and with the fraction of total demand in the unserved direction.
        #    Zero when total demand is zero, so quiet periods are not penalised.
        starvation = 0.0
        phase = int(o.get("current_phase", 0))
        dur   = o.get("phase_duration", 0.0)
        if dur > 30.0:
            ns_q = sum(o.get(k, 0) for k in _NS_KEYS)
            ew_q = sum(o.get(k, 0) for k in _EW_KEYS)
            total_dir = ns_q + ew_q
            if total_dir > 0:
                excess = min((dur - 30.0) / 60.0, 2.0)
                if phase in (0, 1, 2):        # NS side active — EW is unserved
                    starvation = (ew_q / total_dir) * excess
                else:                          # EW side active — NS is unserved
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


# ══════════════════════════════════════════════════
#  Training Step (Single Epoch)
# ══════════════════════════════════════════════════

def train_step(
    agent: CoordinatedPPOAgent,
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
