"""
TraFix v2 — Coordinated Multi-Intersection PPO Agent
=====================================================
GCN (spatial) + Multi-Head Attention (coordination) — GRU removed.

Changes vs original:
  • GRU removed — avoids stale-hidden-state PPO bug at 10s decision intervals
  • Per-node (N,) rewards — fixes credit assignment (was single scalar broadcast)
  • Fixed normalisation scales — counts/30, queue/150, phase one-hot, dur/120
  • NUM_NODE_FEATURES: 7 → 10 (4 one-hot phase bits replace 1 raw phase int)
  • compute_gae returns (T, N) advantages and (T, N) returns
  • entropy_coef 0.02 → 0.005, value_coef 0.5 → 0.25
  • Return normalisation added alongside advantage normalisation
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
        "torch_geometric kurulu değil. Lütfen çalıştır:\n"
        "  pip install torch-geometric"
    )


# ══════════════════════════════════════════════════
#  SUMO Veri İşleme
# ══════════════════════════════════════════════════

# Raw SUMO field names (7 fields in → 10 features out after one-hot phase)
FEATURE_ORDER = [
    "north_count",
    "south_count",
    "east_count",
    "west_count",
    "queue_length",
    "current_phase",
    "phase_duration",
]

# Output feature count after preprocessing:
#   [north/30, south/30, east/30, west/30, queue/150,
#    phase_0, phase_1, phase_2, phase_3, duration/120]
NUM_NODE_FEATURES = 10


def parse_sumo_observations(
    raw_obs: List[Dict],
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    SUMO intersection JSON list → (num_nodes, 10) normalised tensor.

    Fixed scales (not batch-max) preserve absolute magnitude:
      vehicle counts : / 30.0   (clipped at 1.0)
      queue length   : / 150.0  (clipped at 1.0)
      phase          : one-hot 4-bit
      phase duration : / 120.0  (clipped at 1.0)
    """
    sorted_obs = sorted(raw_obs, key=lambda d: d["intersection_id"])

    rows = []
    for obs in sorted_obs:
        north    = min(obs["north_count"]  / 30.0,  1.0)
        south    = min(obs["south_count"]  / 30.0,  1.0)
        east     = min(obs["east_count"]   / 30.0,  1.0)
        west     = min(obs["west_count"]   / 30.0,  1.0)
        queue    = min(obs["queue_length"] / 150.0, 1.0)

        # One-hot encode model phase (0–3)
        phase = int(obs["current_phase"]) % 4
        phase_oh = [0.0, 0.0, 0.0, 0.0]
        phase_oh[phase] = 1.0

        duration = min(obs["phase_duration"] / 120.0, 1.0)

        rows.append([north, south, east, west, queue] + phase_oh + [duration])

    return torch.tensor(rows, dtype=torch.float32, device=device)


# ══════════════════════════════════════════════════
#  Spatio GNN  (GCN only — GRU removed)
# ══════════════════════════════════════════════════

class SpatioTemporalGNN(nn.Module):
    """2-layer GCN with residual + LayerNorm. GRU removed."""

    def __init__(self, num_node_features: int, hidden_dim: int):
        super().__init__()
        self.gcn1 = GCNConv(num_node_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x          : (N, F)
            edge_index : (2, E)
        Returns:
            features   : (N, H)
        """
        h = F.relu(self.gcn1(x, edge_index))
        h = h + F.relu(self.gcn2(h, edge_index))
        return self.layer_norm(h)


# ══════════════════════════════════════════════════
#  Kavşaklar-Arası Koordinasyon Katmanı
# ══════════════════════════════════════════════════

class IntersectionCoordinator(nn.Module):
    """Multi-Head Attention for cross-intersection coordination."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        x = node_features.unsqueeze(0)        # (1, N, H)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x.squeeze(0)                   # (N, H)


# ══════════════════════════════════════════════════
#  PPO Aktör-Kritik Ajan (Koordineli)
# ══════════════════════════════════════════════════

class CoordinatedPPOAgent(nn.Module):
    """
    Flow:  SUMO obs → GCN → Attention → Actor/Critic

    • Actor  → (N, A)  per-intersection action probabilities
    • Critic → (1,)    single global value (mean-pooled)
    """

    def __init__(
        self,
        num_node_features: int = NUM_NODE_FEATURES,   # 10
        hidden_dim: int = 128,
        num_actions: int = 4,
        num_heads: int = 4,
        entropy_coef: float = 0.005,
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
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action_probs : (N, A)
            state_value  : (1,)
        """
        features    = self.st_gnn(x, edge_index)
        coordinated = self.coordinator(features)          # (N, H)

        action_probs = F.softmax(self.actor(coordinated), dim=-1)

        global_feat  = coordinated.mean(dim=0)            # (H,)
        state_value  = self.critic(global_feat)            # (1,)

        return action_probs, state_value

    @torch.no_grad()
    def select_actions(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            actions     : (N,)
            log_probs   : (N,)
            state_value : (1,)
        """
        probs, value = self.forward(x, edge_index)
        dists     = Categorical(probs)
        actions   = dists.sample()
        log_probs = dists.log_prob(actions)
        return actions, log_probs, value

    def compute_ppo_loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        old_actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,   # (N,)
        returns: torch.Tensor,      # (N,)
    ) -> Dict[str, torch.Tensor]:
        """
        Clipped PPO objective + Value loss + Entropy bonus.

        advantages and returns are (N,) — real per-node signals, no expand hack.
        Value loss compares scalar critic against mean of per-node returns.
        """
        probs, value = self.forward(x, edge_index)
        dists = Categorical(probs)

        new_log_probs = dists.log_prob(old_actions)   # (N,)
        entropy       = dists.entropy().mean()

        ratio  = torch.exp(new_log_probs - old_log_probs)
        surr1  = ratio * advantages
        surr2  = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(value.squeeze(), returns.mean())

        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy
        )

        return {
            "total":   total_loss,
            "policy":  policy_loss.detach(),
            "value":   value_loss.detach(),
            "entropy": entropy.detach(),
        }


# ══════════════════════════════════════════════════
#  Ödül Fonksiyonu  — Per-Node (N,) Tensor
# ══════════════════════════════════════════════════

_GREEN_WAVE_EDGES: List[Tuple[int, int]] = [(0, 1), (1, 2), (1, 3), (3, 4)]
_PLATOON_THRESHOLD: int = 20


@dataclass
class RewardWeights:
    pressure:      float = -0.30
    queue:         float = -0.25
    throughput:    float =  0.25
    fairness:      float = -0.10
    phase_penalty: float = -0.08
    wait_penalty:  float = -0.05
    green_wave:    float =  0.20


def _intersection_total(o: Dict) -> int:
    return o["north_count"] + o["south_count"] + o["east_count"] + o["west_count"]


def _compute_green_wave(cur: List[Dict], prev: Optional[List[Dict]]) -> float:
    """Global cross-junction green wave score (distributed evenly to all nodes)."""
    score = 0.0
    for (src_id, dst_id) in _GREEN_WAVE_EDGES:
        src = cur[src_id]
        dst = cur[dst_id]
        src_total    = _intersection_total(src)
        src_phase    = src["current_phase"]
        dst_phase    = dst["current_phase"]
        src_is_green = src_phase in (0, 2)
        dst_aligned  = dst_phase in (0, 2) and (dst_phase % 2 == src_phase % 2)
        has_platoon  = src_total >= _PLATOON_THRESHOLD

        if has_platoon and src_is_green and dst_aligned:
            base = min(src_total / _PLATOON_THRESHOLD, 2.0)
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

    Each node's reward is computed from its own local signals:
      pressure, queue, throughput, fairness, phase_penalty, wait_penalty.
    Green wave is a global cross-junction bonus distributed uniformly.
    """
    cur  = sorted(current_obs,  key=lambda d: d["intersection_id"])
    prev = sorted(previous_obs, key=lambda d: d["intersection_id"]) \
           if previous_obs is not None else None

    rewards = []
    for i, o in enumerate(cur):
        # 1. Pressure
        pressure = _intersection_total(o) / 40.0

        # 2. Queue
        queue = o["queue_length"] / 100.0

        # 3. Throughput — delta for this intersection only
        throughput = 0.0
        if prev is not None:
            prev_total = _intersection_total(prev[i])
            cur_total  = _intersection_total(o)
            throughput = (prev_total - cur_total) / max(prev_total, 1.0)
            throughput = max(throughput, -1.0)

        # 4. Fairness — directional variance at this intersection
        counts = [o["north_count"], o["south_count"], o["east_count"], o["west_count"]]
        mean_c = sum(counts) / 4.0
        var_c  = sum((c - mean_c) ** 2 for c in counts) / 4.0
        fairness = math.sqrt(var_c) / max(mean_c, 1.0)

        # 5. Phase stability — did THIS intersection change phase
        phase_change = 0.0
        if previous_actions is not None:
            phase_change = float(current_actions[i].item() != previous_actions[i].item())

        # 6. Wait penalty — this intersection's own duration
        wait = 0.0
        if o["phase_duration"] > 60.0:
            wait = (o["phase_duration"] - 60.0) / 60.0

        r = (
            weights.pressure      * pressure
            + weights.queue       * queue
            + weights.throughput  * throughput
            + weights.fairness    * fairness
            + weights.phase_penalty * phase_change
            + weights.wait_penalty  * wait
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
    rewards: List[torch.Tensor],   # each (N,)
    values: List[torch.Tensor],    # each scalar (1,) or ()
    next_value: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GAE-Lambda advantage estimation.

    rewards and values can have different shapes: rewards are (N,),
    values are scalars (critic output). Scalar values broadcast over N
    in the delta calculation, yielding (N,) per-step advantages.

    Returns:
        advantages : (T, N)  — normalised
        returns    : (T, N)  — normalised
    """
    N   = rewards[0].shape[0]
    gae = torch.zeros(N, dtype=torch.float32, device=rewards[0].device)

    values_ext = values + [next_value]

    advantages_list: List[torch.Tensor] = []
    for t in reversed(range(len(rewards))):
        v_next = values_ext[t + 1]
        v_curr = values_ext[t]
        # Both v_next/v_curr are scalars → broadcast to (N,)
        delta = rewards[t] + gamma * v_next - v_curr
        gae   = delta + gamma * lam * gae
        advantages_list.insert(0, gae.clone())

    advantages = torch.stack(advantages_list)  # (T, N)

    # Returns = unnormalised advantages + values (scalar broadcast to N)
    values_expanded = torch.stack([
        torch.full((N,), v.item(), dtype=torch.float32, device=advantages.device)
        for v in values
    ])  # (T, N)
    returns = advantages + values_expanded

    # Advantage normalisation
    if advantages.numel() > 1 and advantages.std() > 0.01:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Return normalisation
    if returns.numel() > 1 and returns.std() > 0.01:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return advantages, returns


# ══════════════════════════════════════════════════
#  Eğitim Döngüsü (Tek epoch)
# ══════════════════════════════════════════════════

def train_step(
    agent: CoordinatedPPOAgent,
    optimizer: torch.optim.Optimizer,
    rollout: Dict,
    ppo_epochs: int = 4,
) -> Dict[str, float]:
    """
    Runs multiple PPO epochs over one rollout.

    rollout keys:
        observations : List[Tensor (N, F)]
        edge_index   : Tensor (2, E)
        actions      : List[Tensor (N,)]
        log_probs    : List[Tensor (N,)]
        rewards      : List[Tensor (N,)]   — per-node
        values       : List[Tensor scalar]
        next_value   : Tensor scalar
    """
    advantages, returns = compute_gae(
        rollout["rewards"],
        rollout["values"],
        rollout["next_value"],
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
                advantages=advantages[t],   # (N,) — real per-node signal
                returns=returns[t],          # (N,)
            )

            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(agent.parameters(), agent.max_grad_norm)
            optimizer.step()

            for k in total_metrics:
                total_metrics[k] += losses[k].item()

    n = ppo_epochs * T
    return {k: v / max(n, 1) for k, v in total_metrics.items()}
