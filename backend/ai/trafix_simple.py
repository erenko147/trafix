"""
TraFix Simple — Demo Günü Modeli
=================================
GCN / GConvGRU / MultiHeadAttention YOK.
Sadece: özellik projeksiyonu → GRU → Actor/Critic

Neden sade?
  • torch_geometric bağımlılığı yok → kurulum sorunu yok
  • edge_index yok → topoloji uyuşmazlığı hatası yok
  • Daha az katman → entropi çöküşü riski düşük
  • Daha hızlı eğitim (hidden_dim=64, rollout=32)

Düzeltilen bilinen hatalar (v1'de tespit edildi):
  ✓ Entropi çöküşü  : coef=0.05, decay=0.9998, min=0.01
  ✓ Advantage normalizasyonu : std < 0.01 ise normalize etme
  ✓ Phase duration   : elapsed time (not programmed max)
  ✓ Decision interval: 10 sn (eğitim = inference)
  ✓ Minimum green time: run_sumo_live.py'de zorunlu
  ✓ Graf topolojisi  : YOK — mismatch ihtimali sıfır
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


# ══════════════════════════════════════════════════
#  SUMO Veri İşleme
# ══════════════════════════════════════════════════

FEATURE_ORDER = [
    "north_count",
    "south_count",
    "east_count",
    "west_count",
    "queue_length",
    "current_phase",
    "phase_duration",
]

NUM_NODE_FEATURES = len(FEATURE_ORDER)  # 7


def parse_sumo_observations(
    raw_obs: List[Dict],
    device: torch.device = torch.device("cpu"),
    count_max: Optional[float] = None,   # main.py ile uyumluluk için kabul edilir
    duration_max: float = 120.0,
) -> torch.Tensor:
    sorted_obs = sorted(raw_obs, key=lambda d: d["intersection_id"])
    rows = [[float(obs[f]) for f in FEATURE_ORDER] for obs in sorted_obs]
    x = torch.tensor(rows, dtype=torch.float32, device=device)

    v_max = float(count_max) if count_max is not None else x[:, :4].max().clamp(min=1.0).item()
    v_max = max(v_max, 1.0)
    x[:, :4] = (x[:, :4] / v_max).clamp(max=1.0)

    q_max = x[:, 4].max().clamp(min=1.0)
    x[:, 4] = x[:, 4] / q_max

    x[:, 5] = x[:, 5] / max(4.0, x[:, 5].max().item())
    x[:, 6] = (x[:, 6] / duration_max).clamp(max=1.0)

    return x


# ══════════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════════

class SimplePPOAgent(nn.Module):
    """
    Akış: (N, 7) → Linear → ReLU → GRU → Actor(N, 4) / Critic(1,)

    Her kavşak bağımsız olarak işlenir (N = num_intersections).
    GRU gizli durumu istekler/adımlar arasında dışarıdan beslenir
    (PPO rollout replay için zorunlu — dahili durum kullanılamaz).
    """

    def __init__(
        self,
        num_node_features: int = NUM_NODE_FEATURES,
        hidden_dim: int = 64,
        num_actions: int = 4,
        entropy_coef: float = 0.05,
        value_coef: float = 0.5,
        clip_eps: float = 0.2,
        max_grad_norm: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.num_actions   = num_actions
        self.entropy_coef  = entropy_coef
        self.value_coef    = value_coef
        self.clip_eps      = clip_eps
        self.max_grad_norm = max_grad_norm

        # Özellik projeksiyonu — GRU'dan önce lineer dönüşüm
        self.proj = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.ReLU(),
        )

        # Zamansal hafıza — her kavşak bağımsız batch öğesi
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Actor — her kavşak için aksiyon dağılımı
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

        # Critic — ağ geneli tek value (global ortalama üzerinden)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x           : (N, 7)      kavşak gözlemleri
            hidden_state: (1, N, H)   GRU gizli durum
        Returns:
            action_probs : (N, 4)
            state_value  : (1,)
            new_hidden   : (1, N, H)
        """
        h = self.proj(x)                           # (N, H)
        out, new_hidden = self.gru(                # (N, 1, H), (1, N, H)
            h.unsqueeze(1), hidden_state
        )
        out = out.squeeze(1)                       # (N, H)

        action_probs = F.softmax(self.actor(out), dim=-1)  # (N, 4)
        # Per-intersection value estimates averaged — critic sees each
        # intersection's actual state, not a blurred average of all of them
        state_value  = self.critic(out).mean()             # (N,1) → scalar → keep as (1,)
        state_value  = state_value.view(1)

        return action_probs, state_value, new_hidden

    @torch.no_grad()
    def select_actions(
        self,
        x: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        probs, value, new_hidden = self.forward(x, hidden_state)
        dists     = Categorical(probs)
        actions   = dists.sample()
        log_probs = dists.log_prob(actions)
        return actions, log_probs, value, new_hidden

    def compute_ppo_loss(
        self,
        x: torch.Tensor,
        hidden_state: torch.Tensor,
        old_actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        probs, value, _ = self.forward(x, hidden_state)
        dists         = Categorical(probs)
        new_log_probs = dists.log_prob(old_actions)
        entropy       = dists.entropy().mean()

        ratio       = torch.exp(new_log_probs - old_log_probs)
        surr1       = ratio * advantages
        surr2       = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss  = F.mse_loss(value.squeeze(), returns.squeeze())
        total_loss  = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        return {
            "total":   total_loss,
            "policy":  policy_loss.detach(),
            "value":   value_loss.detach(),
            "entropy": entropy.detach(),
        }

    def init_hidden(self, num_nodes: int = 5, device: torch.device = None) -> torch.Tensor:
        dev = device or next(self.parameters()).device
        return torch.zeros(1, num_nodes, self.hidden_dim, device=dev)


# ══════════════════════════════════════════════════
#  Ödül Fonksiyonu
# ══════════════════════════════════════════════════

@dataclass
class RewardWeights:
    pressure:      float = -0.40
    queue:         float = -0.30
    throughput:    float =  0.30
    phase_penalty: float = -0.10
    wait_penalty:  float = -0.05


def compute_reward(
    current_obs: List[Dict],
    previous_obs: Optional[List[Dict]],
    previous_actions: Optional[torch.Tensor],
    current_actions: torch.Tensor,
    weights: RewardWeights = RewardWeights(),
) -> torch.Tensor:
    cur = sorted(current_obs, key=lambda d: d["intersection_id"])

    def total(o):
        return o["north_count"] + o["south_count"] + o["east_count"] + o["west_count"]

    # 1. Basınç
    pressure = sum(total(o) for o in cur) / max(len(cur) * 40.0, 1.0)

    # 2. Kuyruk
    queue = sum(o["queue_length"] for o in cur) / max(len(cur) * 100.0, 1.0)

    # 3. Throughput
    throughput = 0.0
    if previous_obs is not None:
        prev = sorted(previous_obs, key=lambda d: d["intersection_id"])
        prev_total = sum(total(o) for o in prev)
        cur_total  = sum(total(o) for o in cur)
        throughput = (prev_total - cur_total) / max(prev_total, 1.0)
        throughput = max(throughput, -1.0)

    # 4. Faz stabilitesi
    phase_changes = 0.0
    if previous_actions is not None:
        phase_changes = (current_actions != previous_actions).float().mean().item()

    # 5. Bekleme cezası (>60 sn aynı fazda)
    wait_penalty = 0.0
    for o in cur:
        if o["phase_duration"] > 60.0:
            wait_penalty += (o["phase_duration"] - 60.0) / 60.0
    wait_penalty /= len(cur)

    reward = (
        weights.pressure      * pressure
        + weights.queue        * queue
        + weights.throughput   * throughput
        + weights.phase_penalty * phase_changes
        + weights.wait_penalty  * wait_penalty
    )
    return torch.tensor(reward, dtype=torch.float32)


# ══════════════════════════════════════════════════
#  GAE
# ══════════════════════════════════════════════════

def compute_gae(
    rewards: List[torch.Tensor],
    values: List[torch.Tensor],
    next_value: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    advantages = []
    gae        = torch.tensor(0.0)
    values_ext = values + [next_value]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_ext[t + 1] - values_ext[t]
        gae   = delta + gamma * lam * gae
        advantages.insert(0, gae)

    advantages = torch.stack(advantages)
    returns    = advantages + torch.stack(values)

    # Std eşik koruması: tüm ödüller aynıysa normalize etme
    if advantages.numel() > 1 and advantages.std() > 0.01:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


# ══════════════════════════════════════════════════
#  Eğitim Adımı
# ══════════════════════════════════════════════════

def train_step(
    agent: SimplePPOAgent,
    optimizer: torch.optim.Optimizer,
    rollout: Dict,
    ppo_epochs: int = 4,
) -> Dict[str, float]:
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
                hidden_state=rollout["hidden_states"][t],
                old_actions=rollout["actions"][t],
                old_log_probs=rollout["log_probs"][t],
                advantages=advantages[t].expand(rollout["actions"][t].shape[0]),
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
