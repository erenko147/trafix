"""
TraFix v3 — GConvGRU Tabanlı Koordineli PPO Ajanı
==================================================
v2'den tek fark: SpatioTemporalGNN artık ayrı GCN + GRU katmanları yerine
GConvGRU kullanır — graf evrişimleri GRU kapılarına doğrudan entegre edilmiştir.

Mimari karşılaştırması:
  v2: x → GCNConv → GCNConv → GRU        (önce mekansal, sonra zamansal)
  v3: x → GConvGRU                        (mekansal + zamansal aynı anda)

GConvGRU:
  Z = σ( ChebConv(x) + ChebConv(H) )      # güncelleme kapısı
  R = σ( ChebConv(x) + ChebConv(H) )      # sıfırlama kapısı
  C = tanh( ChebConv(x) + ChebConv(R⊙H) ) # aday gizli durum
  H' = Z⊙H + (1-Z)⊙C

Referans:
  Seo et al., "Structured Sequence Modeling with Graph Convolutional
  Recurrent Networks", ICONIP 2018. https://arxiv.org/abs/1612.07659
  (pytorch-geometric-temporal kaynak kodundan esinlenildi)

Gereksinimler:
  pip install torch-geometric   # torch-geometric-temporal GEREKMİYOR
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

try:
    from torch_geometric.nn import ChebConv
except ImportError:
    raise ImportError(
        "torch_geometric kurulu değil. Lütfen çalıştır:\n"
        "  pip install torch-geometric"
    )


# ══════════════════════════════════════════════════
#  SUMO Veri İşleme  (v2 ile aynı)
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
    count_max: Optional[float] = None,
    duration_max: float = 180.0,
) -> torch.Tensor:
    """
    Args:
        count_max   : Araç sayısı normalizasyonu için tarihsel global maksimum.
                      None ise anlık batch maksimumu kullanılır (legacy davranış).
        duration_max: Faz süresi normalizasyonu için üst limit (saniye).
    """
    sorted_obs = sorted(raw_obs, key=lambda d: d["intersection_id"])
    rows = []
    for obs in sorted_obs:
        row = [float(obs[f]) for f in FEATURE_ORDER]
        rows.append(row)

    x = torch.tensor(rows, dtype=torch.float32, device=device)

    vehicle_cols = slice(0, 4)
    queue_col    = 4
    phase_col    = 5
    duration_col = 6

    # Araç sayıları: rolling global max yoksa anlık batch max'ı kullan
    if count_max is not None:
        v_max = max(count_max, 1.0)
    else:
        v_max = x[:, vehicle_cols].max().clamp(min=1.0).item()
    x[:, vehicle_cols] = (x[:, vehicle_cols] / v_max).clamp(max=1.0)

    q_max = x[:, queue_col].max().clamp(min=1.0)
    x[:, queue_col] = x[:, queue_col] / q_max

    x[:, phase_col]    = x[:, phase_col] / max(4.0, x[:, phase_col].max().item())
    x[:, duration_col] = (x[:, duration_col] / duration_max).clamp(max=1.0)

    return x


# ══════════════════════════════════════════════════
#  GConvGRU — ChebConv tabanlı manuel uygulama
#  (pytorch-geometric-temporal iç algoritmasını taklit eder)
# ══════════════════════════════════════════════════

class GConvGRU(nn.Module):
    """
    Graf Evrişimli GRU hücresi.

    Standart GRU'dan farkı: tam bağlantılı doğrusal katmanlar yerine
    Chebyshev graf evrişimleri kullanılır. Böylece her kapı (Z, R, C)
    komşu kavşakların bilgisini ağırlıklı olarak dahil eder.

    Args:
        in_channels : giriş özellik boyutu (7 = NUM_NODE_FEATURES)
        out_channels: gizli durum boyutu (hidden_dim)
        K           : Chebyshev polinom derecesi (2 ≈ 2-hop komşuluk)
    """

    def __init__(self, in_channels: int, out_channels: int, K: int = 2):
        super().__init__()
        self.out_channels = out_channels

        # Güncelleme kapısı (Z)
        self.conv_z_x = ChebConv(in_channels,  out_channels, K)
        self.conv_z_h = ChebConv(out_channels, out_channels, K)

        # Sıfırlama kapısı (R)
        self.conv_r_x = ChebConv(in_channels,  out_channels, K)
        self.conv_r_h = ChebConv(out_channels, out_channels, K)

        # Aday gizli durum (C)
        self.conv_c_x = ChebConv(in_channels,  out_channels, K)
        self.conv_c_h = ChebConv(out_channels, out_channels, K)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        H: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x          : (N, in_channels)
            edge_index : (2, E)
            H          : (N, out_channels) veya None → sıfır başlatılır
        Returns:
            H_new      : (N, out_channels)
        """
        if H is None:
            H = torch.zeros(x.size(0), self.out_channels, device=x.device)

        Z = torch.sigmoid(self.conv_z_x(x, edge_index) + self.conv_z_h(H, edge_index))
        R = torch.sigmoid(self.conv_r_x(x, edge_index) + self.conv_r_h(H, edge_index))
        C = torch.tanh(   self.conv_c_x(x, edge_index) + self.conv_c_h(R * H, edge_index))

        return Z * H + (1 - Z) * C


# ══════════════════════════════════════════════════
#  Spatio-Temporal GNN  (GConvGRU — v3 yeniliği)
# ══════════════════════════════════════════════════

class SpatioTemporalGNN(nn.Module):
    """
    v2: GCN → GRU (sıralı)
    v3: GConvGRU  (eş zamanlı — bu dosya)

    Arayüz v2 ile birebir aynıdır; CoordinatedPPOAgent değişmez.
    """

    def __init__(self, num_node_features: int, hidden_dim: int):
        super().__init__()
        self.gconv_gru = GConvGRU(
            in_channels=num_node_features,
            out_channels=hidden_dim,
            K=2,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x           : (N, F)
            edge_index  : (2, E)
            hidden_state: (1, N, H)   — v2 ile uyumlu format
        Returns:
            features        : (N, H)
            new_hidden_state: (1, N, H)
        """
        H_prev = hidden_state.squeeze(0)          # (1,N,H) → (N,H)
        H_new  = self.gconv_gru(x, edge_index, H_prev)
        H_new  = self.layer_norm(H_new)
        return H_new, H_new.unsqueeze(0)          # (N,H), (1,N,H)


# ══════════════════════════════════════════════════
#  Kavşaklar-Arası Koordinasyon  (v2 ile aynı)
# ══════════════════════════════════════════════════

class IntersectionCoordinator(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
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
#  PPO Aktör-Kritik Ajan  (v2 ile aynı)
# ══════════════════════════════════════════════════

class CoordinatedPPOAgent(nn.Module):
    def __init__(
        self,
        num_node_features: int = NUM_NODE_FEATURES,
        hidden_dim: int = 128,
        num_actions: int = 4,
        num_heads: int = 4,
        entropy_coef: float = 0.02,
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

        self.st_gnn      = SpatioTemporalGNN(num_node_features, hidden_dim)
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
        hidden_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, new_hidden = self.st_gnn(x, edge_index, hidden_state)
        coordinated  = self.coordinator(features)
        action_probs = F.softmax(self.actor(coordinated), dim=-1)
        global_feat  = coordinated.mean(dim=0)
        state_value  = self.critic(global_feat)
        return action_probs, state_value, new_hidden

    @torch.no_grad()
    def select_actions(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        probs, value, new_hidden = self.forward(x, edge_index, hidden_state)
        dists     = Categorical(probs)
        actions   = dists.sample()
        log_probs = dists.log_prob(actions)
        return actions, log_probs, value, new_hidden

    def compute_ppo_loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        hidden_state: torch.Tensor,
        old_actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        probs, value, _ = self.forward(x, edge_index, hidden_state)
        dists        = Categorical(probs)
        new_log_probs = dists.log_prob(old_actions)
        entropy      = dists.entropy().mean()

        ratio  = torch.exp(new_log_probs - old_log_probs)
        surr1  = ratio * advantages
        surr2  = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
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
#  Ödül, GAE, train_step  (v2 ile birebir aynı)
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


def compute_reward(
    current_obs: List[Dict],
    previous_obs: Optional[List[Dict]],
    previous_actions: Optional[torch.Tensor],
    current_actions: torch.Tensor,
    weights: RewardWeights = RewardWeights(),
) -> torch.Tensor:
    cur  = sorted(current_obs,  key=lambda d: d["intersection_id"])
    prev = sorted(previous_obs, key=lambda d: d["intersection_id"]) \
        if previous_obs is not None else None

    total_vehicles = sum(_intersection_total(o) for o in cur)
    pressure = total_vehicles / max(len(cur) * 40.0, 1.0)

    total_queue = sum(o["queue_length"] for o in cur)
    queue = total_queue / max(len(cur) * 100.0, 1.0)

    throughput = 0.0
    if prev is not None:
        prev_vehicles = sum(_intersection_total(o) for o in prev)
        throughput = (prev_vehicles - total_vehicles) / max(prev_vehicles, 1.0)
        throughput = max(throughput, -1.0)

    fairness_scores = []
    for o in cur:
        counts = [o["north_count"], o["south_count"], o["east_count"], o["west_count"]]
        mean_c = sum(counts) / 4.0
        var_c  = sum((c - mean_c) ** 2 for c in counts) / 4.0
        fairness_scores.append(math.sqrt(var_c) / max(mean_c, 1.0))
    fairness = sum(fairness_scores) / len(fairness_scores)

    phase_changes = 0.0
    if previous_actions is not None:
        phase_changes = (current_actions != previous_actions).float().mean().item()

    wait_penalty = 0.0
    for o in cur:
        if o["phase_duration"] > 60.0:
            wait_penalty += (o["phase_duration"] - 60.0) / 60.0
    wait_penalty /= len(cur)

    # ── Dinamik Yeşil Dalga ──
    green_wave_score = 0.0
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
            green_wave_score += base
            if prev is not None:
                prev_dst_queue = prev[dst_id]["queue_length"]
                curr_dst_queue = dst["queue_length"]
                if curr_dst_queue < prev_dst_queue:
                    reduction = (prev_dst_queue - curr_dst_queue) / max(prev_dst_queue, 1.0)
                    green_wave_score += reduction

    green_wave_score /= max(len(_GREEN_WAVE_EDGES), 1)

    reward = (
        weights.pressure      * pressure
        + weights.queue        * queue
        + weights.throughput   * throughput
        + weights.fairness     * fairness
        + weights.phase_penalty * phase_changes
        + weights.wait_penalty  * wait_penalty
        + weights.green_wave   * green_wave_score
    )
    return torch.tensor(reward, dtype=torch.float32)


def compute_gae(
    rewards: List[torch.Tensor],
    values: List[torch.Tensor],
    next_value: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    advantages  = []
    gae         = torch.tensor(0.0)
    values_ext  = values + [next_value]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_ext[t + 1] - values_ext[t]
        gae   = delta + gamma * lam * gae
        advantages.insert(0, gae)

    advantages = torch.stack(advantages)
    returns    = advantages + torch.stack(values)

    if advantages.numel() > 1 and advantages.std() > 0.01:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def train_step(
    agent: CoordinatedPPOAgent,
    optimizer: torch.optim.Optimizer,
    rollout: Dict,
    ppo_epochs: int = 4,
) -> Dict[str, float]:
    advantages, returns = compute_gae(
        rollout["rewards"],
        rollout["values"],
        rollout["next_value"],
    )

    total_metrics = {"total": 0, "policy": 0, "value": 0, "entropy": 0}
    T = len(rollout["rewards"])

    for _ in range(ppo_epochs):
        for t in range(T):
            losses = agent.compute_ppo_loss(
                x=rollout["observations"][t],
                edge_index=rollout["edge_index"],
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
