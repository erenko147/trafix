"""
TraFix v2 — Coordinated Multi-Intersection PPO Agent
=====================================================
GCN (mekansal) + GRU (zamansal) + Multi-Head Attention (kavşaklar arası koordinasyon)

Yenilikler (v1'e kıyasla):
  • Multi-Head Attention ile kavşaklar arası ortak karar mekanizması
  • Entropi bonusu ile keşif teşviki
  • Basınç + kuyruk + adillik + stabilite tabanlı ödül fonksiyonu
  • SUMO JSON formatıyla doğrudan uyumlu veri işleme
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

# SUMO'dan gelen ham JSON → tensor dönüşümü
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
    SUMO'dan gelen kavşak JSON listesini model girdisine çevirir.

    Args:
        raw_obs: Her eleman şu formatta bir dict:
            {
                "intersection_id": 2,
                "north_count": 4, "south_count": 2,
                "east_count": 7, "west_count": 1,
                "queue_length": 34.5,
                "current_phase": 2, "phase_duration": 18.0
            }
        count_max: Araç sayısı normalizasyonu için tarihsel global maksimum.
            None ise anlık batch maksimumu kullanılır (legacy davranış).
        duration_max: Faz süresi normalizasyonu için üst limit (saniye).
            Varsayılan 180s — 120s'yi aşan fazları kırpmaz.
    Returns:
        (num_nodes, 7) — normalize edilmiş özellik tensörü
    """
    # intersection_id'ye göre sırala → tutarlı node sırası
    sorted_obs = sorted(raw_obs, key=lambda d: d["intersection_id"])

    rows = []
    for obs in sorted_obs:
        row = [float(obs[f]) for f in FEATURE_ORDER]
        rows.append(row)

    x = torch.tensor(rows, dtype=torch.float32, device=device)

    # ── Normalizasyon ──
    vehicle_cols = slice(0, 4)          # north/south/east/west
    queue_col = 4
    phase_col = 5
    duration_col = 6

    # Araç sayıları: rolling global max yoksa anlık batch max'ı kullan
    if count_max is not None:
        v_max = max(count_max, 1.0)
    else:
        v_max = x[:, vehicle_cols].max().clamp(min=1.0).item()
    x[:, vehicle_cols] = (x[:, vehicle_cols] / v_max).clamp(max=1.0)

    q_max = x[:, queue_col].max().clamp(min=1.0)
    x[:, queue_col] = x[:, queue_col] / q_max

    # Faz: one-hot yerine basit normalize (0-1 arasına)
    x[:, phase_col] = x[:, phase_col] / max(4.0, x[:, phase_col].max().item())

    # Süre: adaptif üst limitle normalize, >1.0 değerler kırpılır
    x[:, duration_col] = (x[:, duration_col] / duration_max).clamp(max=1.0)

    return x


# ══════════════════════════════════════════════════
#  Spatio-Temporal GNN  (GCN + GRU)
# ══════════════════════════════════════════════════

class SpatioTemporalGNN(nn.Module):
    """GCN ile mekansal toplama → GRU ile zamansal hafıza."""

    def __init__(self, num_node_features: int, hidden_dim: int):
        super().__init__()
        self.gcn1 = GCNConv(num_node_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x           : (N, F)          kavşak özellikleri
            edge_index  : (2, E)          graf topolojisi
            hidden_state: (1, N, H)       GRU önceki gizli durum
        Returns:
            features        : (N, H)
            new_hidden_state: (1, N, H)
        """
        # 2-katmanlı GCN + residual
        h = F.relu(self.gcn1(x, edge_index))
        h = h + F.relu(self.gcn2(h, edge_index))  # skip connection
        h = self.layer_norm(h)

        # GRU — her kavşak bağımsız batch öğesi
        out, new_hidden = self.gru(h.unsqueeze(1), hidden_state)
        return out.squeeze(1), new_hidden


# ══════════════════════════════════════════════════
#  Kavşaklar-Arası Koordinasyon Katmanı
# ══════════════════════════════════════════════════

class IntersectionCoordinator(nn.Module):
    """
    Multi-Head Attention ile kavşaklar birbirlerinin durumunu
    görerek ORTAK karar alır.

    Neden gerekli: Sıralı ışıklarda yeşil dalga (green wave) oluşturmak,
    bir kavşağın taşmasını komşuların önceden görmesi vb.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,       # (batch, seq, dim)
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
        """
        Args:
            node_features: (N, H) — GNN çıkışı
        Returns:
            coordinated  : (N, H) — koordinasyon sonrası özellikler
        """
        # (N, H) → (1, N, H)  attention için batch boyutu
        x = node_features.unsqueeze(0)

        # Self-attention + residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN + residual
        x = self.norm2(x + self.ffn(x))

        return x.squeeze(0)  # (N, H)


# ══════════════════════════════════════════════════
#  PPO Aktör-Kritik Ajan (Koordineli)
# ══════════════════════════════════════════════════

class CoordinatedPPOAgent(nn.Module):
    """
    Akış:  SUMO obs → GCN+GRU → Attention Koordinasyon → Actor/Critic

    • Actor  → (N, A)  her kavşak için aksiyon olasılıkları
    • Critic → (1,)    ağ geneli tek value
    """

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
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.clip_eps = clip_eps
        self.max_grad_norm = max_grad_norm

        # Omurga
        self.st_gnn = SpatioTemporalGNN(num_node_features, hidden_dim)
        self.coordinator = IntersectionCoordinator(hidden_dim, num_heads)

        # Actor — koordine edilmiş özelliklerden aksiyon
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

        # Critic — global state'den tek value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    # ── İleri Geçiş ──────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            action_probs : (N, A)
            state_value  : (1,)
            hidden_state : (1, N, H)
        """
        features, new_hidden = self.st_gnn(x, edge_index, hidden_state)
        coordinated = self.coordinator(features)  # (N, H)

        action_probs = F.softmax(self.actor(coordinated), dim=-1)

        # Critic: global ortalama → tek skaler
        global_feat = coordinated.mean(dim=0)  # (H,)
        state_value = self.critic(global_feat)  # (1,)

        return action_probs, state_value, new_hidden

    # ── Aksiyon Seçimi ────────────────────────────
    @torch.no_grad()
    def select_actions(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Eğitimde kullanılacak aksiyon, log_prob, value ve yeni hidden döner.

        Returns:
            actions     : (N,)     — her kavşak için seçilen aksiyon
            log_probs   : (N,)     — seçilen aksiyonların log olasılıkları
            state_value : (1,)
            new_hidden  : (1, N, H)
        """
        probs, value, new_hidden = self.forward(x, edge_index, hidden_state)
        dists = Categorical(probs)               # N bağımsız dağılım
        actions = dists.sample()                  # (N,)
        log_probs = dists.log_prob(actions)       # (N,)
        return actions, log_probs, value, new_hidden

    # ── PPO Loss (Entropi Bonuslu) ────────────────
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
        """
        Clipped PPO objective + Value loss + Entropy bonus.

        Args:
            old_actions   : (N,)
            old_log_probs : (N,)
            advantages    : (N,)  — GAE veya basit advantage
            returns       : (1,)  — hedef value (discounted return)
        Returns:
            Dict with 'total', 'policy', 'value', 'entropy' losses
        """
        probs, value, _ = self.forward(x, edge_index, hidden_state)
        dists = Categorical(probs)

        new_log_probs = dists.log_prob(old_actions)     # (N,)
        entropy = dists.entropy().mean()                # skaler

        # ── Policy Loss (PPO-Clip) ──
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # ── Value Loss ──
        value_loss = F.mse_loss(value.squeeze(), returns.squeeze())

        # ── Toplam: policy + value - entropy bonus ──
        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy      # entropi bonusu (keşif)
        )

        return {
            "total": total_loss,
            "policy": policy_loss.detach(),
            "value": value_loss.detach(),
            "entropy": entropy.detach(),
        }

    # ── Yardımcılar ──────────────────────────────
    def init_hidden(self, num_nodes: int = 5, device: torch.device = None) -> torch.Tensor:
        dev = device or next(self.parameters()).device
        return torch.zeros(1, num_nodes, self.hidden_dim, device=dev)


# ══════════════════════════════════════════════════
#  Ödül Fonksiyonu  (SUMO Verisi ile)
# ══════════════════════════════════════════════════

# Ağ topolojisi kenarları — map.net.xml'deki 5-kavşak yapısına göre:
#   0 — 1 — 2
#       |   |
#       3 — 4
_GREEN_WAVE_EDGES: List[Tuple[int, int]] = [(0, 1), (1, 2), (1, 3), (3, 4)]

# Kaç araç birikirse "platooon" sayılır?
_PLATOON_THRESHOLD: int = 20


@dataclass
class RewardWeights:
    """Ödül bileşeni ağırlıkları — hiperparametre olarak ayarlanabilir."""
    pressure:       float = -0.30   # yüksek basınç → ceza
    queue:          float = -0.25   # uzun kuyruk → ceza
    throughput:     float =  0.25   # geçiş artışı → ödül
    fairness:       float = -0.10   # yön dengesizliği → ceza
    phase_penalty:  float = -0.08   # gereksiz faz değişimi → ceza
    wait_penalty:   float = -0.05   # çok uzun aynı fazda kalma → ceza
    green_wave:     float =  0.20   # dinamik platoon koordinasyonu → ödül


def _intersection_total(o: Dict) -> int:
    return o["north_count"] + o["south_count"] + o["east_count"] + o["west_count"]


def compute_reward(
    current_obs: List[Dict],
    previous_obs: Optional[List[Dict]],
    previous_actions: Optional[torch.Tensor],
    current_actions: torch.Tensor,
    weights: RewardWeights = RewardWeights(),
) -> torch.Tensor:
    """
    Çok bileşenli ödül fonksiyonu.

    Bileşenler:
        1. Basınç      Σ araç sayısı → minimize
        2. Kuyruk      Σ queue_length → minimize
        3. Throughput  Δ araç azalması → maximize
        4. Adillik     Yön varyansı → minimize
        5. Stabilite   Gereksiz faz değişimi → ceza
        6. Bekleme     Aynı fazda >60sn → ceza
        7. Yeşil Dalga Platoon tespiti + aşağı kavşak hazırlığı → ödül

    Yeşil Dalga (7) nasıl çalışır:
        - Upstream kavşakta ≥ PLATOON_THRESHOLD araç VAR
        - Upstream yeşil (platoon serbest bırakılıyor)
        - Downstream AYNI EKSENDE yeşil (platoon durmadan geçecek)
        → base bonus + eğer downstream kuyruk bu adımda azaldıysa ekstra bonus
    """
    cur = sorted(current_obs, key=lambda d: d["intersection_id"])
    prev = sorted(previous_obs, key=lambda d: d["intersection_id"]) \
        if previous_obs is not None else None

    # ── 1. Basınç ──
    total_vehicles = sum(_intersection_total(o) for o in cur)
    pressure = total_vehicles / max(len(cur) * 40.0, 1.0)

    # ── 2. Kuyruk ──
    total_queue = sum(o["queue_length"] for o in cur)
    queue = total_queue / max(len(cur) * 100.0, 1.0)

    # ── 3. Throughput ──
    throughput = 0.0
    if prev is not None:
        prev_vehicles = sum(_intersection_total(o) for o in prev)
        throughput = (prev_vehicles - total_vehicles) / max(prev_vehicles, 1.0)
        throughput = max(throughput, -1.0)

    # ── 4. Adillik ──
    fairness_scores = []
    for o in cur:
        counts = [o["north_count"], o["south_count"], o["east_count"], o["west_count"]]
        mean_c = sum(counts) / 4.0
        var_c  = sum((c - mean_c) ** 2 for c in counts) / 4.0
        fairness_scores.append(math.sqrt(var_c) / max(mean_c, 1.0))
    fairness = sum(fairness_scores) / len(fairness_scores)

    # ── 5. Faz stabilitesi ──
    phase_changes = 0.0
    if previous_actions is not None:
        phase_changes = (current_actions != previous_actions).float().mean().item()

    # ── 6. Bekleme cezası ──
    wait_penalty = 0.0
    for o in cur:
        if o["phase_duration"] > 60.0:
            wait_penalty += (o["phase_duration"] - 60.0) / 60.0
    wait_penalty /= len(cur)

    # ── 7. Dinamik Yeşil Dalga ──
    # Koşul A — Platoon tespiti + downstream hazırlığı (base bonus):
    #   upstream toplam araç ≥ eşik VE upstream yeşil (0 veya 2)
    #   VE downstream aynı eksende yeşil → platoon durmadan geçer
    #
    # Koşul B — Geçiş teyidi (ekstra bonus):
    #   önceki adıma göre downstream kuyruğu azaldıysa gerçek geçiş oldu
    green_wave_score = 0.0
    for (src_id, dst_id) in _GREEN_WAVE_EDGES:
        src = cur[src_id]
        dst = cur[dst_id]

        src_total   = _intersection_total(src)
        src_phase   = src["current_phase"]
        dst_phase   = dst["current_phase"]

        src_is_green = src_phase in (0, 2)
        dst_aligned  = dst_phase in (0, 2) and (dst_phase % 2 == src_phase % 2)
        has_platoon  = src_total >= _PLATOON_THRESHOLD

        if has_platoon and src_is_green and dst_aligned:
            # Base: ödül platoon büyüklüğüyle orantılı (1.0 eşikte, üstünde daha fazla)
            base = min(src_total / _PLATOON_THRESHOLD, 2.0)
            green_wave_score += base

            # Ekstra: downstream kuyruğu gerçekten azaldıysa
            if prev is not None:
                prev_dst_queue = prev[dst_id]["queue_length"]
                curr_dst_queue = dst["queue_length"]
                if curr_dst_queue < prev_dst_queue:
                    reduction = (prev_dst_queue - curr_dst_queue) / max(prev_dst_queue, 1.0)
                    green_wave_score += reduction

    green_wave_score /= max(len(_GREEN_WAVE_EDGES), 1)

    # ── Toplam ──
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


# ══════════════════════════════════════════════════
#  GAE (Generalized Advantage Estimation)
# ══════════════════════════════════════════════════

def compute_gae(
    rewards: List[torch.Tensor],
    values: List[torch.Tensor],
    next_value: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GAE-Lambda advantage hesaplaması.

    Returns:
        advantages : (T,)   — normalize edilmiş advantage'lar
        returns    : (T,)   — discounted return hedefleri
    """
    advantages = []
    gae = torch.tensor(0.0)

    values_ext = values + [next_value]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_ext[t + 1] - values_ext[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)

    advantages = torch.stack(advantages)
    returns = advantages + torch.stack(values)

    # Advantage normalizasyonu — eğitim stabilitesi
    # Threshold: std çok küçükse (tüm ödüller aynı) normalize etme.
    # Aksi hâlde sıfıra yakın avantajlar N(0,1)'e şişirilir ve sahte gradyanlar oluşur.
    if advantages.numel() > 1 and advantages.std() > 0.01:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
    Bir rollout üzerinde birden fazla PPO epoch'u çalıştırır.

    Args:
        rollout: {
            'observations': List[Tensor],     — (N, F) her zaman adımı
            'edge_index':   Tensor,            — (2, E) sabit topoloji
            'actions':      List[Tensor],      — (N,)
            'log_probs':    List[Tensor],      — (N,)
            'rewards':      List[Tensor],      — skaler
            'values':       List[Tensor],      — skaler
            'hidden_states':List[Tensor],      — (1, N, H)
            'next_value':   Tensor,            — skaler
        }
    """
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
