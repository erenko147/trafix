"""
TraFix — Spatio-Temporal GNN + PPO Agent
=========================================
5 kavşaklı asimetrik ağ için GCN (mekansal) + GRU (zamansal) tabanlı
PPO aktör-kritik mimarisi.

Düzeltmeler (orijinal koda kıyasla):
  • num_node_features = 7  (backend FEATURE_ORDER ile uyumlu)
  • Actor head her kavşak için AYRI aksiyon üretir  (5, num_actions)
  • Critic global mean ile tek skaler value döner
  • GRU hidden state semantiği yorumlarla netleştirildi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    raise ImportError(
        "torch_geometric kurulu değil. Lütfen çalıştır:\n"
        "  pip install torch-geometric"
    )


# ──────────────────────────────────────────────
#  Spatio-Temporal GNN  (GCN + GRU)
# ──────────────────────────────────────────────
class Simple_ST_GNN(nn.Module):
    """
    Mekansal özellikler GCNConv ile komşu kavşaklardan toplanır,
    zamansal bağlam GRU ile korunur.
    """

    def __init__(self, num_node_features: int, hidden_dim: int):
        super().__init__()
        self.gcn = GCNConv(num_node_features, hidden_dim)
        # batch_first=True  →  giriş/çıkış  (batch, seq, feature)
        # Burada "batch" = kavşak sayısı (5), "seq" = 1 zaman adımı
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                hidden_state: torch.Tensor):
        """
        Args:
            x           : (num_nodes, num_node_features)  — kavşak özellikleri
            edge_index  : (2, num_edges)                  — graf topolojisi
            hidden_state: (1, num_nodes, hidden_dim)      — GRU önceki gizli durum
                          num_layers=1 olduğu için ilk boyut 1.
                          Her kavşak bağımsız bir "batch öğesi" olarak işlenir.
        Returns:
            out             : (num_nodes, hidden_dim)
            new_hidden_state: (1, num_nodes, hidden_dim)
        """
        # 1. Mekansal — komşu kavşak mesajları
        x = F.relu(self.gcn(x, edge_index))  # (num_nodes, hidden_dim)

        # 2. Zamansal — GRU her kavşağı bağımsız bir "batch" olarak işler
        #    GRU girişi: (batch=num_nodes, seq_len=1, features=hidden_dim)
        x = x.unsqueeze(1)  # (num_nodes, 1, hidden_dim)

        # hidden_state'i GRU beklentisine çevir: (num_layers, batch, hidden)
        # Gelen: (1, num_nodes, hidden_dim) — zaten doğru formatta
        out, new_hidden_state = self.gru(x, hidden_state)

        return out.squeeze(1), new_hidden_state  # (num_nodes, hidden_dim)


# ──────────────────────────────────────────────
#  PPO Aktör-Kritik Ajan
# ──────────────────────────────────────────────
class Core_PPO_Agent(nn.Module):
    """
    • Actor  → her kavşak için ayrı aksiyon olasılıkları  (num_nodes, num_actions)
    • Critic → tüm ağ için tek skaler değer               (1,)
    """

    def __init__(self, num_node_features: int, hidden_dim: int,
                 num_actions: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.st_gnn = Simple_ST_GNN(num_node_features, hidden_dim)

        # Actor: her kavşağa uygulanır → (num_nodes, num_actions)
        self.actor = nn.Linear(hidden_dim, num_actions)

        # Critic: global özetten tek value → (1,)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                hidden_state: torch.Tensor):
        """
        Returns:
            action_probs : (num_nodes, num_actions) — her kavşak için faz olasılıkları
            state_value  : (1,)                     — ağın genel değer tahmini
            hidden_state : (1, num_nodes, hidden_dim)
        """
        features, new_hidden = self.st_gnn(x, edge_index, hidden_state)
        # features: (num_nodes, hidden_dim)

        # Actor — her kavşağa ayrı softmax
        action_probs = F.softmax(self.actor(features), dim=-1)
        # (num_nodes, num_actions)

        # Critic — tüm kavşakların ortalamasından tek değer
        global_state = torch.mean(features, dim=0)  # (hidden_dim,)
        state_value = self.critic(global_state)      # (1,)

        return action_probs, state_value, new_hidden

    def init_hidden(self, num_nodes: int = 5) -> torch.Tensor:
        """Sıfır başlangıç hidden state oluşturur."""
        return torch.zeros(1, num_nodes, self.hidden_dim)
