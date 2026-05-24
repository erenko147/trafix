"""
Coordinated multi-intersection PPO agent (v2 architecture).

Used by the training pipeline (backend/training/) — NOT by the live inference
server, which uses TraFixV6 from model/architecture.py instead.

Architecture:
  SpatioTemporalGNN  (2-layer GCN + residual + LayerNorm)
  IntersectionCoordinator  (Multi-Head Attention, transformer-style)
  Actor  5 × Linear(hidden//2 → 4)
  Critic 1 × Linear(hidden//2 → 1)
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    raise ImportError(
        "torch_geometric not found. Install with:\n"
        "  pip install torch-geometric"
    )

from backend.ai.observation import NUM_NODE_FEATURES


class SpatioTemporalGNN(nn.Module):
    """2-layer GCN with residual connection and LayerNorm."""

    def __init__(self, num_node_features: int, hidden_dim: int):
        super().__init__()
        self.gcn1       = GCNConv(num_node_features, hidden_dim)
        self.gcn2       = GCNConv(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gcn1(x, edge_index))
        h = h + F.relu(self.gcn2(h, edge_index))
        return self.layer_norm(h)


class IntersectionCoordinator(nn.Module):
    """Multi-Head Attention for cross-intersection coordination."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        x          = node_features.unsqueeze(0)
        attn_out, _ = self.attn(x, x, x)
        x          = self.norm1(x + attn_out)
        x          = self.norm2(x + self.ffn(x))
        return x.squeeze(0)


class CoordinatedPPOAgent(nn.Module):
    """
    Flow: SUMO obs → GCN → Attention → Actor / Critic

    Used for training with the v2 architecture. The production server
    uses TraFixV6 (GRU + GAT) from model/architecture.py.
    """

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
        self.hidden_dim    = hidden_dim
        self.num_actions   = num_actions
        self.entropy_coef  = entropy_coef
        self.value_coef    = value_coef
        self.clip_eps      = clip_eps
        self.max_grad_norm = max_grad_norm

        self.st_gnn      = SpatioTemporalGNN(num_node_features, hidden_dim)
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
        features    = self.st_gnn(x, edge_index)
        coordinated = self.coordinator(features)
        action_probs = F.softmax(self.actor(coordinated), dim=-1)
        state_value  = self.critic(coordinated.mean(dim=0))
        return action_probs, state_value

    @torch.no_grad()
    def select_actions(self, x, edge_index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs, value = self.forward(x, edge_index)
        dists        = Categorical(probs)
        actions      = dists.sample()
        log_probs    = dists.log_prob(actions)
        return actions, log_probs, value

    def compute_ppo_loss(
        self, x, edge_index, old_actions, old_log_probs, advantages, returns
    ) -> Dict[str, torch.Tensor]:
        probs, value  = self.forward(x, edge_index)
        dists         = Categorical(probs)
        new_log_probs = dists.log_prob(old_actions)
        entropy       = dists.entropy().mean()
        ratio         = torch.exp(new_log_probs - old_log_probs)
        surr1         = ratio * advantages
        surr2         = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss   = -torch.min(surr1, surr2).mean()
        value_loss    = F.mse_loss(value.squeeze(), returns.mean())
        total_loss    = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        return {
            "total":   total_loss,
            "policy":  policy_loss.detach(),
            "value":   value_loss.detach(),
            "entropy": entropy.detach(),
        }
