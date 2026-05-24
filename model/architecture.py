"""
TraFix Model v6
================
Temporal (GRU) + Graph (GATConv) actor-critic for 5-junction traffic control.

Changes from v5:
  OBS_DIM  = 20  (was 10) — 12 per-lane counts + queue + 6-phase one-hot + duration
  NUM_PHASES = 6  (was 4) — NS-through, N-left, S-left, EW-through, E-left, W-left

Architecture:
  1. GRU temporal encoder  — hidden_dim=128
  2. GATConv graph encoder — heads=4, out=32 each → 128 total
  3. Shared MLP trunk      — Linear(128→128)→ReLU→Linear(128→64)→ReLU
  4. Actor heads           — 5 × Linear(64→6), raw logits
  5. Local critic heads    — 5 × Linear(64→1), per-junction value
  6. Global critic head    — Linear(64→1) on mean-pooled trunk output
     V_j = V_local_j + V_global  →  per-junction values [batch, J]
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical
from torch_geometric.nn import GATConv


# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────

OBS_DIM    = 20
NUM_PHASES = 6
NUM_JUNCTIONS = 5


def _make_chain_edge_index(n: int) -> Tensor:
    """Bidirectional chain edges: 0-1-2-..-(n-1)."""
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    return torch.tensor([src, dst], dtype=torch.long)


# ──────────────────────────────────────────────
#  Sub-modules
# ──────────────────────────────────────────────

class _TemporalEncoder(nn.Module):
    """Shared GRU across all junctions; processes them in a single batch."""

    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size=obs_dim, hidden_size=hidden_dim,
                          num_layers=1, batch_first=True)

    def forward(self, obs: Tensor) -> Tensor:
        """obs: [batch, T, J, obs_dim] → [batch, J, hidden_dim]"""
        B, T, J, D = obs.shape
        x = obs.permute(0, 2, 1, 3).reshape(B * J, T, D)
        _, h_n = self.gru(x)
        return h_n.squeeze(0).reshape(B, J, self.hidden_dim)


class _GraphEncoder(nn.Module):
    """GATConv over a fixed 5-node chain graph."""

    def __init__(self, in_channels: int, heads: int, head_dim: int):
        super().__init__()
        self.gat = GATConv(in_channels=in_channels, out_channels=head_dim,
                           heads=heads, concat=True)
        self.out_dim = heads * head_dim

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.gat(x, edge_index)


class _SharedTrunk(nn.Module):
    def __init__(self, in_dim: int, mid_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, out_dim), nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ──────────────────────────────────────────────
#  Main Model
# ──────────────────────────────────────────────

class TraFixV6(nn.Module):
    """
    Temporal-Graph actor-critic for 5-junction PPO (v6).

    Args:
        obs_dim:      number of features per junction per timestep (default 20)
        num_phases:   discrete phases per junction (default 6)
        hidden_dim:   GRU hidden size (default 128)
        gat_heads:    number of GAT attention heads (default 4)
        gat_head_dim: output channels per GAT head (default 32 → 128 total)
        trunk_mid:    MLP hidden size (default 128)
        trunk_out:    MLP output size (default 64)
    """

    _GAT_IN = 128   # must equal hidden_dim

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        num_phases: int = NUM_PHASES,
        hidden_dim: int = 128,
        gat_heads: int = 4,
        gat_head_dim: int = 32,
        trunk_mid: int = 128,
        trunk_out: int = 64,
    ):
        super().__init__()

        self.num_phases = num_phases
        self.hidden_dim = hidden_dim
        self.trunk_out = trunk_out

        self.temporal_enc = _TemporalEncoder(obs_dim=obs_dim, hidden_dim=hidden_dim)
        self.graph_enc = _GraphEncoder(in_channels=hidden_dim, heads=gat_heads,
                                       head_dim=gat_head_dim)
        gat_out = gat_heads * gat_head_dim  # 128

        self.trunk = _SharedTrunk(in_dim=gat_out, mid_dim=trunk_mid, out_dim=trunk_out)

        # One actor head per junction, outputting 6 logits
        self.actor_heads = nn.ModuleList([
            nn.Linear(trunk_out, num_phases) for _ in range(NUM_JUNCTIONS)
        ])
        # Hybrid critic: per-junction local + one global network-wide head
        self.local_critics = nn.ModuleList([
            nn.Linear(trunk_out, 1) for _ in range(NUM_JUNCTIONS)
        ])
        self.global_critic = nn.Linear(trunk_out, 1)

        self.register_buffer("edge_index", _make_chain_edge_index(NUM_JUNCTIONS))

    def _encode(self, obs: Tensor, edge_index: Tensor) -> tuple:
        B = obs.shape[0]
        h = self.temporal_enc(obs)

        h_flat = h.reshape(B * NUM_JUNCTIONS, self.hidden_dim)
        ei = self._batch_edge_index(edge_index, B)
        g = self.graph_enc(h_flat, ei)
        g = g.reshape(B, NUM_JUNCTIONS, -1)

        t = self.trunk(g)                                           # [B, J, trunk_out]
        local_v  = torch.stack(
            [self.local_critics[j](t[:, j, :]) for j in range(NUM_JUNCTIONS)], dim=1
        )                                                           # [B, J, 1]
        global_v = self.global_critic(t.mean(dim=1)).unsqueeze(1)  # [B, 1, 1]
        v = (local_v + global_v).squeeze(-1)                       # [B, J]
        return t, v

    def _batch_edge_index(self, edge_index: Tensor, batch_size: int) -> Tensor:
        if batch_size == 1:
            return edge_index
        offsets = torch.arange(batch_size, device=edge_index.device) * NUM_JUNCTIONS
        return torch.cat([edge_index + off for off in offsets], dim=1)

    def forward(self, obs: Tensor, edge_index: Tensor = None) -> tuple:
        """
        obs: [batch, T, J, obs_dim]
        Returns: (logits_list, value)
          logits_list: list of 5 tensors each [batch, 6]
          value:       [batch, J]  — per-junction (local + global)
        """
        if edge_index is None:
            edge_index = self.edge_index
        trunk, value = self._encode(obs, edge_index)
        logits = [self.actor_heads[j](trunk[:, j, :]) for j in range(NUM_JUNCTIONS)]
        return logits, value

    def get_action(self, obs: Tensor) -> tuple:
        """Sample actions. Returns (actions [B,J], log_probs [B,J], value [B,1])."""
        logits_list, value = self.forward(obs)
        actions_list, lp_list = [], []
        for logits in logits_list:
            dist = Categorical(logits=logits)
            a = dist.sample()
            actions_list.append(a)
            lp_list.append(dist.log_prob(a))
        return torch.stack(actions_list, dim=1), torch.stack(lp_list, dim=1), value

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> tuple:
        """Evaluate log_probs and entropy for given actions (PPO update).
        Returns: (log_probs [B,J], entropy [B,J], value [B,1])."""
        logits_list, value = self.forward(obs)
        lp_list, ent_list = [], []
        for j, logits in enumerate(logits_list):
            dist = Categorical(logits=logits)
            lp_list.append(dist.log_prob(actions[:, j]))
            ent_list.append(dist.entropy())
        return torch.stack(lp_list, dim=1), torch.stack(ent_list, dim=1), value

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"TraFixV6(\n"
            f"  temporal_enc : {self.temporal_enc.gru}\n"
            f"  graph_enc    : {self.graph_enc.gat}\n"
            f"  trunk        : {self.trunk.net}\n"
            f"  actor_heads  : 5 x Linear({self.trunk_out}, {self.num_phases})\n"
            f"  local_critics: 5 x Linear({self.trunk_out}, 1)\n"
            f"  global_critic: Linear({self.trunk_out}, 1)\n"
            f"  Total params : {total:,}  (trainable: {trainable:,})\n"
            f")"
        )


# ──────────────────────────────────────────────
#  Smoke test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    BATCH, T, J = 2, 5, NUM_JUNCTIONS

    model = TraFixV6()
    print(model)

    obs = torch.randn(BATCH, T, J, OBS_DIM)
    logits_list, value = model(obs)

    print("\nforward() output shapes:")
    for i, l in enumerate(logits_list):
        print(f"  logits[{i}]: {list(l.shape)}")
        assert l.shape == (BATCH, NUM_PHASES)
    assert value.shape == (BATCH, NUM_JUNCTIONS)

    actions, lp, v2 = model.get_action(obs)
    assert actions.shape == (BATCH, J)
    assert lp.shape == (BATCH, J)

    new_lp, ent, v3 = model.evaluate_actions(obs, actions)
    assert new_lp.shape == (BATCH, J)
    assert ent.shape == (BATCH, J)

    print("All assertions passed.")
