"""
TraFix Model v5
================
Temporal (GRU) + Graph (GATConv) actor-critic for 5-junction traffic control.

Architecture:
  1. GRU temporal encoder  — per junction, hidden_dim=128, batched via reshape
  2. GATConv graph encoder — heads=4, out=32 each → 128 total, chain 0-1-2-3-4
  3. Shared MLP trunk      — Linear(128→128)→ReLU→Linear(128→64)→ReLU
  4. Actor heads           — 5 × Linear(64→num_phases), raw logits
  5. Critic head           — Linear(64→1) on mean-pooled trunk output
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical
from torch_geometric.nn import GATConv


# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────

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
    """One GRU shared across all junctions; processes them in a single batch."""

    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size=obs_dim, hidden_size=hidden_dim,
                          num_layers=1, batch_first=True)

    def forward(self, obs: Tensor) -> Tensor:
        """
        obs: [batch, T, num_junctions, obs_dim]
        returns: [batch, num_junctions, hidden_dim]
        """
        B, T, J, D = obs.shape
        # Merge batch and junction dims → single GRU forward
        x = obs.permute(0, 2, 1, 3).reshape(B * J, T, D)   # [B*J, T, D]
        _, h_n = self.gru(x)                                  # h_n: [1, B*J, H]
        return h_n.squeeze(0).reshape(B, J, self.hidden_dim)  # [B, J, H]


class _GraphEncoder(nn.Module):
    """GATConv over a fixed 5-node chain graph."""

    def __init__(self, in_channels: int, heads: int, head_dim: int):
        super().__init__()
        self.gat = GATConv(in_channels=in_channels,
                           out_channels=head_dim,
                           heads=heads,
                           concat=True)
        self.out_dim = heads * head_dim

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        x: [N, in_channels] where N = batch * num_junctions (called per sample)
        edge_index: [2, E]
        returns: [N, out_dim]
        """
        return self.gat(x, edge_index)


class _SharedTrunk(nn.Module):
    def __init__(self, in_dim: int, mid_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ──────────────────────────────────────────────
#  Main Model
# ──────────────────────────────────────────────

class TraFixV5(nn.Module):
    """
    Temporal-Graph actor-critic for 5-junction PPO.

    Args:
        obs_dim:     number of features per junction per timestep
        num_phases:  number of discrete phases per junction (default 4)
        hidden_dim:  GRU hidden size (default 128)
        gat_heads:   number of GAT attention heads (default 4)
        gat_head_dim: output channels per GAT head (default 32 → 128 total)
        trunk_mid:   MLP hidden size (default 128)
        trunk_out:   MLP output size passed to actor/critic (default 64)
    """

    _GAT_IN = 128   # must equal hidden_dim; kept as internal constant

    def __init__(
        self,
        obs_dim: int,
        num_phases: int = 4,
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

        # ── Temporal encoder ──
        self.temporal_enc = _TemporalEncoder(obs_dim=obs_dim, hidden_dim=hidden_dim)

        # ── Graph encoder ──
        self.graph_enc = _GraphEncoder(
            in_channels=hidden_dim,
            heads=gat_heads,
            head_dim=gat_head_dim,
        )
        gat_out = gat_heads * gat_head_dim  # 128 by default

        # ── Shared MLP trunk ──
        self.trunk = _SharedTrunk(in_dim=gat_out, mid_dim=trunk_mid, out_dim=trunk_out)

        # ── Actor heads (one per junction) ──
        self.actor_heads = nn.ModuleList([
            nn.Linear(trunk_out, num_phases) for _ in range(NUM_JUNCTIONS)
        ])

        # ── Critic head ──
        self.critic_head = nn.Linear(trunk_out, 1)

        # ── Fixed chain edge_index as buffer ──
        self.register_buffer("edge_index", _make_chain_edge_index(NUM_JUNCTIONS))

    # ── Helpers ───────────────────────────────

    def _encode(self, obs: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """
        Returns trunk embeddings and value.
        obs: [batch, T, num_junctions, obs_dim]
        edge_index: [2, E]
        returns: trunk [batch, num_junctions, trunk_out], value [batch, 1]
        """
        B = obs.shape[0]

        # Temporal: [B, J, hidden_dim]
        h = self.temporal_enc(obs)

        # Graph: reshape to [B*J, hidden_dim], apply GAT, reshape back
        h_flat = h.reshape(B * NUM_JUNCTIONS, self.hidden_dim)
        # Repeat edge_index for each sample in the batch with offset
        ei = self._batch_edge_index(edge_index, B)
        g = self.graph_enc(h_flat, ei)                          # [B*J, gat_out]
        g = g.reshape(B, NUM_JUNCTIONS, -1)                     # [B, J, gat_out]

        # Trunk: [B, J, trunk_out]
        t = self.trunk(g)

        # Value: mean-pool over junctions → [B, 1]
        v = self.critic_head(t.mean(dim=1))

        return t, v

    def _batch_edge_index(self, edge_index: Tensor, batch_size: int) -> Tensor:
        """Replicate edge_index for batch_size graphs with node offsets."""
        if batch_size == 1:
            return edge_index
        offsets = torch.arange(batch_size, device=edge_index.device) * NUM_JUNCTIONS
        ei_list = [edge_index + off for off in offsets]
        return torch.cat(ei_list, dim=1)

    # ── Public API ───────────────────────────

    def forward(
        self,
        obs: Tensor,
        edge_index: Tensor = None,
    ) -> tuple[list[Tensor], Tensor]:
        """
        Args:
            obs:        [batch, T, num_junctions, obs_dim]
            edge_index: optional override; uses internal chain buffer if None
        Returns:
            (logits_list, value)
            logits_list: list of 5 tensors each [batch, num_phases]
            value:       [batch, 1]
        """
        if edge_index is None:
            edge_index = self.edge_index

        trunk, value = self._encode(obs, edge_index)

        logits = [self.actor_heads[j](trunk[:, j, :]) for j in range(NUM_JUNCTIONS)]
        return logits, value

    def get_action(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Sample actions from the policy.

        Args:
            obs: [batch, T, num_junctions, obs_dim]
        Returns:
            actions:   [batch, num_junctions]
            log_probs: [batch, num_junctions]
            value:     [batch, 1]
        """
        logits_list, value = self.forward(obs)

        actions_list = []
        log_probs_list = []
        for logits in logits_list:
            dist = Categorical(logits=logits)
            a = dist.sample()
            actions_list.append(a)
            log_probs_list.append(dist.log_prob(a))

        actions = torch.stack(actions_list, dim=1)       # [B, J]
        log_probs = torch.stack(log_probs_list, dim=1)   # [B, J]
        return actions, log_probs, value

    def evaluate_actions(
        self,
        obs: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Evaluate log-probs and entropy for given actions (used in PPO update).

        Args:
            obs:     [batch, T, num_junctions, obs_dim]
            actions: [batch, num_junctions]  (integer phase indices)
        Returns:
            log_probs: [batch, num_junctions]
            entropy:   [batch, num_junctions]
            value:     [batch, 1]
        """
        logits_list, value = self.forward(obs)

        log_probs_list = []
        entropy_list = []
        for j, logits in enumerate(logits_list):
            dist = Categorical(logits=logits)
            log_probs_list.append(dist.log_prob(actions[:, j]))
            entropy_list.append(dist.entropy())

        log_probs = torch.stack(log_probs_list, dim=1)  # [B, J]
        entropy = torch.stack(entropy_list, dim=1)      # [B, J]
        return log_probs, entropy, value

    # ── Repr ─────────────────────────────────

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lines = [
            "TraFixV5(",
            f"  temporal_enc : {self.temporal_enc.gru}",
            f"  graph_enc    : {self.graph_enc.gat}",
            f"  trunk        : {self.trunk.net}",
            f"  actor_heads  : 5 × Linear({self.trunk_out} → {self.num_phases})",
            f"  critic_head  : Linear({self.trunk_out} → 1)",
            f"  Total params : {total:,}  (trainable: {trainable:,})",
            ")",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────
#  Smoke test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    OBS_DIM = 10
    BATCH = 2
    T = 5
    J = NUM_JUNCTIONS
    NUM_PHASES = 4

    model = TraFixV5(obs_dim=OBS_DIM)
    print(model)
    print()

    obs = torch.randn(BATCH, T, J, OBS_DIM)

    # forward
    logits_list, value = model(obs)
    print("forward() output shapes:")
    for i, l in enumerate(logits_list):
        print(f"  logits[{i}]: {list(l.shape)}")
        assert l.shape == (BATCH, NUM_PHASES), f"logits[{i}] shape mismatch"
    print(f"  value:      {list(value.shape)}")
    assert value.shape == (BATCH, 1), "value shape mismatch"

    # get_action
    actions, log_probs, val2 = model.get_action(obs)
    print("\nget_action() output shapes:")
    print(f"  actions:   {list(actions.shape)}")
    print(f"  log_probs: {list(log_probs.shape)}")
    print(f"  value:     {list(val2.shape)}")
    assert actions.shape == (BATCH, J)
    assert log_probs.shape == (BATCH, J)
    assert val2.shape == (BATCH, 1)

    # evaluate_actions
    lp, ent, val3 = model.evaluate_actions(obs, actions)
    print("\nevaluate_actions() output shapes:")
    print(f"  log_probs: {list(lp.shape)}")
    print(f"  entropy:   {list(ent.shape)}")
    print(f"  value:     {list(val3.shape)}")
    assert lp.shape == (BATCH, J)
    assert ent.shape == (BATCH, J)
    assert val3.shape == (BATCH, 1)

    print("\nAll assertions passed.")
