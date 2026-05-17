"""
TraFix v6 — Rule Governor
==========================
Constrains AI logits using traffic-domain rules before sampling.

Updated for 6-phase model:
  Phase 0: NS-through   Phase 3: EW-through
  Phase 1: N-left       Phase 4: E-left
  Phase 2: S-left       Phase 5: W-left

Feature vector layout (20-dim, from parse_sumo_observations in trafix_v2.py):
  [0]  north_left/15   [1]  north_through/30  [2]  north_right/15
  [3]  south_left/15   [4]  south_through/30  [5]  south_right/15
  [6]  east_left/15    [7]  east_through/30   [8]  east_right/15
  [9]  west_left/15    [10] west_through/30   [11] west_right/15
  [12] queue/200       [13-18] phase one-hot (6 bits)   [19] min(duration/120, 3.0)
"""

from __future__ import annotations

from collections import deque
from typing import List, Tuple

import torch
from torch import Tensor
from torch.distributions import Categorical

# ── Feature index constants ───────────────────────────────────────────────────
_IDX_N_LEFT    = 0
_IDX_N_THROUGH = 1
_IDX_N_RIGHT   = 2
_IDX_S_LEFT    = 3
_IDX_S_THROUGH = 4
_IDX_S_RIGHT   = 5
_IDX_E_LEFT    = 6
_IDX_E_THROUGH = 7
_IDX_E_RIGHT   = 8
_IDX_W_LEFT    = 9
_IDX_W_THROUGH = 10
_IDX_W_RIGHT   = 11
_IDX_QUEUE     = 12
_IDX_PHASE     = slice(13, 19)   # 6-bit one-hot block
_IDX_DURATION  = 19

_NORM_LEFT_RIGHT = 15.0
_NORM_THROUGH    = 30.0
_NORM_DURATION   = 120.0

_NEG_INF = -1e9

# Min/max green times by phase type
MIN_GREEN_THROUGH = 10.0   # seconds for NS-through (0) and EW-through (3)
MIN_GREEN_LEFT    = 8.0    # seconds for left-turn phases (1, 2, 4, 5)
MAX_GREEN_THROUGH = 90.0
MAX_GREEN_LEFT    = 45.0


def _decode_obs(obs_j: Tensor) -> Tuple[int, float]:
    """Returns (current_phase 0-5, phase_duration_seconds)."""
    phase    = int(obs_j[_IDX_PHASE].argmax().item())
    duration = float(obs_j[_IDX_DURATION].item()) * _NORM_DURATION
    return phase, duration


# ─────────────────────────────────────────────────────────────────────────────

class RuleGovernor:
    """
    Applies traffic-law rules to raw model logits via additive masking.

    Rules:
      [Hard, Stateless] Min green time : block switch before min hold
      [Hard, Stateless] Max green time : force switch after max hold
      [Soft, Stateful]  Anti-flicker   : penalise A→B→A reversals
      [Soft, Stateless] Pressure boost : boost most-congested phase

    Args:
        num_junctions:    TLS node count (default 5)
        num_phases:       discrete phases (default 6)
        min_green_s:      base minimum green (through phases — left uses MIN_GREEN_LEFT)
        max_green_s:      base maximum green (through phases — left uses MAX_GREEN_LEFT)
        flicker_window:   history window for anti-flicker (default 2)
        flicker_penalty:  logit subtracted from reversal candidate (default 3.0)
        pressure_boost:   logit added to most-congested phase (default 1.0)
        pressure_thresh:  fraction of total flow to trigger boost (default 0.35)
    """

    def __init__(
        self,
        num_junctions: int = 5,
        num_phases: int = 6,
        min_green_s: float = 10.0,
        max_green_s: float = 90.0,
        flicker_window: int = 2,
        flicker_penalty: float = 3.0,
        pressure_boost: float = 1.0,
        pressure_thresh: float = 0.35,
    ):
        self.num_junctions   = num_junctions
        self.num_phases      = num_phases
        self.min_green_s     = min_green_s
        self.max_green_s     = max_green_s
        self.flicker_window  = flicker_window
        self.flicker_penalty = flicker_penalty
        self.pressure_boost  = pressure_boost
        self.pressure_thresh = pressure_thresh

        self._recent: List[deque] = [
            deque(maxlen=flicker_window) for _ in range(num_junctions)
        ]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self):
        """Clear per-episode action history."""
        self._recent = [deque(maxlen=self.flicker_window) for _ in range(self.num_junctions)]

    def update_state(self, actions_1d: Tensor):
        """Record chosen actions for anti-flicker. actions_1d: [J] int tensor."""
        for j in range(self.num_junctions):
            self._recent[j].append(int(actions_1d[j].item()))

    # ── Per-junction mask builders ────────────────────────────────────────────

    def _hard_mask(self, phase: int, duration: float) -> Tensor:
        """[num_phases] additive mask with 0 or _NEG_INF entries (stateless)."""
        mask = torch.zeros(self.num_phases)

        is_through = phase in (0, 3)
        min_green = MIN_GREEN_THROUGH if is_through else MIN_GREEN_LEFT
        max_green = MAX_GREEN_THROUGH if is_through else MAX_GREEN_LEFT

        if duration < min_green:
            for p in range(self.num_phases):
                if p != phase:
                    mask[p] = _NEG_INF
        elif duration > max_green:
            mask[phase] = _NEG_INF

        return mask

    def _pressure_bonus(self, obs_j: Tensor) -> Tensor:
        """
        [num_phases] soft bonus for the most-congested movement group.
        Phase 0: N-through + S-through
        Phase 1: N-left
        Phase 2: S-left
        Phase 3: E-through + W-through
        Phase 4: E-left
        Phase 5: W-left
        """
        bonus = torch.zeros(self.num_phases)

        demands = [
            obs_j[_IDX_N_THROUGH] * _NORM_THROUGH + obs_j[_IDX_S_THROUGH] * _NORM_THROUGH,
            obs_j[_IDX_N_LEFT]    * _NORM_LEFT_RIGHT,
            obs_j[_IDX_S_LEFT]    * _NORM_LEFT_RIGHT,
            obs_j[_IDX_E_THROUGH] * _NORM_THROUGH + obs_j[_IDX_W_THROUGH] * _NORM_THROUGH,
            obs_j[_IDX_E_LEFT]    * _NORM_LEFT_RIGHT,
            obs_j[_IDX_W_LEFT]    * _NORM_LEFT_RIGHT,
        ]

        total = sum(d.item() for d in demands)
        if total > 0:
            best_phase = int(max(range(6), key=lambda i: demands[i].item()))
            fraction = demands[best_phase].item() / total
            if fraction > self.pressure_thresh:
                bonus[best_phase] = self.pressure_boost * fraction

        return bonus

    def _flicker_penalty(self, j: int) -> Tensor:
        """[num_phases] soft penalty discouraging A→B→A reversal (stateful)."""
        penalty = torch.zeros(self.num_phases)
        recent = list(self._recent[j])
        if len(recent) >= 2 and recent[-1] != recent[-2]:
            reversal_target = recent[-2]
            if 0 <= reversal_target < self.num_phases:
                penalty[reversal_target] = -self.flicker_penalty
        return penalty

    # ── Public API ────────────────────────────────────────────────────────────

    def apply(self, logits_list: List[Tensor], obs_last: Tensor) -> List[Tensor]:
        """
        Full governor (stateless hard + stateful soft rules).
        Use during rollout collection.
        logits_list: J-length list, each [1, num_phases]
        obs_last:    [J, obs_dim]
        """
        out = []
        for j, logits in enumerate(logits_list):
            phase, duration = _decode_obs(obs_last[j])
            additive = (
                self._hard_mask(phase, duration)
                + self._pressure_bonus(obs_last[j])
                + self._flicker_penalty(j)
            ).to(logits.device)
            out.append(logits + additive.unsqueeze(0))
        return out

    def apply_stateless(self, logits_list: List[Tensor], obs_last: Tensor) -> List[Tensor]:
        """
        Stateless-only governor — hard rules + pressure bonus.
        Safe for PPO evaluate_actions (no flicker tracking).
        logits_list: J-length list, each [batch, num_phases]
        obs_last:    [J, obs_dim]
        """
        out = []
        for j, logits in enumerate(logits_list):
            phase, duration = _decode_obs(obs_last[j])
            additive = (
                self._hard_mask(phase, duration) + self._pressure_bonus(obs_last[j])
            ).to(logits.device)
            out.append(logits + additive.unsqueeze(0))
        return out

    def apply_stateless_batch(
        self, logits_list: List[Tensor], obs_last_batch: Tensor
    ) -> List[Tensor]:
        """
        Stateless-only, batched for PPO minibatch evaluation.
        logits_list:    J-length list, each [batch, num_phases]
        obs_last_batch: [batch, J, obs_dim]
        """
        batch = obs_last_batch.shape[0]
        out = []
        for j, logits in enumerate(logits_list):
            rows = []
            for b in range(batch):
                phase, duration = _decode_obs(obs_last_batch[b, j])
                rows.append(
                    self._hard_mask(phase, duration) + self._pressure_bonus(obs_last_batch[b, j])
                )
            additive = torch.stack(rows).to(logits.device)
            out.append(logits + additive)
        return out


# ── Sampling helpers ──────────────────────────────────────────────────────────

def sample_governed(masked_logits: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """
    Sample actions from masked logits.
    masked_logits: J-length list, each [1, num_phases]
    Returns: actions [1,J], log_probs [1,J]
    """
    actions_list, lp_list = [], []
    for logits in masked_logits:
        dist = Categorical(logits=logits)
        a = dist.sample()
        actions_list.append(a)
        lp_list.append(dist.log_prob(a))
    return torch.stack(actions_list, dim=1), torch.stack(lp_list, dim=1)


def evaluate_governed(
    masked_logits: List[Tensor], actions_batch: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Recompute log_probs and entropy for recorded actions under masked distribution.
    masked_logits: J-length list, each [batch, num_phases]
    actions_batch: [batch, J]
    Returns: log_probs [batch,J], entropy [batch,J]
    """
    lp_list, ent_list = [], []
    for j, logits in enumerate(masked_logits):
        dist = Categorical(logits=logits)
        lp_list.append(dist.log_prob(actions_batch[:, j]))
        ent_list.append(dist.entropy())
    return torch.stack(lp_list, dim=1), torch.stack(ent_list, dim=1)
