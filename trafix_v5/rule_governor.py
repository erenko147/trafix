"""
TraFix v5 — Rule Governor
==========================
Constrains the AI's raw logits using traffic-domain rules before sampling.

Designed to be used identically in two places:
  1. stage3_train_ppo.py — during rollout collection (full rules)
                         — inside ppo_update evaluate_actions (stateless only)
  2. backend/main.py     — at live inference (full rules)

Rules
-----
  [Hard, Stateless] Min green time : block phase switch before MIN_GREEN_S seconds
  [Hard, Stateless] Max green time : force phase switch after MAX_GREEN_S seconds
  [Soft, Stateful]  Anti-flicker   : penalise A→B→A rapid reversals per junction
  [Soft, Stateless] Pressure boost : boost logit of most-congested direction

Feature vector layout (10-dim, from parse_sumo_observations in trafix_v2.py):
  [0] north / 30    [1] south / 30    [2] east / 30    [3] west / 30
  [4] queue / 150   [5-8] phase one-hot   [9] duration / 120
"""

from __future__ import annotations

from collections import deque
from typing import List, Tuple

import torch
from torch import Tensor
from torch.distributions import Categorical

# ── Feature index constants ───────────────────────────────────────────────────
_IDX_NORTH    = 0
_IDX_SOUTH    = 1
_IDX_EAST     = 2
_IDX_WEST     = 3
_IDX_PHASE    = slice(5, 9)   # one-hot 4-bit block
_IDX_DURATION = 9

_NORM_COUNT    = 30.0
_NORM_DURATION = 120.0

_NEG_INF = -1e9

# Phase index → which count index represents the green direction
# Phase 0=North, 1=East, 2=South, 3=West
_PHASE_TO_COUNT = {0: 0, 1: 2, 2: 1, 3: 3}  # phase → [n,s,e,w] index
_COUNT_TO_PHASE = {0: 0, 2: 1, 1: 2, 3: 3}  # count index → phase


def _decode_obs(obs_j: Tensor) -> Tuple[int, float, List[float]]:
    """
    Decode one junction's normalised 10-dim obs vector.
    Returns (current_phase 0-3, phase_duration_seconds, [n, s, e, w] raw counts).
    """
    phase    = int(obs_j[_IDX_PHASE].argmax().item())
    duration = float(obs_j[_IDX_DURATION].item()) * _NORM_DURATION
    counts   = [
        float(obs_j[_IDX_NORTH].item()) * _NORM_COUNT,
        float(obs_j[_IDX_SOUTH].item()) * _NORM_COUNT,
        float(obs_j[_IDX_EAST].item())  * _NORM_COUNT,
        float(obs_j[_IDX_WEST].item())  * _NORM_COUNT,
    ]
    return phase, duration, counts


# ─────────────────────────────────────────────────────────────────────────────

class RuleGovernor:
    """
    Applies traffic-law rules to raw model logits via additive masking.

    Stateless rules are safe to replay during PPO evaluate_actions (no memory
    of past decisions needed). Stateful rules track per-episode action history
    and must only be applied during rollout collection.

    Usage (training)
    ----------------
        governor = RuleGovernor()
        for episode in range(N):
            governor.reset()
            ...
            # collection step
            logits_list, value = model.forward(obs_input)
            obs_last = obs_input[0, -1]          # [J, obs_dim]
            masked   = governor.apply(logits_list, obs_last)
            actions, log_probs = sample_from(masked)
            governor.update_state(actions_1d)
            ...
            # ppo_update (inside evaluate_actions replacement)
            masked_batch = governor.apply_stateless_batch(logits_list, obs_last_batch)

    Usage (inference)
    -----------------
        governor = RuleGovernor()           # one instance, lives for session
        ...
        masked = governor.apply(logits_list, obs_last)
        action_probs = softmax(masked)
        governor.update_state(chosen_actions)

    Args:
        num_junctions:    TLS node count (default 5)
        num_phases:       discrete phases per junction (default 4)
        min_green_s:      minimum green-phase hold time in seconds (default 10)
        max_green_s:      maximum green-phase hold time in seconds (default 90)
        flicker_window:   history window for anti-flicker (default 2)
        flicker_penalty:  logit subtracted from reversal candidate (default 3.0)
        pressure_boost:   logit added to most-congested direction (default 1.0)
        pressure_thresh:  fraction of total flow to trigger boost (default 0.40)
    """

    def __init__(
        self,
        num_junctions: int = 5,
        num_phases: int = 4,
        min_green_s: float = 10.0,
        max_green_s: float = 90.0,
        flicker_window: int = 2,
        flicker_penalty: float = 3.0,
        pressure_boost: float = 1.0,
        pressure_thresh: float = 0.40,
    ):
        self.num_junctions  = num_junctions
        self.num_phases     = num_phases
        self.min_green_s    = min_green_s
        self.max_green_s    = max_green_s
        self.flicker_window = flicker_window
        self.flicker_penalty = flicker_penalty
        self.pressure_boost  = pressure_boost
        self.pressure_thresh = pressure_thresh

        self._recent: List[deque] = [
            deque(maxlen=flicker_window) for _ in range(num_junctions)
        ]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self):
        """Clear per-episode action history. Call at the start of each episode."""
        self._recent = [deque(maxlen=self.flicker_window) for _ in range(self.num_junctions)]

    def update_state(self, actions_1d: Tensor):
        """
        Record chosen actions for anti-flicker tracking.
        Call once per step after apply(), before env.step().
        actions_1d: [J] int tensor.
        """
        for j in range(self.num_junctions):
            self._recent[j].append(int(actions_1d[j].item()))

    # ── Per-junction mask builders (work on single obs vector) ───────────────

    def _hard_mask(self, phase: int, duration: float) -> Tensor:
        """
        [num_phases] additive mask with 0 or _NEG_INF entries.
        Derived solely from current_phase and phase_duration — stateless.
        """
        mask = torch.zeros(self.num_phases)

        if duration < self.min_green_s:
            # Must stay on current phase until minimum hold time is met
            for p in range(self.num_phases):
                if p != phase:
                    mask[p] = _NEG_INF

        elif duration > self.max_green_s:
            # Phase has run too long — force a switch
            mask[phase] = _NEG_INF

        return mask

    def _pressure_bonus(self, counts: List[float]) -> Tensor:
        """[num_phases] soft bonus for the most-congested direction (stateless)."""
        bonus = torch.zeros(self.num_phases)
        total = sum(counts)
        if total > 0:
            best_i = int(max(range(4), key=lambda i: counts[i]))
            fraction = counts[best_i] / total
            if fraction > self.pressure_thresh:
                best_phase = _COUNT_TO_PHASE[best_i]
                bonus[best_phase] = self.pressure_boost * fraction
        return bonus

    def _flicker_penalty(self, j: int) -> Tensor:
        """[num_phases] soft penalty discouraging A→B→A reversal (stateful)."""
        penalty = torch.zeros(self.num_phases)
        recent = list(self._recent[j])
        if len(recent) >= 2 and recent[-1] != recent[-2]:
            # Last two decisions were different — penalise reverting to the older one
            reversal_target = recent[-2]
            if 0 <= reversal_target < self.num_phases:
                penalty[reversal_target] = -self.flicker_penalty
        return penalty

    # ── Public API ────────────────────────────────────────────────────────────

    def apply(
        self,
        logits_list: List[Tensor],
        obs_last: Tensor,
    ) -> List[Tensor]:
        """
        Full governor (stateless hard rules + stateful soft rules).
        Use during rollout collection.

        Args:
            logits_list : J-length list, each [1, num_phases]
            obs_last    : [J, obs_dim] — last timestep of the observation window
        Returns:
            Masked logits_list — same structure and device as input.
        """
        out = []
        for j, logits in enumerate(logits_list):
            phase, duration, counts = _decode_obs(obs_last[j])
            additive = (
                self._hard_mask(phase, duration)
                + self._pressure_bonus(counts)
                + self._flicker_penalty(j)
            ).to(logits.device)
            out.append(logits + additive.unsqueeze(0))
        return out

    def apply_stateless(
        self,
        logits_list: List[Tensor],
        obs_last: Tensor,
    ) -> List[Tensor]:
        """
        Stateless-only governor — hard rules + pressure bonus, no flicker tracking.
        Safe to call during PPO evaluate_actions for a single sample.

        Args:
            logits_list : J-length list, each [batch, num_phases]
            obs_last    : [J, obs_dim]
        """
        out = []
        for j, logits in enumerate(logits_list):
            phase, duration, counts = _decode_obs(obs_last[j])
            additive = (
                self._hard_mask(phase, duration) + self._pressure_bonus(counts)
            ).to(logits.device)
            out.append(logits + additive.unsqueeze(0))
        return out

    def apply_stateless_batch(
        self,
        logits_list: List[Tensor],
        obs_last_batch: Tensor,
    ) -> List[Tensor]:
        """
        Stateless-only, batched for use inside ppo_update minibatch evaluation.

        Args:
            logits_list    : J-length list, each [batch, num_phases]
            obs_last_batch : [batch, J, obs_dim] — last timestep per sample
        Returns:
            Masked logits_list, each [batch, num_phases].
        """
        batch = obs_last_batch.shape[0]
        out = []
        for j, logits in enumerate(logits_list):
            rows = []
            for b in range(batch):
                phase, duration, counts = _decode_obs(obs_last_batch[b, j])
                rows.append(
                    self._hard_mask(phase, duration) + self._pressure_bonus(counts)
                )
            additive = torch.stack(rows).to(logits.device)  # [batch, num_phases]
            out.append(logits + additive)
        return out


# ── Sampling helpers (used by training loop) ─────────────────────────────────

def sample_governed(
    masked_logits: List[Tensor],
) -> Tuple[Tensor, Tensor]:
    """
    Sample actions and compute log_probs from a list of masked logits.

    Args:
        masked_logits: J-length list, each [1, num_phases]
    Returns:
        actions   : [1, J]
        log_probs : [1, J]
    """
    actions_list, lp_list = [], []
    for logits in masked_logits:
        dist = Categorical(logits=logits)
        a = dist.sample()
        actions_list.append(a)
        lp_list.append(dist.log_prob(a))
    return torch.stack(actions_list, dim=1), torch.stack(lp_list, dim=1)


def evaluate_governed(
    masked_logits: List[Tensor],
    actions_batch: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Recompute log_probs and entropy for recorded actions under masked distribution.
    For use inside ppo_update in place of model.evaluate_actions.

    Args:
        masked_logits : J-length list, each [batch, num_phases]
        actions_batch : [batch, J]
    Returns:
        log_probs : [batch, J]
        entropy   : [batch, J]
    """
    lp_list, ent_list = [], []
    for j, logits in enumerate(masked_logits):
        dist = Categorical(logits=logits)
        lp_list.append(dist.log_prob(actions_batch[:, j]))
        ent_list.append(dist.entropy())
    return torch.stack(lp_list, dim=1), torch.stack(ent_list, dim=1)
