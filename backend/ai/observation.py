"""
SUMO observation parsing utilities.

Converts raw junction observation dicts (as sent by the SUMO runner or the
live API) into the normalised [J, 20] feature tensor consumed by TraFixV6.

Feature layout (20-dim):
  [0-11]  12 per-lane counts (normalised)
  [12]    total queue / 200
  [13-18] 6-bit current-phase one-hot
  [19]    phase duration / 120  (capped at 3.0)
"""

from typing import Dict, List

import torch

# ── Lane keys (fixed order → indices 0-11) ───────────────────────────────────

LANE_KEYS = [
    "north_left", "north_through", "north_right",
    "south_left", "south_through", "south_right",
    "east_left",  "east_through",  "east_right",
    "west_left",  "west_through",  "west_right",
]

_NS_KEYS = ["north_left", "north_through", "north_right",
            "south_left", "south_through", "south_right"]
_EW_KEYS = ["east_left",  "east_through",  "east_right",
            "west_left",  "west_through",  "west_right"]

# Per-lane normalisers: left/right = 15 veh, through = 30 veh
_NORM: Dict[str, float] = {
    "north_left":    15.0, "north_through": 30.0, "north_right":  15.0,
    "south_left":    15.0, "south_through": 30.0, "south_right":  15.0,
    "east_left":     15.0, "east_through":  30.0, "east_right":   15.0,
    "west_left":     15.0, "west_through":  30.0, "west_right":   15.0,
}

NUM_NODE_FEATURES = 20


def parse_sumo_observations(
    obs_list: List[Dict],
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Convert a list of junction observation dicts to a [J, 20] normalised tensor.

    Each dict must contain: intersection_id, the 12 per-lane keys, queue_length,
    current_phase (model-space 0-5), and phase_duration.
    """
    rows = []
    for o in sorted(obs_list, key=lambda x: x["intersection_id"]):
        row = []

        for key in LANE_KEYS:
            row.append(o.get(key, 0) / _NORM[key])

        row.append(o.get("queue_length", 0.0) / 200.0)

        one_hot = [0.0] * 6
        one_hot[int(o.get("current_phase", 0)) % 6] = 1.0
        row.extend(one_hot)

        row.append(min(o.get("phase_duration", 0.0) / 120.0, 3.0))

        rows.append(row)

    return torch.tensor(rows, dtype=torch.float32, device=device)
