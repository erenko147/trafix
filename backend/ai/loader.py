"""
Model loader — resolves weight path and initialises TraFixV6 + RuleGovernor.
Called once during API startup; writes results into backend.api.state.
"""

import os
import torch

from model.architecture import TraFixV6
from model.rule_governor import RuleGovernor
import backend.api.state as state


def load_model() -> bool:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    weight_paths = [
        os.path.join(project_root, state._WEIGHT_FILENAME),
        state._WEIGHT_FILENAME,
    ]

    agent = TraFixV6(obs_dim=state.NUM_FEATURES, num_phases=state.NUM_ACTIONS)

    for path in weight_paths:
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            continue
        try:
            ckpt       = torch.load(abs_path, map_location="cpu", weights_only=True)
            model_state = ckpt.get("model_state_dict", ckpt)
            agent.load_state_dict(model_state)
            agent.eval()

            state.ai_agent   = agent
            state._v6_governor = RuleGovernor(
                num_junctions=state.NUM_NODES,
                num_phases=6,
                min_green_s=10.0,
                max_green_s=90.0,
                flicker_window=2,
                flicker_penalty=3.0,
                pressure_boost=1.0,
                pressure_thresh=0.35,
            )
            print(f"[OK] TraFixV6 loaded: {abs_path}")
            print(f"[OK] RuleGovernor active (6 phases, min_green_through=10s)")
            return True
        except RuntimeError as e:
            print(f"[WARN] v6 weight mismatch: {abs_path} — {e}")
            continue

    print("[WARN] TraFixV6 weights not found. Heuristic fallback active.")
    return False
