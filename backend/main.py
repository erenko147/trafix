"""
TraFix Backend — FastAPI
========================
Receives telemetry, queries AI model, returns phase decisions.
Supports model versions: v2, v3, simple, v5, v6.

Set TRAFIX_MODEL_VERSION=v6 to use the 3-lane 6-phase model.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import deque
import torch
import os
import logging
from typing import List, Optional

_MODEL_VERSION = os.environ.get("TRAFIX_MODEL_VERSION", "v2").strip().lower()

_USE_GRAPH = False
_USE_V5    = False
_USE_V6    = False

if _MODEL_VERSION == "v3":
    from backend.ai.trafix_v3 import CoordinatedPPOAgent, parse_sumo_observations
    _WEIGHT_FILENAME = "coordinated_agent_weights_v3.pth"
    _USE_GRAPH = True

elif _MODEL_VERSION == "simple":
    from backend.ai.trafix_simple import SimplePPOAgent as CoordinatedPPOAgent, parse_sumo_observations
    _WEIGHT_FILENAME = "coordinated_agent_weights_simple.pth"

elif _MODEL_VERSION == "v5":
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(__file__))))
    from trafix_v5.trafix_v5 import TraFixV5
    from trafix_v5.rule_governor import RuleGovernor
    from backend.ai.trafix_v2 import parse_sumo_observations
    _WEIGHT_FILENAME = "trafix_v5/checkpoints/trafix_v5_final.pt"
    _USE_V5 = True
    CoordinatedPPOAgent = None

elif _MODEL_VERSION == "v6":
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(__file__))))
    from trafix_v6.trafix_v6 import TraFixV6
    from trafix_v6.rule_governor import RuleGovernor
    from backend.ai.trafix_v2 import parse_sumo_observations
    _WEIGHT_FILENAME = "trafix_v6/checkpoints/trafix_v6_final.pt"
    _USE_V6 = True
    CoordinatedPPOAgent = None

else:  # v2 default
    from backend.ai.trafix_v2 import CoordinatedPPOAgent, parse_sumo_observations
    _WEIGHT_FILENAME = "coordinated_agent_weights.pth"
    _USE_GRAPH = True

logger = logging.getLogger("trafix")

app = FastAPI(title="TraFix API", version="6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────

state_dict = {"0": {}, "1": {}, "2": {}, "3": {}, "4": {}}

# ── Telemetry schema ──────────────────────────────────────────────────────────

class Telemetry(BaseModel):
    intersection_id: int
    # 12 per-lane counts (v6 format)
    north_left:    int = 0
    north_through: int = 0
    north_right:   int = 0
    south_left:    int = 0
    south_through: int = 0
    south_right:   int = 0
    east_left:     int = 0
    east_through:  int = 0
    east_right:    int = 0
    west_left:     int = 0
    west_through:  int = 0
    west_right:    int = 0
    queue_length:  float = 0.0
    current_phase: int = 0
    phase_duration: float = 0.0

    # v2/v3 backward compat fields (optional)
    north_count: Optional[int] = None
    south_count: Optional[int] = None
    east_count:  Optional[int] = None
    west_count:  Optional[int] = None


# ── Model configuration ───────────────────────────────────────────────────────

NUM_FEATURES = 20   # 12 per-lane + queue + 6-phase one-hot + duration
HIDDEN_DIM   = 128
NUM_ACTIONS  = 6    # 6 phases for v6
NUM_NODES    = 5
NUM_HEADS    = 4

# Chain edge_index for 5 junctions
edge_index = torch.tensor([
    [0, 1, 1, 2, 1, 3, 2, 4, 3, 4],
    [1, 0, 2, 1, 3, 1, 4, 2, 4, 3],
], dtype=torch.long)

# ── Model state ───────────────────────────────────────────────────────────────

ai_agent = None
last_decisions_cache: list = []

_V5_T_WINDOW = 10
_v5_window: deque = deque(maxlen=_V5_T_WINDOW)
_v5_governor = None

_V6_T_WINDOW = 30
_v6_window: deque = deque(maxlen=_V6_T_WINDOW)
_v6_governor = None
_last_batch_step: int = -1


def load_model():
    global ai_agent, _v5_governor, _v6_governor

    base_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(base_dir)

    print(f"[INFO] Model version: {_MODEL_VERSION.upper()} | Weight file: {_WEIGHT_FILENAME}")

    # ── v6 ────────────────────────────────────────────────────────────────────
    if _USE_V6:
        weight_paths = [
            os.path.join(project_root, _WEIGHT_FILENAME),
            _WEIGHT_FILENAME,
        ]
        agent = TraFixV6(obs_dim=NUM_FEATURES, num_phases=NUM_ACTIONS)
        for path in weight_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                try:
                    ckpt = torch.load(abs_path, map_location="cpu", weights_only=True)
                    state = ckpt.get("model_state_dict", ckpt)
                    agent.load_state_dict(state)
                    agent.eval()
                    ai_agent = agent
                    _v6_governor = RuleGovernor(
                        num_junctions=NUM_NODES,
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

    # ── v5 ────────────────────────────────────────────────────────────────────
    if _USE_V5:
        weight_paths = [
            os.path.join(project_root, _WEIGHT_FILENAME),
            _WEIGHT_FILENAME,
        ]
        agent = TraFixV5(obs_dim=NUM_FEATURES, num_phases=4)
        for path in weight_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                try:
                    ckpt = torch.load(abs_path, map_location="cpu", weights_only=True)
                    state = ckpt.get("model_state_dict", ckpt)
                    agent.load_state_dict(state)
                    agent.eval()
                    ai_agent = agent
                    _v5_governor = RuleGovernor(
                        num_junctions=NUM_NODES,
                        num_phases=4,
                        min_green_s=10.0,
                        max_green_s=90.0,
                        flicker_window=2,
                        flicker_penalty=3.0,
                        pressure_boost=1.0,
                    )
                    print(f"[OK] TraFixV5 loaded: {abs_path}")
                    return True
                except RuntimeError as e:
                    print(f"[WARN] v5 weight mismatch: {abs_path} — {e}")
                    continue
        print("[WARN] TraFixV5 weights not found. Heuristic fallback active.")
        return False

    # ── v2/v3 ─────────────────────────────────────────────────────────────────
    agent = CoordinatedPPOAgent(
        num_node_features=NUM_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_actions=NUM_ACTIONS,
        num_heads=NUM_HEADS,
    )
    weight_paths = [
        os.path.join(base_dir, "ai", _WEIGHT_FILENAME),
        os.path.join(base_dir, "..", _WEIGHT_FILENAME),
        _WEIGHT_FILENAME,
    ]
    for path in weight_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            try:
                sd = torch.load(abs_path, map_location="cpu", weights_only=True)
                if isinstance(sd, dict) and "model_state_dict" in sd:
                    sd = sd["model_state_dict"]
                agent.load_state_dict(sd)
                agent.eval()
                ai_agent = agent
                print(f"[OK] AI model loaded: {abs_path}")
                return True
            except RuntimeError as e:
                print(f"[WARN] Weight mismatch: {abs_path} — {e}")
                continue

    print("[WARN] AI model weights not found. Heuristic fallback active.")
    return False


@app.on_event("startup")
async def startup_event():
    load_model()


# ── Helper: build obs_list from state_dict ─────────────────────────────────

def _build_obs_list() -> list:
    obs_list = []
    for i in range(NUM_NODES):
        node = state_dict.get(str(i), {})
        if node:
            obs_list.append(node)
        else:
            obs_list.append({
                "intersection_id": i,
                "north_left": 0, "north_through": 0, "north_right": 0,
                "south_left": 0, "south_through": 0, "south_right": 0,
                "east_left":  0, "east_through":  0, "east_right":  0,
                "west_left":  0, "west_through":  0, "west_right":  0,
                "queue_length": 0.0, "current_phase": 0, "phase_duration": 0.0,
            })
    return obs_list


class TelemetryBatch(BaseModel):
    step: int
    intersections: List[Telemetry]


# ── POST /telemetry_batch ──────────────────────────────────────────────────────

@app.post("/telemetry_batch")
async def receive_telemetry_batch(batch: TelemetryBatch):
    global last_decisions_cache, _last_batch_step

    # Detect simulation restart (step counter reset) and clear stale window/governor state
    if batch.step < _last_batch_step:
        _v6_window.clear()
        _v5_window.clear()
        if _v6_governor is not None:
            _v6_governor.reset()
        if _v5_governor is not None:
            _v5_governor.reset()
    _last_batch_step = batch.step

    for data in batch.intersections:
        d = data.dict()
        # If only old-format counts provided (v2 compat), distribute to through lanes
        if d.get("north_count") is not None and d.get("north_through") == 0:
            d["north_through"] = d.pop("north_count", 0)
            d["south_through"] = d.pop("south_count", 0)
            d["east_through"]  = d.pop("east_count", 0)
            d["west_through"]  = d.pop("west_count", 0)
        state_dict[str(data.intersection_id)] = d

    decisions = []

    # ── AI MODEL ──────────────────────────────────────────────────────────────
    if ai_agent is not None:
        obs_list = _build_obs_list()
        node_features = parse_sumo_observations(obs_list)  # [5, 20]

        with torch.no_grad():

            # ── v6 ────────────────────────────────────────────────────────────
            if _USE_V6:
                from trafix_v6.rule_governor import sample_governed
                if len(_v6_window) == 0:
                    for _ in range(_V6_T_WINDOW):
                        _v6_window.append(node_features.detach())
                else:
                    _v6_window.append(node_features.detach())

                window_tensor = torch.stack(list(_v6_window)).unsqueeze(0)  # [1,T,5,20]
                logits_list, _ = ai_agent(window_tensor)

                obs_last = window_tensor[0, -1]
                if _v6_governor is not None:
                    logits_list = _v6_governor.apply(logits_list, obs_last)

                action_probs = torch.stack(
                    [torch.softmax(l, dim=-1).squeeze(0) for l in logits_list], dim=0
                )

            # ── v5 ────────────────────────────────────────────────────────────
            elif _USE_V5:
                from trafix_v5.rule_governor import sample_governed as _sg
                if len(_v5_window) == 0:
                    for _ in range(_V5_T_WINDOW):
                        _v5_window.append(node_features.detach())
                else:
                    _v5_window.append(node_features.detach())
                window_tensor = torch.stack(list(_v5_window)).unsqueeze(0)
                logits_list, _ = ai_agent(window_tensor)
                obs_last = window_tensor[0, -1]
                if _v5_governor is not None:
                    logits_list = _v5_governor.apply(logits_list, obs_last)
                action_probs = torch.stack(
                    [torch.softmax(l, dim=-1).squeeze(0) for l in logits_list], dim=0
                )

            # ── v2/v3 ─────────────────────────────────────────────────────────
            elif _USE_GRAPH:
                action_probs, _ = ai_agent(node_features, edge_index)
            else:
                action_probs, _ = ai_agent(node_features)

            chosen_phases = []
            for data in batch.intersections:
                idx = data.intersection_id
                if 0 <= idx < action_probs.shape[0]:
                    next_phase = int(torch.argmax(action_probs[idx]).item())
                    confidence = round(float(action_probs[idx].max().item()), 3)
                else:
                    next_phase = data.current_phase
                    confidence = 0.0
                chosen_phases.append(next_phase)

                total_veh = (
                    data.north_left + data.north_through + data.north_right
                    + data.south_left + data.south_through + data.south_right
                    + data.east_left + data.east_through + data.east_right
                    + data.west_left + data.west_through + data.west_right
                )
                decisions.append({
                    "intersection_id": idx,
                    "next_phase": next_phase,
                    "confidence": confidence,
                    "total_vehicles": total_veh,
                    "queue_length": round(data.queue_length, 1),
                })

            # Update governor state for anti-flicker (guard: must have all junctions)
            if _USE_V6 and _v6_governor is not None and len(chosen_phases) == NUM_NODES:
                _v6_governor.update_state(torch.tensor(chosen_phases, dtype=torch.long))
            elif _USE_V5 and _v5_governor is not None and len(chosen_phases) == NUM_NODES:
                _v5_governor.update_state(torch.tensor(chosen_phases, dtype=torch.long))

        last_decisions_cache = decisions
        return {"decisions": decisions}

    # ── HEURISTIC FALLBACK ────────────────────────────────────────────────────
    for data in batch.intersections:
        # Use through-lane counts for heuristic
        ns_demand = data.north_through + data.south_through
        ew_demand = data.east_through + data.west_through

        if data.phase_duration < 10.0 or (ns_demand == 0 and ew_demand == 0):
            next_phase = data.current_phase
        else:
            # Phase 0 = NS-through, Phase 3 = EW-through
            next_phase = 0 if ns_demand >= ew_demand else 3

        total_veh = (
            data.north_left + data.north_through + data.north_right
            + data.south_left + data.south_through + data.south_right
            + data.east_left + data.east_through + data.east_right
            + data.west_left + data.west_through + data.west_right
        )
        decisions.append({
            "intersection_id": data.intersection_id,
            "next_phase": next_phase,
            "confidence": 0.0,
            "total_vehicles": total_veh,
            "queue_length": round(data.queue_length, 1),
        })

    last_decisions_cache = decisions
    return {"decisions": decisions}


@app.get("/last_decisions")
async def get_last_decisions():
    return {"decisions": last_decisions_cache}


@app.get("/state")
async def get_state():
    return state_dict
