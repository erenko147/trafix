"""
Telemetry ingestion routes.

POST /telemetry_batch  — receives per-step observation bundle from SUMO runner,
                         runs the AI model (or heuristic fallback), and returns
                         phase decisions for every junction.
"""

import torch
from fastapi import APIRouter

import backend.api.state as state
from backend.api.schemas import TelemetryBatch
from backend.ai.observation import parse_sumo_observations

router = APIRouter()


def _build_obs_list() -> list:
    obs_list = []
    for i in range(state.NUM_NODES):
        node = state.state_dict.get(str(i), {})
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


@router.post("/telemetry_batch")
async def receive_telemetry_batch(batch: TelemetryBatch):
    # Detect simulation restart and clear stale window / governor state
    if batch.step < state._last_batch_step:
        state._v6_window.clear()
        if state._v6_governor is not None:
            state._v6_governor.reset()
    state._last_batch_step = batch.step

    for data in batch.intersections:
        d = data.dict()
        if d.get("north_count") is not None and d.get("north_through") == 0:
            d["north_through"] = d.pop("north_count", 0)
            d["south_through"] = d.pop("south_count", 0)
            d["east_through"]  = d.pop("east_count", 0)
            d["west_through"]  = d.pop("west_count", 0)
        state.state_dict[str(data.intersection_id)] = d

    decisions = []

    # ── AI model path ─────────────────────────────────────────────────────────
    if state.ai_agent is not None:
        obs_list     = _build_obs_list()
        node_features = parse_sumo_observations(obs_list)

        with torch.no_grad():
            if len(state._v6_window) == 0:
                for _ in range(state._V6_T_WINDOW):
                    state._v6_window.append(node_features.detach())
            else:
                state._v6_window.append(node_features.detach())

            window_tensor          = torch.stack(list(state._v6_window)).unsqueeze(0)
            logits_list, _         = state.ai_agent(window_tensor)
            obs_last               = window_tensor[0, -1]

            if state._v6_governor is not None:
                logits_list = state._v6_governor.apply(logits_list, obs_last)

            action_probs = torch.stack(
                [torch.softmax(l, dim=-1).squeeze(0) for l in logits_list], dim=0
            )

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
                    "next_phase":      next_phase,
                    "confidence":      confidence,
                    "total_vehicles":  total_veh,
                    "queue_length":    round(data.queue_length, 1),
                })

            if state._v6_governor is not None and len(chosen_phases) == state.NUM_NODES:
                state._v6_governor.update_state(
                    torch.tensor(chosen_phases, dtype=torch.long)
                )

        state.last_decisions_cache = decisions
        return {"decisions": decisions}

    # ── Heuristic fallback ────────────────────────────────────────────────────
    for data in batch.intersections:
        ns_demand = data.north_through + data.south_through
        ew_demand = data.east_through  + data.west_through

        if data.phase_duration < 10.0 or (ns_demand == 0 and ew_demand == 0):
            next_phase = data.current_phase
        else:
            next_phase = 0 if ns_demand >= ew_demand else 3

        total_veh = (
            data.north_left + data.north_through + data.north_right
            + data.south_left + data.south_through + data.south_right
            + data.east_left + data.east_through + data.east_right
            + data.west_left + data.west_through + data.west_right
        )
        decisions.append({
            "intersection_id": data.intersection_id,
            "next_phase":      next_phase,
            "confidence":      0.0,
            "total_vehicles":  total_veh,
            "queue_length":    round(data.queue_length, 1),
        })

    state.last_decisions_cache = decisions
    return {"decisions": decisions}
