"""
SUMO training environment — wraps TraCI to expose a gym-like interface.

SumoEnvironment handles:
  • Starting / closing SUMO per episode
  • Collecting per-lane vehicle counts from all junctions
  • Applying model actions with proper yellow-transition logic
  • Providing per-step metrics for logging

build_edge_index() extracts the junction adjacency graph from the SUMO net file.
"""

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch

from backend.training.config import TrainConfig

# ── SUMO TraCI ────────────────────────────────────────────────────────────────

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    for candidate in [
        "C:\\Program Files (x86)\\Eclipse\\Sumo\\tools",
        "C:\\Program Files\\Eclipse\\Sumo\\tools",
        "/usr/share/sumo/tools",
        "/usr/local/share/sumo/tools",
    ]:
        if os.path.isdir(candidate):
            sys.path.append(candidate)
            break

try:
    import traci
    import sumolib
except ImportError:
    raise ImportError(
        "SUMO TraCI not found. Set SUMO_HOME environment variable:\n"
        "  export SUMO_HOME=/path/to/sumo"
    )

# ── Phase mappings ────────────────────────────────────────────────────────────

MODEL_TO_SUMO_GREEN = {0: 0, 1: 2, 2: 4, 3: 6, 4: 8, 5: 10}
SUMO_TO_MODEL_PHASE = {0:0,1:0, 2:1,3:1, 4:2,5:2, 6:3,7:3, 8:4,9:4, 10:5,11:5}
LANE_TYPE           = {0: "right", 1: "through", 2: "left"}


class SumoEnvironment:
    """
    SUMO simulation managed via TraCI.

    Each episode: start() → repeated step(actions) → close().
    """

    YELLOW_STEPS = 3

    def __init__(self, cfg: TrainConfig):
        self.cfg                    = cfg
        self.tls_ids: List[str]     = []
        self.num_nodes              = 0
        self._step_count            = 0
        self._episode_count         = 0
        self._pending_target:       Dict[str, int] = {}
        self._yellow_steps_remaining: Dict[str, int] = {}
        self._phase_held_since:     Dict[str, int] = {}

    # ── Episode lifecycle ─────────────────────────────────────────────────────

    def start(self, episode: int = 0):
        cfg_path = os.path.abspath(self.cfg.sumo_cfg)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"\n{'=' * 60}\n"
                f"  SUMO config not found!\n"
                f"  Path : {cfg_path}\n"
                f"  CWD  : {os.getcwd()}\n"
                f"{'=' * 60}\n"
                f"  Fix: run from project root:\n"
                f"    python -m backend.training.trainer\n"
                f"  or pass --sumo-cfg explicitly.\n"
                f"{'=' * 60}"
            )

        sumo_binary = "sumo-gui" if self.cfg.gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", cfg_path,
            "--step-length",          str(self.cfg.sumo_step_length),
            "--waiting-time-memory",  "1000",
            "--time-to-teleport",     "-1",
            "--no-warnings",          "true",
            "--random",
            "--seed",                 str(self.cfg.seed + episode),
        ]

        logging.info(f"  Starting SUMO: {cfg_path}")
        traci.start(sumo_cmd)
        self._step_count              = 0
        self._episode_count           = episode
        self._pending_target          = {}
        self._yellow_steps_remaining  = {}
        self._phase_held_since        = {}

        self.tls_ids   = sorted(traci.trafficlight.getIDList())
        self.num_nodes = len(self.tls_ids)

        if self.num_nodes == 0:
            raise RuntimeError(
                "No traffic lights found in SUMO network. "
                "Ensure map.net.xml contains <tlLogic> elements."
            )

        for _ in range(self.cfg.warmup_steps):
            traci.simulationStep()
            self._step_count += 1

        self._phase_held_since = {tls_id: self._step_count for tls_id in self.tls_ids}

    def close(self):
        try:
            traci.close()
        except Exception:
            pass

    @property
    def is_running(self) -> bool:
        try:
            return traci.simulation.getMinExpectedNumber() > 0
        except Exception:
            return False

    # ── Observation collection ────────────────────────────────────────────────

    def get_observations(self) -> List[Dict]:
        observations = []

        for idx, tls_id in enumerate(self.tls_ids):
            jx, jy = traci.junction.getPosition(tls_id)

            counts = {
                "north_left": 0, "north_through": 0, "north_right": 0,
                "south_left": 0, "south_through": 0, "south_right": 0,
                "east_left":  0, "east_through":  0, "east_right":  0,
                "west_left":  0, "west_through":  0, "west_right":  0,
            }

            controlled_links = traci.trafficlight.getControlledLinks(tls_id)
            seen_lanes       = set()

            for link in controlled_links:
                if not link:
                    continue
                from_lane = link[0][0]
                if from_lane in seen_lanes:
                    continue
                seen_lanes.add(from_lane)

                edge_id   = from_lane.rsplit("_", 1)[0]
                lane_idx  = int(from_lane.rsplit("_", 1)[1])
                lane_type = LANE_TYPE.get(lane_idx, "through")
                direction = self._classify_edge_direction(edge_id, jx, jy)

                if direction and lane_type:
                    key = f"{direction}_{lane_type}"
                    if key in counts:
                        counts[key] += traci.lane.getLastStepVehicleNumber(from_lane)

            sumo_phase  = traci.trafficlight.getPhase(tls_id)
            model_phase = SUMO_TO_MODEL_PHASE.get(sumo_phase, 0)
            elapsed     = float(
                self._step_count - self._phase_held_since.get(tls_id, self._step_count)
            )
            total_queue = sum(counts.values())

            observations.append({
                "intersection_id": idx,
                **counts,
                "queue_length":  min(total_queue * 1.5, 200.0),
                "current_phase": model_phase,
                "phase_duration": elapsed,
            })

        return observations

    def _classify_edge_direction(self, edge_id: str, jx: float, jy: float) -> str:
        try:
            shape = traci.lane.getShape(f"{edge_id}_0")
            if not shape:
                return ""
            x0, y0 = shape[0]
            dx, dy = x0 - jx, y0 - jy
            if abs(dx) > abs(dy):
                return "west" if dx < 0 else "east"
            else:
                return "south" if dy < 0 else "north"
        except Exception:
            return ""

    # ── Action application ────────────────────────────────────────────────────

    def apply_actions(self, actions: torch.Tensor):
        for i, tls_id in enumerate(self.tls_ids):
            if self._yellow_steps_remaining.get(tls_id, 0) > 0:
                continue

            model_action      = int(actions[i].item()) % 6
            target_sumo_phase = MODEL_TO_SUMO_GREEN[model_action]
            current_sumo_phase = traci.trafficlight.getPhase(tls_id)

            if target_sumo_phase == current_sumo_phase:
                traci.trafficlight.setPhase(tls_id, current_sumo_phase)
                continue

            self._pending_target[tls_id]         = target_sumo_phase
            self._yellow_steps_remaining[tls_id] = self.YELLOW_STEPS

            yellow_phase = (
                current_sumo_phase + 1
                if current_sumo_phase % 2 == 0
                else current_sumo_phase
            )
            traci.trafficlight.setPhase(tls_id, yellow_phase)

    def _advance_transitions(self):
        for tls_id in self.tls_ids:
            remaining = self._yellow_steps_remaining.get(tls_id, 0)
            if remaining <= 0:
                continue
            remaining -= 1
            self._yellow_steps_remaining[tls_id] = remaining
            if remaining == 0:
                target = self._pending_target.pop(tls_id, None)
                if target is not None:
                    traci.trafficlight.setPhase(tls_id, target)
                    self._phase_held_since[tls_id] = self._step_count

    def step(self, actions: Optional[torch.Tensor] = None) -> Tuple[List[Dict], bool]:
        if actions is not None:
            self.apply_actions(actions)

        for _ in range(self.cfg.decision_interval):
            if not self.is_running:
                return self.get_observations(), True
            self._advance_transitions()
            traci.simulationStep()
            self._step_count += 1

        done = (
            not self.is_running
            or self._step_count >= self.cfg.max_steps_per_episode
        )
        return self.get_observations(), done

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, float]:
        try:
            vehicles = traci.vehicle.getIDList()
            if not vehicles:
                return {"avg_speed": 0, "avg_waiting": 0,
                        "total_vehicles": 0, "total_halting": 0}

            speeds  = [traci.vehicle.getSpeed(v) for v in vehicles]
            waiting = [traci.vehicle.getWaitingTime(v) for v in vehicles]
            halting = sum(1 for s in speeds if s < 0.1)

            return {
                "avg_speed":      sum(speeds) / len(speeds),
                "avg_waiting":    sum(waiting) / len(waiting),
                "total_vehicles": len(vehicles),
                "total_halting":  halting,
            }
        except Exception:
            return {"avg_speed": 0, "avg_waiting": 0,
                    "total_vehicles": 0, "total_halting": 0}


# ── Graph topology ────────────────────────────────────────────────────────────

def build_edge_index(num_nodes: int, net_file: str = None) -> torch.Tensor:
    """
    Build junction adjacency graph.

    Tries to parse the SUMO net file with sumolib first; falls back to a
    hard-coded 5-junction chain topology if that fails.
    """
    if net_file and os.path.exists(net_file):
        try:
            net      = sumolib.net.readNet(net_file)
            tls_nodes = sorted(
                [n for n in net.getNodes() if n.getType() == "traffic_light"],
                key=lambda n: n.getID(),
            )
            if len(tls_nodes) >= 2:
                node_ids  = [n.getID() for n in tls_nodes]
                id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

                edges_src, edges_dst = [], []
                for edge in net.getEdges():
                    src = edge.getFromNode().getID()
                    dst = edge.getToNode().getID()
                    if src in id_to_idx and dst in id_to_idx:
                        s, d = id_to_idx[src], id_to_idx[dst]
                        if s != d:
                            edges_src.append(s)
                            edges_dst.append(d)

                if edges_src:
                    all_src = edges_src + edges_dst
                    all_dst = edges_dst + edges_src
                    seen, final_src, final_dst = set(), [], []
                    for s, d in zip(all_src, all_dst):
                        if (s, d) not in seen:
                            seen.add((s, d))
                            final_src.append(s)
                            final_dst.append(d)

                    logging.info(
                        f"Graph extracted from net file: "
                        f"{len(tls_nodes)} junctions, {len(final_src)} edges"
                    )
                    return torch.tensor([final_src, final_dst], dtype=torch.long)
        except Exception as e:
            logging.warning(f"Could not extract graph from net file: {e}")

    # Default: 5-junction asymmetric grid
    #   0 — 1 — 2
    #       |   |
    #       3 — 4
    logging.info("Using default 5-junction topology")
    return torch.tensor([
        [0, 1, 1, 2, 1, 3, 2, 4, 3, 4],
        [1, 0, 2, 1, 3, 1, 4, 2, 4, 3],
    ], dtype=torch.long)
