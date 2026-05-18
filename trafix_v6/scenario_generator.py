"""
TraFix v6 — Scenario Generator
================================
Generates randomized but structured SUMO .rou.xml files per episode,
following a curriculum schedule that gradually increases difficulty.

ScenarioType variants:
  OFFPEAK       — uniform low flow across all routes
  MORNING_PEAK  — heavy inbound (J0/J1/J2-side → J3/J4-side)
  EVENING_PEAK  — heavy outbound (J3/J4-side → J0/J1/J2-side)
  INCIDENT      — OFFPEAK base with one junction fully blocked for a window
  PULSE         — near-zero quiet window then a directional traffic burst

ScenarioEnvironment wraps SumoEnvironment with a per-episode --route-files
override. Importing this module does NOT import traci; SUMO is only needed
when ScenarioEnvironment.start() is called.

Network topology (inferred from sumo/map.net.xml):
  Traffic-light junctions: J0, J1, J2, J3, J4
  Fringe entry edges per junction:
    J0: -E5, -E6
    J1: -E7, -E8
    J2: -E9
    J3: -E10, -E11
    J4: -E12, -E13, -E14
"""

import os
import sys
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DEFAULT_NET_FILE = str(_PROJECT_ROOT / "sumo" / "map.net.xml")
_DEFAULT_OUTPUT_DIR = str(_SCRIPT_DIR / "scenarios")


# ──────────────────────────────────────────────
#  Scenario types
# ──────────────────────────────────────────────

class ScenarioType(Enum):
    OFFPEAK = "OFFPEAK"
    MORNING_PEAK = "MORNING_PEAK"
    EVENING_PEAK = "EVENING_PEAK"
    INCIDENT = "INCIDENT"
    PULSE = "PULSE"


# ──────────────────────────────────────────────
#  Network constants (from sumo/map.net.xml)
# ──────────────────────────────────────────────

_JUNCTION_FRINGE_IN: Dict[int, List[str]] = {
    0: ["-E5", "-E6"],
    1: ["-E7", "-E8"],
    2: ["-E9"],
    3: ["-E10", "-E11"],
    4: ["-E12", "-E13", "-E14"],
}

_MAIN_INBOUND_OD: List[Tuple[str, str]] = [
    ("-E5", "E10"), ("-E5", "E11"), ("-E5", "E12"), ("-E5", "E13"),
    ("-E6", "E10"), ("-E6", "E12"), ("-E6", "E14"),
    ("-E7", "E10"), ("-E7", "E11"), ("-E7", "E14"),
    ("-E8", "E11"), ("-E8", "E12"), ("-E8", "E13"),
    ("-E9", "E10"), ("-E9", "E11"), ("-E9", "E12"), ("-E9", "E14"),
]

_MAIN_OUTBOUND_OD: List[Tuple[str, str]] = [
    ("-E10", "E5"), ("-E10", "E7"), ("-E10", "E9"),
    ("-E11", "E5"), ("-E11", "E8"), ("-E11", "E9"),
    ("-E12", "E5"), ("-E12", "E7"), ("-E12", "E9"),
    ("-E13", "E6"), ("-E13", "E8"), ("-E13", "E9"),
    ("-E14", "E6"), ("-E14", "E7"), ("-E14", "E9"),
]

_LOCAL_OD: List[Tuple[str, str]] = [
    ("-E5", "E6"), ("-E5", "E7"), ("-E5", "E8"), ("-E5", "E9"),
    ("-E6", "E5"), ("-E6", "E7"), ("-E6", "E8"), ("-E6", "E9"),
    ("-E7", "E5"), ("-E7", "E6"), ("-E7", "E8"), ("-E7", "E9"),
    ("-E8", "E5"), ("-E8", "E6"), ("-E8", "E7"), ("-E8", "E9"),
    ("-E9", "E5"), ("-E9", "E6"), ("-E9", "E7"), ("-E9", "E8"),
    ("-E10", "E11"), ("-E11", "E10"),
    ("-E10", "E12"), ("-E10", "E13"), ("-E10", "E14"),
    ("-E11", "E12"), ("-E11", "E13"), ("-E11", "E14"),
    ("-E12", "E13"), ("-E12", "E14"),
    ("-E13", "E12"), ("-E13", "E14"),
    ("-E14", "E12"), ("-E14", "E13"),
]

_ALL_OD: List[Tuple[str, str]] = _MAIN_INBOUND_OD + _MAIN_OUTBOUND_OD + _LOCAL_OD

# Weights order: OFFPEAK, MORNING_PEAK, EVENING_PEAK, INCIDENT, PULSE
_CURRICULUM: List[Tuple[int, List[float]]] = [
    (800, [0.15, 0.20, 0.20, 0.25, 0.20]),
    (500, [0.20, 0.25, 0.25, 0.15, 0.15]),
    (200, [0.35, 0.25, 0.25, 0.00, 0.15]),
    (0,   [0.85, 0.00, 0.00, 0.00, 0.15]),
]
_SCENARIO_ORDER = [
    ScenarioType.OFFPEAK,
    ScenarioType.MORNING_PEAK,
    ScenarioType.EVENING_PEAK,
    ScenarioType.INCIDENT,
    ScenarioType.PULSE,
]


class ScenarioGenerator:
    """Generates randomized SUMO .rou.xml files per episode."""

    def __init__(
        self,
        net_file: str = _DEFAULT_NET_FILE,
        output_dir: str = _DEFAULT_OUTPUT_DIR,
        seed: Optional[int] = None,
        flow_horizon: int = 1800,
    ):
        self.net_file = net_file
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.flow_horizon = flow_horizon
        self.last_summary = ""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._valid_edges: set = self._parse_net_edges(net_file)

        for from_e, to_e in _ALL_OD:
            self._check_edge(from_e)
            self._check_edge(to_e)

    def generate(self, scenario_type: ScenarioType, episode: int) -> str:
        rng = self._make_rng(episode)
        episode_duration = int(rng.integers(300, 501))
        out_path = self.output_dir / f"ep{episode:04d}.rou.xml"

        if scenario_type == ScenarioType.OFFPEAK:
            params = self._gen_offpeak(rng, episode, episode_duration, self.flow_horizon)
        elif scenario_type == ScenarioType.MORNING_PEAK:
            params = self._gen_morning_peak(rng, episode, episode_duration, self.flow_horizon)
        elif scenario_type == ScenarioType.EVENING_PEAK:
            params = self._gen_evening_peak(rng, episode, episode_duration, self.flow_horizon)
        elif scenario_type == ScenarioType.INCIDENT:
            params = self._gen_incident(rng, episode, episode_duration, self.flow_horizon)
        elif scenario_type == ScenarioType.PULSE:
            params = self._gen_pulse(rng, episode, episode_duration, self.flow_horizon)
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")

        self._write_rou_xml(out_path, params["flows"])
        self.last_summary = self.summary(scenario_type, params)
        return str(out_path)

    def sample(self, episode: int) -> Tuple[ScenarioType, str]:
        scenario_type = self.curriculum_schedule(episode)
        route_file = self.generate(scenario_type, episode)
        return scenario_type, route_file

    def curriculum_schedule(self, episode: int) -> ScenarioType:
        weights = None
        for threshold, w in _CURRICULUM:
            if episode >= threshold:
                weights = w
                break
        assert weights is not None
        rng = self._make_rng(episode + 10_000_000)
        choice_idx = int(rng.choice(len(_SCENARIO_ORDER), p=weights))
        return _SCENARIO_ORDER[choice_idx]

    def summary(self, scenario_type: ScenarioType, params: dict) -> str:
        ep = params.get("episode", 0)
        dur = params.get("episode_duration", 0)
        if scenario_type == ScenarioType.OFFPEAK:
            return f"ep{ep:04d} | OFFPEAK | flow={params['base_flow']:.0f} veh/hr | duration={dur} steps"
        elif scenario_type == ScenarioType.MORNING_PEAK:
            return f"ep{ep:04d} | MORNING_PEAK | main={params['main_flow']:.0f} | side={params['side_flow']:.0f} | duration={dur}"
        elif scenario_type == ScenarioType.EVENING_PEAK:
            return f"ep{ep:04d} | EVENING_PEAK | main={params['main_flow']:.0f} | side={params['side_flow']:.0f} | duration={dur}"
        elif scenario_type == ScenarioType.INCIDENT:
            return (f"ep{ep:04d} | INCIDENT | flow={params['base_flow']:.0f} | "
                    f"junction=J{params['incident_junction']} | onset={params['onset_step']}s | duration={dur}")
        elif scenario_type == ScenarioType.PULSE:
            od_names = ["INBOUND", "OUTBOUND", "ALL"]
            return (f"ep{ep:04d} | PULSE | quiet=0-{params['quiet_end']}s"
                    f" burst={params['quiet_end']}-{params['burst_end']}s"
                    f" flow={params['burst_flow']:.0f} ods={od_names[params['od_choice']]}"
                    f" | duration={dur}")
        return f"ep{ep:04d} | {scenario_type.value} | duration={dur}"

    def _gen_offpeak(self, rng, episode, duration, flow_horizon):
        base_flow = float(rng.uniform(200, 500))
        per_od = base_flow / len(_ALL_OD)
        flows = [(from_e, to_e, 0, flow_horizon, per_od) for from_e, to_e in _ALL_OD]
        return {"episode": episode, "episode_duration": duration, "base_flow": base_flow, "flows": flows}

    def _gen_morning_peak(self, rng, episode, duration, flow_horizon):
        main_flow = float(rng.uniform(800, 1200))
        side_flow = float(rng.uniform(100, 300))
        per_main = main_flow / len(_MAIN_INBOUND_OD)
        per_side = side_flow / (len(_MAIN_OUTBOUND_OD) + len(_LOCAL_OD))
        flows = (
            [(from_e, to_e, 0, flow_horizon, per_main) for from_e, to_e in _MAIN_INBOUND_OD]
            + [(from_e, to_e, 0, flow_horizon, per_side) for from_e, to_e in _MAIN_OUTBOUND_OD + _LOCAL_OD]
        )
        return {"episode": episode, "episode_duration": duration, "main_flow": main_flow, "side_flow": side_flow, "flows": flows}

    def _gen_evening_peak(self, rng, episode, duration, flow_horizon):
        main_flow = float(rng.uniform(800, 1200))
        side_flow = float(rng.uniform(100, 300))
        per_main = main_flow / len(_MAIN_OUTBOUND_OD)
        per_side = side_flow / (len(_MAIN_INBOUND_OD) + len(_LOCAL_OD))
        flows = (
            [(from_e, to_e, 0, flow_horizon, per_main) for from_e, to_e in _MAIN_OUTBOUND_OD]
            + [(from_e, to_e, 0, flow_horizon, per_side) for from_e, to_e in _MAIN_INBOUND_OD + _LOCAL_OD]
        )
        return {"episode": episode, "episode_duration": duration, "main_flow": main_flow, "side_flow": side_flow, "flows": flows}

    def _gen_incident(self, rng, episode, duration, flow_horizon):
        base_flow = float(rng.uniform(200, 500))
        per_od = base_flow / len(_ALL_OD)
        incident_junction = int(rng.integers(0, 5))
        onset_step = int(rng.integers(0, duration // 2))
        block_duration = int(rng.integers(50, 151))
        incident_end = min(onset_step + block_duration, duration)
        blocked_fringe = set(_JUNCTION_FRINGE_IN[incident_junction])
        flows = []
        for from_e, to_e in _ALL_OD:
            if from_e in blocked_fringe:
                if onset_step > 0:
                    flows.append((from_e, to_e, 0, onset_step, per_od))
                if incident_end < flow_horizon:
                    flows.append((from_e, to_e, incident_end, flow_horizon, per_od))
            else:
                flows.append((from_e, to_e, 0, flow_horizon, per_od))
        return {"episode": episode, "episode_duration": duration, "base_flow": base_flow,
                "incident_junction": incident_junction, "onset_step": onset_step,
                "block_duration": block_duration, "flows": flows}

    def _gen_pulse(self, rng, episode, duration, flow_horizon):
        quiet_end  = int(rng.integers(60, 301))                   # 1–5 min quiet window
        burst_dur  = int(rng.integers(120, 301))                  # 2–5 min burst
        burst_end  = min(quiet_end + burst_dur, flow_horizon)
        burst_flow = float(rng.uniform(600, 1000))                # veh/hr during burst
        quiet_flow = float(rng.uniform(20, 80))                   # near-zero background

        # Randomly pick which OD subset carries the burst
        od_choice = int(rng.integers(0, 3))
        burst_ods = [_MAIN_INBOUND_OD, _MAIN_OUTBOUND_OD, _ALL_OD][od_choice]

        per_burst = burst_flow / len(burst_ods)
        per_quiet = quiet_flow / len(_ALL_OD)

        flows = []
        if quiet_end > 0:
            flows += [(f, t, 0, quiet_end, per_quiet) for f, t in _ALL_OD]
        flows += [(f, t, quiet_end, burst_end, per_burst) for f, t in burst_ods]
        if burst_end < flow_horizon:
            flows += [(f, t, burst_end, flow_horizon, per_quiet) for f, t in _ALL_OD]

        return {"episode": episode, "episode_duration": duration,
                "quiet_end": quiet_end, "burst_end": burst_end,
                "burst_flow": burst_flow, "od_choice": od_choice, "flows": flows}

    def _write_rou_xml(self, out_path, flows):
        for from_e, to_e, begin, end, rate in flows:
            self._check_edge(from_e)
            self._check_edge(to_e)
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
            ' xsi:noNamespaceSchemaLocation='
            '"http://sumo.dlr.de/xsd/routes_file.xsd">',
            '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5"'
            ' length="5" minGap="2.5" maxSpeed="50"/>',
        ]
        for idx, (from_e, to_e, begin, end, rate) in enumerate(flows):
            if rate < 0.5 or end <= begin:
                continue
            safe_from = from_e.lstrip("-")
            safe_to = to_e.lstrip("-")
            fid = f"f_{safe_from}_{safe_to}_{idx}"
            lines.append(
                f'    <flow id="{fid}" from="{from_e}" to="{to_e}"'
                f' begin="{begin}" end="{end}"'
                f' vehsPerHour="{rate:.1f}"'
                f' departLane="best" departSpeed="max"/>'
            )
        lines.append("</routes>")
        out_path.write_text("\n".join(lines), encoding="utf-8")

    def _check_edge(self, edge_id):
        if edge_id not in self._valid_edges:
            raise ValueError(f"Edge '{edge_id}' not found in net file: {self.net_file}")

    @staticmethod
    def _parse_net_edges(net_file):
        if not os.path.exists(net_file):
            warnings.warn(f"Net file not found: {net_file}. Edge validation skipped.", UserWarning, stacklevel=3)
            return set()
        tree = ET.parse(net_file)
        return {e.get("id", "") for e in tree.getroot().findall("edge")
                if e.get("id", "") and not e.get("id", "").startswith(":")}

    def _make_rng(self, episode):
        if self.seed is not None:
            return np.random.default_rng(self.seed + episode)
        return np.random.default_rng()


class ScenarioEnvironment:
    """Wraps SumoEnvironment with per-episode --route-files injection."""

    def __init__(self, cfg):
        self._cfg = cfg
        self._route_file: Optional[str] = None
        self._env = None

    def set_route_file(self, path: str):
        self._route_file = path

    def start(self, episode: int = 0):
        import traci as _traci
        SumoEnvironment = self._import_sumo_env()
        if self._env is None:
            self._env = SumoEnvironment(self._cfg)

        cfg = self._cfg
        cfg_path = os.path.abspath(cfg.sumo_cfg)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"\n{'=' * 60}\n"
                f"  SUMO config not found: {cfg_path}\n"
                f"  Working dir: {os.getcwd()}\n"
                f"{'=' * 60}"
            )

        sumo_binary = "sumo-gui" if cfg.gui else "sumo"
        sumo_cmd = [
            sumo_binary, "-c", cfg_path,
            "--step-length", str(cfg.sumo_step_length),
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1",   # disable teleportation during training
            "--no-warnings", "true",
            "--random",
            "--seed", str(cfg.seed + episode),
        ]

        if self._route_file and os.path.exists(self._route_file):
            sumo_cmd.extend(["--route-files", self._route_file])

        _traci.start(sumo_cmd)

        self._env._step_count = 0
        self._env._episode_count = episode
        self._env.tls_ids = sorted(_traci.trafficlight.getIDList())
        self._env.num_nodes = len(self._env.tls_ids)

        if self._env.num_nodes == 0:
            raise RuntimeError("No TL nodes found. Check map.net.xml has <tlLogic> elements.")

        for _ in range(cfg.warmup_steps):
            _traci.simulationStep()
            self._env._step_count += 1

        # Reset per-episode state that SumoEnvironment.start() normally handles
        # but is bypassed here. Without this, _phase_held_since stays empty →
        # phase_duration = 0 always → governor hard-mask blocks all switches →
        # entropy = 0 → policy never updates.
        self._env._pending_target = {}
        self._env._yellow_steps_remaining = {}
        self._env._phase_held_since = {
            tls_id: self._env._step_count for tls_id in self._env.tls_ids
        }

    def close(self):
        if self._env is not None:
            self._env.close()

    def __getattr__(self, name):
        env = object.__getattribute__(self, "_env")
        if env is None:
            raise AttributeError(f"'{name}' accessed before start() was called.")
        return getattr(env, name)

    @staticmethod
    def _import_sumo_env():
        sd = Path(__file__).resolve().parent
        pr = sd.parent
        for path in [str(sd), str(pr)]:
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            from backend.ai.train_v2 import SumoEnvironment
            return SumoEnvironment
        except ImportError:
            pass
        try:
            from train_v2 import SumoEnvironment
            return SumoEnvironment
        except ImportError:
            pass
        sys.path.insert(0, str(pr / "backend" / "ai"))
        from train_v2 import SumoEnvironment
        return SumoEnvironment
