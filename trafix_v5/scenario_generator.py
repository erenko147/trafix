"""
TraFix v5 — Scenario Generator
================================
Generates randomized but structured SUMO .rou.xml files per episode,
following a curriculum schedule that gradually increases difficulty.

ScenarioType variants:
  OFFPEAK       — uniform low flow across all routes
  MORNING_PEAK  — heavy inbound (J0/J1/J2-side → J3/J4-side)
  EVENING_PEAK  — heavy outbound (J3/J4-side → J0/J1/J2-side)
  INCIDENT      — OFFPEAK base with one junction fully blocked for a window

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


# ──────────────────────────────────────────────
#  Network constants (from sumo/map.net.xml)
# ──────────────────────────────────────────────

# Fringe entry edges per traffic-light junction index (sorted TLS ID order)
_JUNCTION_FRINGE_IN: Dict[int, List[str]] = {
    0: ["-E5", "-E6"],
    1: ["-E7", "-E8"],
    2: ["-E9"],
    3: ["-E10", "-E11"],
    4: ["-E12", "-E13", "-E14"],
}

# OD pairs — inbound: J0/J1/J2-fringe sources → J3/J4-fringe destinations
_MAIN_INBOUND_OD: List[Tuple[str, str]] = [
    ("-E5", "E10"), ("-E5", "E11"), ("-E5", "E12"), ("-E5", "E13"),
    ("-E6", "E10"), ("-E6", "E12"), ("-E6", "E14"),
    ("-E7", "E10"), ("-E7", "E11"), ("-E7", "E14"),
    ("-E8", "E11"), ("-E8", "E12"), ("-E8", "E13"),
    ("-E9", "E10"), ("-E9", "E11"), ("-E9", "E12"), ("-E9", "E14"),
]

# OD pairs — outbound: J3/J4-fringe sources → J0/J1/J2-fringe destinations
_MAIN_OUTBOUND_OD: List[Tuple[str, str]] = [
    ("-E10", "E5"), ("-E10", "E7"), ("-E10", "E9"),
    ("-E11", "E5"), ("-E11", "E8"), ("-E11", "E9"),
    ("-E12", "E5"), ("-E12", "E7"), ("-E12", "E9"),
    ("-E13", "E6"), ("-E13", "E8"), ("-E13", "E9"),
    ("-E14", "E6"), ("-E14", "E7"), ("-E14", "E9"),
]

# OD pairs — local: short cross-traffic and intra-side routes
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

# All OD pairs combined (for OFFPEAK and INCIDENT base)
_ALL_OD: List[Tuple[str, str]] = _MAIN_INBOUND_OD + _MAIN_OUTBOUND_OD + _LOCAL_OD

# ── Curriculum schedule ──
# Each row: (min_episode_inclusive, weights [OFFPEAK, MORNING, EVENING, INCIDENT])
_CURRICULUM: List[Tuple[int, List[float]]] = [
    (800, [0.20, 0.25, 0.25, 0.30]),
    (500, [0.25, 0.30, 0.30, 0.15]),
    (200, [0.40, 0.30, 0.30, 0.00]),
    (0,   [1.00, 0.00, 0.00, 0.00]),
]
_SCENARIO_ORDER = [
    ScenarioType.OFFPEAK,
    ScenarioType.MORNING_PEAK,
    ScenarioType.EVENING_PEAK,
    ScenarioType.INCIDENT,
]


# ──────────────────────────────────────────────
#  ScenarioGenerator
# ──────────────────────────────────────────────

class ScenarioGenerator:
    """
    Generates randomized SUMO .rou.xml files per episode following a
    curriculum that increases traffic complexity over time.
    """

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

        # Parse valid edge IDs from the network file once at construction time
        self._valid_edges: set = self._parse_net_edges(net_file)

        # Validate all OD pairs against the network
        for from_e, to_e in _ALL_OD:
            self._check_edge(from_e)
            self._check_edge(to_e)

    # ── Public API ────────────────────────────

    def generate(self, scenario_type: ScenarioType, episode: int) -> str:
        """
        Randomise parameters for scenario_type, write a SUMO .rou.xml to
        output_dir/ep{episode:04d}.rou.xml, and return the path.

        Uses self.seed + episode as the numpy seed when self.seed is set,
        so the same (seed, episode) pair always produces identical output.
        """
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

        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")

        self._write_rou_xml(out_path, params["flows"])
        self.last_summary = self.summary(scenario_type, params)
        return str(out_path)

    def sample(self, episode: int) -> Tuple[ScenarioType, str]:
        """
        Sample a ScenarioType from the curriculum distribution for this
        episode, generate the route file, and return (type, route_file_path).
        """
        scenario_type = self.curriculum_schedule(episode)
        route_file = self.generate(scenario_type, episode)
        return scenario_type, route_file

    def curriculum_schedule(self, episode: int) -> ScenarioType:
        """
        Return a ScenarioType sampled from the curriculum distribution for
        the given episode. Implemented via a lookup table, not if-else chains.
        """
        # Pick the first row whose threshold is <= episode (rows sorted desc)
        weights = None
        for threshold, w in _CURRICULUM:
            if episode >= threshold:
                weights = w
                break
        assert weights is not None, "Curriculum table must have a row with threshold 0"

        rng = self._make_rng(episode + 10_000_000)  # separate stream from generate()
        choice_idx = int(rng.choice(len(_SCENARIO_ORDER), p=weights))
        return _SCENARIO_ORDER[choice_idx]

    def summary(self, scenario_type: ScenarioType, params: dict) -> str:
        """Return a one-line human-readable description of the episode."""
        ep = params.get("episode", 0)
        dur = params.get("episode_duration", 0)

        if scenario_type == ScenarioType.OFFPEAK:
            return (
                f"ep{ep:04d} | OFFPEAK | "
                f"flow={params['base_flow']:.0f} veh/hr | "
                f"duration={dur} steps"
            )
        elif scenario_type == ScenarioType.MORNING_PEAK:
            return (
                f"ep{ep:04d} | MORNING_PEAK | "
                f"main={params['main_flow']:.0f} veh/hr | "
                f"side={params['side_flow']:.0f} veh/hr | "
                f"duration={dur} steps"
            )
        elif scenario_type == ScenarioType.EVENING_PEAK:
            return (
                f"ep{ep:04d} | EVENING_PEAK | "
                f"main={params['main_flow']:.0f} veh/hr | "
                f"side={params['side_flow']:.0f} veh/hr | "
                f"duration={dur} steps"
            )
        elif scenario_type == ScenarioType.INCIDENT:
            return (
                f"ep{ep:04d} | INCIDENT | "
                f"flow={params['base_flow']:.0f} veh/hr | "
                f"junction=J{params['incident_junction']} | "
                f"onset={params['onset_step']}s | "
                f"block={params['block_duration']}s | "
                f"duration={dur} steps"
            )
        return f"ep{ep:04d} | {scenario_type.value} | duration={dur} steps"

    # ── Parameter generators ──────────────────

    def _gen_offpeak(self, rng, episode: int, duration: int, flow_horizon: int) -> dict:
        base_flow = float(rng.uniform(200, 500))
        per_od = base_flow / len(_ALL_OD)
        flows = [
            (from_e, to_e, 0, flow_horizon, per_od)
            for from_e, to_e in _ALL_OD
        ]
        return {
            "episode": episode,
            "episode_duration": duration,
            "base_flow": base_flow,
            "flows": flows,
        }

    def _gen_morning_peak(self, rng, episode: int, duration: int, flow_horizon: int) -> dict:
        main_flow = float(rng.uniform(800, 1200))
        side_flow = float(rng.uniform(100, 300))

        per_main = main_flow / len(_MAIN_INBOUND_OD)
        per_side = side_flow / (len(_MAIN_OUTBOUND_OD) + len(_LOCAL_OD))

        flows = []
        for from_e, to_e in _MAIN_INBOUND_OD:
            flows.append((from_e, to_e, 0, flow_horizon, per_main))
        for from_e, to_e in _MAIN_OUTBOUND_OD + _LOCAL_OD:
            flows.append((from_e, to_e, 0, flow_horizon, per_side))

        return {
            "episode": episode,
            "episode_duration": duration,
            "main_flow": main_flow,
            "side_flow": side_flow,
            "flows": flows,
        }

    def _gen_evening_peak(self, rng, episode: int, duration: int, flow_horizon: int) -> dict:
        main_flow = float(rng.uniform(800, 1200))
        side_flow = float(rng.uniform(100, 300))

        per_main = main_flow / len(_MAIN_OUTBOUND_OD)
        per_side = side_flow / (len(_MAIN_INBOUND_OD) + len(_LOCAL_OD))

        flows = []
        for from_e, to_e in _MAIN_OUTBOUND_OD:
            flows.append((from_e, to_e, 0, flow_horizon, per_main))
        for from_e, to_e in _MAIN_INBOUND_OD + _LOCAL_OD:
            flows.append((from_e, to_e, 0, flow_horizon, per_side))

        return {
            "episode": episode,
            "episode_duration": duration,
            "main_flow": main_flow,
            "side_flow": side_flow,
            "flows": flows,
        }

    def _gen_incident(self, rng, episode: int, duration: int, flow_horizon: int) -> dict:
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
                # Split flow around the incident window; resume after block through full horizon
                if onset_step > 0:
                    flows.append((from_e, to_e, 0, onset_step, per_od))
                if incident_end < flow_horizon:
                    flows.append((from_e, to_e, incident_end, flow_horizon, per_od))
            else:
                flows.append((from_e, to_e, 0, flow_horizon, per_od))

        return {
            "episode": episode,
            "episode_duration": duration,
            "base_flow": base_flow,
            "incident_junction": incident_junction,
            "onset_step": onset_step,
            "block_duration": block_duration,
            "flows": flows,
        }

    # ── Route file writer ─────────────────────

    def _write_rou_xml(
        self,
        out_path: Path,
        flows: List[Tuple[str, str, int, int, float]],
    ):
        """Write a valid SUMO .rou.xml. Validates all edge IDs before writing."""
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

    # ── Validation helpers ────────────────────

    def _check_edge(self, edge_id: str):
        if edge_id not in self._valid_edges:
            raise ValueError(
                f"Edge '{edge_id}' not found in net file: {self.net_file}"
            )

    @staticmethod
    def _parse_net_edges(net_file: str) -> set:
        """Return the set of non-internal edge IDs from a SUMO .net.xml."""
        if not os.path.exists(net_file):
            warnings.warn(
                f"Net file not found: {net_file}. Edge validation will be skipped.",
                UserWarning,
                stacklevel=3,
            )
            return set()

        tree = ET.parse(net_file)
        edges = set()
        for edge in tree.getroot().findall("edge"):
            eid = edge.get("id", "")
            if eid and not eid.startswith(":"):
                edges.add(eid)
        return edges

    # ── RNG ───────────────────────────────────

    def _make_rng(self, episode: int) -> np.random.Generator:
        if self.seed is not None:
            return np.random.default_rng(self.seed + episode)
        return np.random.default_rng()


# ──────────────────────────────────────────────
#  ScenarioEnvironment
# ──────────────────────────────────────────────

class ScenarioEnvironment:
    """
    Drop-in replacement for SumoEnvironment that injects a per-episode
    --route-files flag into the SUMO command, overriding the route file
    embedded in the .sumocfg.

    Importing this class does NOT import traci or SumoEnvironment.
    Both are imported lazily inside start().
    """

    def __init__(self, cfg):
        self._cfg = cfg
        self._route_file: Optional[str] = None
        self._env = None  # SumoEnvironment instance, created on first start()

    def set_route_file(self, path: str):
        """Set the route file that will be injected on the next start() call."""
        self._route_file = path

    def start(self, episode: int = 0):
        """
        Build the SUMO command (same structure as SumoEnvironment.start),
        inject --route-files if a route file is set, and start the simulation.
        """
        import traci as _traci

        SumoEnvironment = self._import_sumo_env()
        if self._env is None:
            self._env = SumoEnvironment(self._cfg)

        cfg = self._cfg
        cfg_path = os.path.abspath(cfg.sumo_cfg)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"\n{'=' * 60}\n"
                f"  SUMO konfigürasyon dosyası bulunamadı!\n"
                f"  Aranan yol : {cfg_path}\n"
                f"  Çalışma dizini: {os.getcwd()}\n"
                f"{'=' * 60}"
            )

        sumo_binary = "sumo-gui" if cfg.gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", cfg_path,
            "--step-length", str(cfg.sumo_step_length),
            "--waiting-time-memory", "1000",
            "--no-warnings", "true",
            "--random",
            "--seed", str(cfg.seed + episode),
        ]

        if self._route_file and os.path.exists(self._route_file):
            sumo_cmd.extend(["--route-files", self._route_file])

        _traci.start(sumo_cmd)

        # Sync inner SumoEnvironment state so all delegated methods work correctly
        self._env._step_count = 0
        self._env._episode_count = episode
        self._env.tls_ids = sorted(_traci.trafficlight.getIDList())
        self._env.num_nodes = len(self._env.tls_ids)

        if self._env.num_nodes == 0:
            raise RuntimeError(
                "No traffic-light nodes found in SUMO network. "
                "Check that map.net.xml contains <tlLogic> elements."
            )

        for _ in range(cfg.warmup_steps):
            _traci.simulationStep()
            self._env._step_count += 1

    def close(self):
        if self._env is not None:
            self._env.close()

    def __getattr__(self, name: str):
        """Delegate all other attribute access to the inner SumoEnvironment."""
        env = object.__getattribute__(self, "_env")
        if env is None:
            raise AttributeError(
                f"Attribute '{name}' accessed on ScenarioEnvironment before "
                f"start() was called."
            )
        return getattr(env, name)

    @staticmethod
    def _import_sumo_env():
        """Import SumoEnvironment lazily so this module is importable without SUMO."""
        sd = Path(__file__).resolve().parent
        pr = sd.parent
        sys.path.insert(0, str(sd))
        sys.path.insert(0, str(pr))
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


# ──────────────────────────────────────────────
#  Standalone test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        gen = ScenarioGenerator(
            net_file=_DEFAULT_NET_FILE,
            output_dir=tmpdir,
            seed=0,
        )

        for ep, stype in enumerate([
            ScenarioType.OFFPEAK,
            ScenarioType.MORNING_PEAK,
            ScenarioType.EVENING_PEAK,
            ScenarioType.INCIDENT,
        ]):
            path = gen.generate(stype, episode=ep)
            print(gen.last_summary)
            assert os.path.exists(path), f"Route file not written: {path}"

        print("\nCurriculum schedule samples:")
        for ep in [0, 100, 200, 350, 500, 650, 800, 950]:
            stype = gen.curriculum_schedule(ep)
            print(f"  ep={ep:>4d} → {stype.value}")

        print("\nAll assertions passed.")
