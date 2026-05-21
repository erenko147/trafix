"""
Generate deterministic SUMO route files for the test suite.

Run once from the project root before running the test suite:
    python tests/scenarios/generate_scenarios.py

All flows are exact values (no random ranges) so the output is byte-identical
across runs.  Route files are committed to the repo and not regenerated at
test time.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from trafix_v6.scenario_generator import (
    _MAIN_INBOUND_OD, _MAIN_OUTBOUND_OD, _LOCAL_OD, _ALL_OD,
    _JUNCTION_FRINGE_IN,
)

_OUT = _HERE


# ── route-file writer ────────────────────────────────────────────────────────

def _write_rou(path: Path, flows: list, horizon: int = 3600):
    """
    flows: list of (from_edge, to_edge, begin, end, veh_per_hr)
    Writes a SUMO .rou.xml file with fixed vehicle type and flow entries.
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        ' xsi:noNamespaceSchemaLocation='
        '"http://sumo.dlr.de/xsd/routes_file.xsd">',
        '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5"'
        ' length="5" minGap="2.5" maxSpeed="50" emissionClass="HBEFA3/PC_G_EU4"/>',
    ]
    for idx, (fe, te, begin, end, rate) in enumerate(flows):
        if rate < 0.01 or end <= begin:
            continue
        safe_from = fe.lstrip("-")
        safe_to = te.lstrip("-")
        fid = f"f_{safe_from}_{safe_to}_{idx}"
        lines.append(
            f'    <flow id="{fid}" from="{fe}" to="{te}"'
            f' begin="{begin}" end="{end}"'
            f' vehsPerHour="{rate:.4f}"'
            f' departLane="best" departSpeed="max"/>'
        )
    lines.append("</routes>")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {path.name}")


# ── Test Type 1 — variable traffic load ─────────────────────────────────────

def gen_type1_low(horizon: int = 3600):
    """~30% capacity — uniform 200 veh/hr across all 66 OD pairs."""
    total = 200.0
    rate = total / len(_ALL_OD)
    flows = [(f, t, 0, horizon, rate) for f, t in _ALL_OD]
    _write_rou(_OUT / "type1_low.rou.xml", flows, horizon)


def gen_type1_medium(horizon: int = 3600):
    """~60% capacity — uniform 500 veh/hr across all 66 OD pairs."""
    total = 500.0
    rate = total / len(_ALL_OD)
    flows = [(f, t, 0, horizon, rate) for f, t in _ALL_OD]
    _write_rou(_OUT / "type1_medium.rou.xml", flows, horizon)


def gen_type1_high(horizon: int = 3600):
    """~90%+ capacity — morning-peak directional (600 inbound + 150 out + 200 local)."""
    flows = []
    in_rate  = 600.0 / len(_MAIN_INBOUND_OD)
    out_rate = 150.0 / len(_MAIN_OUTBOUND_OD)
    loc_rate = 200.0 / len(_LOCAL_OD)
    flows += [(f, t, 0, horizon, in_rate)  for f, t in _MAIN_INBOUND_OD]
    flows += [(f, t, 0, horizon, out_rate) for f, t in _MAIN_OUTBOUND_OD]
    flows += [(f, t, 0, horizon, loc_rate) for f, t in _LOCAL_OD]
    _write_rou(_OUT / "type1_high.rou.xml", flows, horizon)


# ── Test Type 2 — scenario robustness ───────────────────────────────────────

def gen_type2_morning_peak(horizon: int = 3600):
    """Heavy inbound (700 veh/hr) + 200 veh/hr background."""
    main_rate = 700.0 / len(_MAIN_INBOUND_OD)
    bg_rate   = 200.0 / (len(_MAIN_OUTBOUND_OD) + len(_LOCAL_OD))
    flows  = [(f, t, 0, horizon, main_rate) for f, t in _MAIN_INBOUND_OD]
    flows += [(f, t, 0, horizon, bg_rate)   for f, t in _MAIN_OUTBOUND_OD + _LOCAL_OD]
    _write_rou(_OUT / "type2_morning_peak.rou.xml", flows, horizon)


def gen_type2_evening_peak(horizon: int = 3600):
    """Heavy outbound (700 veh/hr) + 200 veh/hr background."""
    main_rate = 700.0 / len(_MAIN_OUTBOUND_OD)
    bg_rate   = 200.0 / (len(_MAIN_INBOUND_OD) + len(_LOCAL_OD))
    flows  = [(f, t, 0, horizon, main_rate) for f, t in _MAIN_OUTBOUND_OD]
    flows += [(f, t, 0, horizon, bg_rate)   for f, t in _MAIN_INBOUND_OD + _LOCAL_OD]
    _write_rou(_OUT / "type2_evening_peak.rou.xml", flows, horizon)


def gen_type2_incident(horizon: int = 3600):
    """OFFPEAK 500 veh/hr, J2 fringe blocked 300–600 s."""
    base_rate = 500.0 / len(_ALL_OD)
    blocked = set(_JUNCTION_FRINGE_IN[2])    # {"-E9"}
    onset, end_block = 300, 600
    flows = []
    for f, t in _ALL_OD:
        if f in blocked:
            if onset > 0:
                flows.append((f, t, 0, onset, base_rate))
            if end_block < horizon:
                flows.append((f, t, end_block, horizon, base_rate))
        else:
            flows.append((f, t, 0, horizon, base_rate))
    _write_rou(_OUT / "type2_incident.rou.xml", flows, horizon)


def gen_type2_pulse(horizon: int = 3600):
    """Quiet 0–300 s → inbound burst 300–900 s → quiet 900–3600 s."""
    quiet_rate = 50.0 / len(_ALL_OD)
    burst_rate = 600.0 / len(_MAIN_INBOUND_OD)
    flows = []
    flows += [(f, t, 0,   300, quiet_rate) for f, t in _ALL_OD]
    flows += [(f, t, 300, 900, burst_rate) for f, t in _MAIN_INBOUND_OD]
    flows += [(f, t, 900, horizon, quiet_rate) for f, t in _ALL_OD]
    _write_rou(_OUT / "type2_pulse.rou.xml", flows, horizon)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating test scenario route files...")
    gen_type1_low()
    gen_type1_medium()
    gen_type1_high()
    gen_type2_morning_peak()
    gen_type2_evening_peak()
    gen_type2_incident()
    gen_type2_pulse()
    print("Done. 7 route files written to tests/scenarios/")
