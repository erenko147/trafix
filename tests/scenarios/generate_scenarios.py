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


# ── Unseen scenarios (out-of-distribution — never in training curriculum) ────
#
# Training saw: OFFPEAK 200-500 veh/hr, MORNING/EVENING_PEAK 800-1200 veh/hr
# main flow, INCIDENT on OFFPEAK base, PULSE step-function burst.
# The six scenarios below differ in flow magnitude, spatial pattern, temporal
# shape, or combination of stressors that training never mixed together.

def gen_unseen_supersaturation(horizon: int = 3600):
    """
    150%+ capacity — 1 500 veh/hr uniform across all OD pairs.
    Training OFFPEAK max was 500, MORNING_PEAK total ~950. This exceeds both,
    stressing every junction simultaneously with no directional relief valve.
    """
    rate = 1500.0 / len(_ALL_OD)
    flows = [(f, t, 0, horizon, rate) for f, t in _ALL_OD]
    _write_rou(_OUT / "unseen_supersaturation.rou.xml", flows, horizon)


def gen_unseen_stadium_exit(horizon: int = 3600):
    """
    Stadium-exit: 90% of demand concentrated on J4's 3 fringe edges only.
    Training MORNING_PEAK spread 800-1200 veh/hr across 17 inbound OD pairs
    (~47-70 veh/hr each). This puts ~400 veh/hr on just 3 edges — a spatial
    concentration the model has never seen.
    """
    j4_fringes = set(_JUNCTION_FRINGE_IN[4])          # {"-E12", "-E13", "-E14"}
    stadium_od    = [(f, t) for f, t in _ALL_OD if f in j4_fringes]
    background_od = [(f, t) for f, t in _ALL_OD if f not in j4_fringes]
    stadium_rate = 1200.0 / len(stadium_od)
    bg_rate      =  100.0 / max(len(background_od), 1)
    flows  = [(f, t, 0, horizon, stadium_rate) for f, t in stadium_od]
    flows += [(f, t, 0, horizon, bg_rate)      for f, t in background_od]
    _write_rou(_OUT / "unseen_stadium_exit.rou.xml", flows, horizon)


def gen_unseen_peak_plus_incident(horizon: int = 3600):
    """
    Morning-peak demand with J1 blocked mid-simulation (600–1 800 s).
    Training INCIDENT always used OFFPEAK base (~200-500 veh/hr total).
    Training MORNING_PEAK never had a simultaneous junction closure.
    This is the only scenario with both stressors active at the same time.
    J1 (not J2 used in type2_incident) adds further novelty.
    """
    blocked = set(_JUNCTION_FRINGE_IN[1])        # J1: {"-E7", "-E8"}
    onset, end_block = 600, 1800
    in_rate = 700.0 / len(_MAIN_INBOUND_OD)
    bg_rate = 200.0 / (len(_MAIN_OUTBOUND_OD) + len(_LOCAL_OD))
    flows = []
    for od_list, rate in [(_MAIN_INBOUND_OD, in_rate),
                           (_MAIN_OUTBOUND_OD + _LOCAL_OD, bg_rate)]:
        for f, t in od_list:
            if f in blocked:
                if onset > 0:
                    flows.append((f, t, 0, onset, rate))
                if end_block < horizon:
                    flows.append((f, t, end_block, horizon, rate))
            else:
                flows.append((f, t, 0, horizon, rate))
    _write_rou(_OUT / "unseen_peak_plus_incident.rou.xml", flows, horizon)


def gen_unseen_oscillating(horizon: int = 3600):
    """
    Direction flips every 10 minutes: inbound heavy → outbound heavy → repeat.
    Training flows were always sustained in one direction per episode.
    The model must continuously re-adapt its phase strategy — never trained for this.
    """
    window   = 600      # 10-minute alternation window
    in_rate  = 700.0 / len(_MAIN_INBOUND_OD)
    out_rate = 700.0 / len(_MAIN_OUTBOUND_OD)
    bg_rate  = 100.0 / len(_LOCAL_OD)
    flows = []
    t = 0
    inbound_turn = True
    while t < horizon:
        end = min(t + window, horizon)
        if inbound_turn:
            flows += [(f, e, t, end, in_rate)  for f, e in _MAIN_INBOUND_OD]
            flows += [(f, e, t, end, bg_rate)  for f, e in _LOCAL_OD]
        else:
            flows += [(f, e, t, end, out_rate) for f, e in _MAIN_OUTBOUND_OD]
            flows += [(f, e, t, end, bg_rate)  for f, e in _LOCAL_OD]
        t += window
        inbound_turn = not inbound_turn
    _write_rou(_OUT / "unseen_oscillating.rou.xml", flows, horizon)


def gen_unseen_tidal_ramp(horizon: int = 3600):
    """
    Three-phase tidal pattern: quiet → peak inbound → decay.
    Training demand changes were step-functions (instant onset/offset).
    This uses 20-minute phases to simulate a smooth real-world commute arc.
    The peak (1 000 veh/hr inbound) is also above training's MORNING_PEAK
    flow range, adding magnitude novelty on top of temporal novelty.
    """
    flows = []
    # Phase 1: quiet background (0–1 200 s)
    r1 = 200.0 / len(_ALL_OD)
    flows += [(f, t, 0, 1200, r1) for f, t in _ALL_OD]
    # Phase 2: hard peak inbound (1 200–2 400 s)
    in_rate = 1000.0 / len(_MAIN_INBOUND_OD)
    bg_rate =  150.0 / (len(_MAIN_OUTBOUND_OD) + len(_LOCAL_OD))
    flows += [(f, t, 1200, 2400, in_rate) for f, t in _MAIN_INBOUND_OD]
    flows += [(f, t, 1200, 2400, bg_rate) for f, t in _MAIN_OUTBOUND_OD + _LOCAL_OD]
    # Phase 3: decay (2 400–3 600 s)
    r3 = 350.0 / len(_ALL_OD)
    flows += [(f, t, 2400, horizon, r3) for f, t in _ALL_OD]
    _write_rou(_OUT / "unseen_tidal_ramp.rou.xml", flows, horizon)


def gen_unseen_bidirectional_peak(horizon: int = 3600):
    """
    Heavy inbound AND outbound simultaneously at equal rates (~1 100 veh/hr total).
    Training always had one direction dominant by a large margin; the opposite
    direction was always suppressed to 100-300 veh/hr background.
    Equal opposing flows create deadlock pressure at every junction — the model
    must balance NS vs EW phases without a clear directional priority signal.
    """
    in_rate  = 550.0 / len(_MAIN_INBOUND_OD)
    out_rate = 550.0 / len(_MAIN_OUTBOUND_OD)
    loc_rate = 100.0 / len(_LOCAL_OD)
    flows  = [(f, t, 0, horizon, in_rate)  for f, t in _MAIN_INBOUND_OD]
    flows += [(f, t, 0, horizon, out_rate) for f, t in _MAIN_OUTBOUND_OD]
    flows += [(f, t, 0, horizon, loc_rate) for f, t in _LOCAL_OD]
    _write_rou(_OUT / "unseen_bidirectional_peak.rou.xml", flows, horizon)


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
    print("  7 in-distribution scenarios written.")

    print("Generating unseen (OOD) scenarios...")
    gen_unseen_supersaturation()
    gen_unseen_stadium_exit()
    gen_unseen_peak_plus_incident()
    gen_unseen_oscillating()
    gen_unseen_tidal_ramp()
    gen_unseen_bidirectional_peak()
    print("  6 unseen scenarios written.")
    print("Done. 13 route files total in tests/scenarios/")
