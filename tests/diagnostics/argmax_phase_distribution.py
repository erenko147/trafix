"""
Argmax phase-distribution diagnostic
====================================
Single source of truth for "is the greedy policy fixed?".

Runs TraFixV6 + the *production* RuleGovernor (same params as
backend/main.py::load_model) on a route file, choosing each junction's phase by
governor.apply(...) followed by pure ARGMAX, with **no** starvation overrides
(the safety net is deliberately disabled here so we measure the raw policy the
way it would behave if the overrides never fired). Phases are actuated through
the standard 3-step yellow transition, exactly like tests/utils/sumo_runner.py.

It prints, per junction, the % of decisions spent on each of the 6 phases, and
flags any (junction, phase) that has *standing demand* but is chosen < 1% of the
time — i.e. a starved movement.

Usage:
  .venv/bin/python tests/diagnostics/argmax_phase_distribution.py
  .venv/bin/python tests/diagnostics/argmax_phase_distribution.py \
        --route tests/scenarios/type1_medium.rou.xml \
        --checkpoint trafix_v6/checkpoints/trafix_v6_final.pt
"""

import argparse
import os
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List

# ── Project path setup ────────────────────────────────────────────────────────
_DIAG_DIR = Path(__file__).resolve().parent
_TESTS_DIR = _DIAG_DIR.parent
_PROJECT_ROOT = _TESTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── SUMO TraCI ────────────────────────────────────────────────────────────────
if "SUMO_HOME" in os.environ:
    _sumo_tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if _sumo_tools not in sys.path:
        sys.path.append(_sumo_tools)
else:
    for _candidate in ["/usr/share/sumo/tools", "/usr/local/share/sumo/tools"]:
        if os.path.isdir(_candidate):
            sys.path.append(_candidate)
            os.environ.setdefault("SUMO_HOME", os.path.dirname(_candidate))
            break

import traci  # noqa: E402
import torch  # noqa: E402

# Reuse the real pipeline helpers so the diagnostic matches production exactly.
from tests.utils.sumo_runner import (  # noqa: E402
    _get_observations,
    _MODEL_TO_SUMO_GREEN,
    _YELLOW_STEPS,
    _NET_FILE,
    _NUM_JUNCTIONS,
    _NUM_PHASES,
    _T_WINDOW,
)
from backend.ai.trafix_v2 import parse_sumo_observations  # noqa: E402
from trafix_v6.trafix_v6 import TraFixV6  # noqa: E402
from trafix_v6.rule_governor import RuleGovernor  # noqa: E402

_DEFAULT_CKPT = str(_PROJECT_ROOT / "trafix_v6" / "checkpoints" / "trafix_v6_final.pt")
_DEFAULT_ROUTE = str(_TESTS_DIR / "scenarios" / "type1_low.rou.xml")

_PHASE_NAMES = ["NS-thru", "N-left", "S-left", "EW-thru", "E-left", "W-left"]

# Per-phase served-movement demand (vehicles). Mirrors RuleGovernor._pressure_bonus.
def _phase_demand(o: Dict) -> List[float]:
    return [
        o["north_through"] + o["south_through"],   # 0 NS-through
        o["north_left"],                            # 1 N-left
        o["south_left"],                            # 2 S-left
        o["east_through"] + o["west_through"],      # 3 EW-through
        o["east_left"],                             # 4 E-left
        o["west_left"],                             # 5 W-left
    ]


def run_diagnostic(route_file, checkpoint_path, seed, sim_duration,
                   warmup_steps, decision_interval,
                   pressure_thresh=0.12, pressure_boost=1.0, flicker_penalty=3.0):
    device = torch.device("cpu")
    model = TraFixV6().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    # Production governor params (backend/main.py::load_model). pressure_thresh /
    # pressure_boost / flicker_penalty are exposed so Step 5 (governor tuning) can be
    # A/B-tested against a fixed checkpoint without retraining.
    governor = RuleGovernor(
        num_junctions=_NUM_JUNCTIONS, num_phases=_NUM_PHASES,
        min_green_s=10.0, max_green_s=90.0,
        flicker_window=2, flicker_penalty=flicker_penalty,
        pressure_boost=pressure_boost, pressure_thresh=pressure_thresh,
    )

    cmd = [
        "sumo",
        "--net-file", _NET_FILE,
        "--route-files", route_file,
        "--seed", str(seed),
        "--no-warnings", "true",
        "--step-length", "1.0",
        "--begin", "0", "--end", str(sim_duration),
        "--waiting-time-memory", "1000",
        "--time-to-teleport", "-1",
        "--time-to-teleport.highways", "-1",
    ]
    traci.start(cmd)
    tls_ids = sorted(traci.trafficlight.getIDList())

    step = 0
    for _ in range(warmup_steps):
        traci.simulationStep()
        step += 1
    phase_held_since = {t: step for t in tls_ids}

    obs0 = _get_observations(tls_ids, phase_held_since, step)
    x0 = parse_sumo_observations(obs0, device=device)
    window = deque([x0.detach()] * _T_WINDOW, maxlen=_T_WINDOW)
    governor.reset()

    # Per-junction counters
    phase_counts = [[0] * _NUM_PHASES for _ in range(_NUM_JUNCTIONS)]
    # Per (junction, phase): sum of served-movement demand over all decisions, and
    # count of decisions where that movement had >=1 waiting vehicle.
    demand_sum = [[0.0] * _NUM_PHASES for _ in range(_NUM_JUNCTIONS)]
    demand_present = [[0] * _NUM_PHASES for _ in range(_NUM_JUNCTIONS)]
    total_decisions = 0
    switches = 0            # chosen phase != previous chosen phase (over-switch guard)
    prev_chosen = [None] * _NUM_JUNCTIONS

    pending_target: Dict[str, int] = {}
    yellow_remaining: Dict[str, int] = {}
    next_decision = step + decision_interval

    while step < sim_duration:
        if traci.simulation.getMinExpectedNumber() == 0:
            break

        if step >= next_decision:
            obs = _get_observations(tls_ids, phase_held_since, step)
            x = parse_sumo_observations(obs, device=device)
            window.append(x.detach())
            obs_input = torch.stack(list(window)).unsqueeze(0).to(device)

            with torch.no_grad():
                logits_list, _ = model.forward(obs_input)
                obs_last = obs_input[0, -1]
                masked = governor.apply(logits_list, obs_last)
                actions_1d = torch.stack(
                    [torch.argmax(l, dim=-1).reshape(()) for l in masked]
                )  # ARGMAX, NO starvation overrides
            governor.update_state(actions_1d)

            total_decisions += 1
            for i in range(_NUM_JUNCTIONS):
                mp = int(actions_1d[i].item()) % _NUM_PHASES
                phase_counts[i][mp] += 1
                if prev_chosen[i] is not None and prev_chosen[i] != mp:
                    switches += 1
                prev_chosen[i] = mp
                dem = _phase_demand(obs[i])
                for p in range(_NUM_PHASES):
                    demand_sum[i][p] += dem[p]
                    if dem[p] >= 1.0:
                        demand_present[i][p] += 1

            # Actuate via yellow transition (same as sumo_runner)
            for i, tls_id in enumerate(tls_ids):
                if yellow_remaining.get(tls_id, 0) > 0:
                    continue
                mp = int(actions_1d[i].item()) % _NUM_PHASES
                target_sumo = _MODEL_TO_SUMO_GREEN[mp]
                current_sumo = traci.trafficlight.getPhase(tls_id)
                if target_sumo == current_sumo:
                    continue
                pending_target[tls_id] = target_sumo
                yellow_remaining[tls_id] = _YELLOW_STEPS
                yellow_phase = current_sumo + 1 if current_sumo % 2 == 0 else current_sumo
                traci.trafficlight.setPhase(tls_id, yellow_phase)

            next_decision = step + decision_interval

        for tls_id in list(yellow_remaining.keys()):
            rem = yellow_remaining[tls_id] - 1
            yellow_remaining[tls_id] = rem
            if rem <= 0:
                target = pending_target.pop(tls_id, None)
                if target is not None:
                    traci.trafficlight.setPhase(tls_id, target)
                    phase_held_since[tls_id] = step

        traci.simulationStep()
        step += 1

    traci.close()

    return {
        "route": route_file,
        "checkpoint": checkpoint_path,
        "total_decisions": total_decisions,
        "phase_counts": phase_counts,
        "demand_sum": demand_sum,
        "demand_present": demand_present,
        "switches": switches,
        "switch_rate": switches / max(total_decisions * _NUM_JUNCTIONS, 1),
        "governor": {"pressure_thresh": pressure_thresh,
                     "pressure_boost": pressure_boost,
                     "flicker_penalty": flicker_penalty},
    }


def print_report(result):
    pc = result["phase_counts"]
    n = max(result["total_decisions"], 1)
    print(f"\nRoute      : {result['route']}")
    print(f"Checkpoint : {result['checkpoint']}")
    print(f"Decisions  : {result['total_decisions']} per junction")
    g = result.get("governor", {})
    print(f"Governor   : pressure_thresh={g.get('pressure_thresh')} "
          f"pressure_boost={g.get('pressure_boost')} flicker_penalty={g.get('flicker_penalty')}")
    print(f"Switch rate: {result.get('switch_rate', 0.0)*100:.1f}% of decisions change phase\n")

    header = "  junction   " + "  ".join(f"{name:>7s}" for name in _PHASE_NAMES)
    print(header)
    for j in range(_NUM_JUNCTIONS):
        pcts = [100.0 * pc[j][p] / n for p in range(_NUM_PHASES)]
        row = f"  J{j}        " + "  ".join(f"{v:6.0f}%" for v in pcts)
        print(row)

    # Starvation flags: phase has standing demand (present >=10% of decisions)
    # but is chosen < 1% of the time.
    print("\nStarved movements (standing demand, chosen <1% of decisions):")
    any_flag = False
    for j in range(_NUM_JUNCTIONS):
        for p in range(_NUM_PHASES):
            pct_chosen = 100.0 * pc[j][p] / n
            pct_demand = 100.0 * result["demand_present"][j][p] / n
            if pct_demand >= 10.0 and pct_chosen < 1.0:
                any_flag = True
                print(f"  J{j} phase {p} ({_PHASE_NAMES[p]}): "
                      f"demand present {pct_demand:.0f}% of decisions, "
                      f"chosen {pct_chosen:.2f}%")
    if not any_flag:
        print("  (none) — no movement with standing demand is starved.")

    # Lock summary
    print("\nPhase-lock summary (max single-phase share per junction):")
    locked = False
    for j in range(_NUM_JUNCTIONS):
        pcts = [100.0 * pc[j][p] / n for p in range(_NUM_PHASES)]
        top = max(range(_NUM_PHASES), key=lambda p: pcts[p])
        flag = " <-- LOCKED" if pcts[top] > 70.0 else ""
        if pcts[top] > 70.0:
            locked = True
        print(f"  J{j}: {_PHASE_NAMES[top]} {pcts[top]:.0f}%{flag}")
    print(f"\nOVERALL: {'COLLAPSED (a junction locks >70%)' if locked else 'OK (no junction locks >70%)'}")


def main():
    p = argparse.ArgumentParser(description="Argmax phase-distribution diagnostic")
    p.add_argument("--route", default=_DEFAULT_ROUTE)
    p.add_argument("--checkpoint", default=_DEFAULT_CKPT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sim-duration", type=int, default=3600)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--decision-interval", type=int, default=10)
    p.add_argument("--pressure-thresh", type=float, default=0.12,
                   help="governor pressure_thresh (Step 5 A/B; production = 0.35)")
    p.add_argument("--pressure-boost", type=float, default=1.0)
    p.add_argument("--flicker-penalty", type=float, default=3.0)
    args = p.parse_args()

    result = run_diagnostic(
        route_file=args.route,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        sim_duration=args.sim_duration,
        warmup_steps=args.warmup,
        decision_interval=args.decision_interval,
        pressure_thresh=args.pressure_thresh,
        pressure_boost=args.pressure_boost,
        flicker_penalty=args.flicker_penalty,
    )
    print_report(result)


if __name__ == "__main__":
    main()
