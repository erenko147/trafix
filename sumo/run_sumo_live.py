import os
import sys
import time
import logging
import argparse
import requests
from collections import defaultdict
from pathlib import Path

# SUMO_HOME check
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("ERROR: SUMO_HOME not set.")

import traci
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="traci")

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("sumocfg", nargs="?", default=None)
_parser.add_argument("--no-gui", action="store_true")
_args, _ = _parser.parse_known_args()

_api_port = int(os.environ.get("TRAFIX_API_PORT", "8000"))
API_URL = f"http://127.0.0.1:{_api_port}/telemetry"

# ── Phase mapping ─────────────────────────────────────────────────────────────
MODEL_TO_SUMO_GREEN = {0: 0, 1: 2, 2: 4, 3: 6, 4: 8, 5: 10}
SUMO_TO_MODEL       = {0:0,1:0, 2:1,3:1, 4:2,5:2, 6:3,7:3, 8:4,9:4, 10:5,11:5}

PHASE_NAMES = {
    0: "NS-through",
    1: "N-left",
    2: "S-left",
    3: "EW-through",
    4: "E-left",
    5: "W-left",
}

MIN_GREEN_THROUGH = 10
MIN_GREEN_LEFT    = 8
YELLOW_STEPS      = 3
DECISION_INTERVAL = 10

LANE_TYPE = {0: "right", 1: "through", 2: "left"}

# ── Logging setup ─────────────────────────────────────────────────────────────
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "decisions_live.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(str(_LOG_FILE), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("decisions")


def _classify_edge_direction(edge_id: str, jx: float, jy: float) -> str:
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


def build_intersection_map():
    intersection_map = {}
    for tls_id in traci.trafficlight.getIDList():
        jx, jy = traci.junction.getPosition(tls_id)
        intersection_map[tls_id] = {"jx": jx, "jy": jy}
    return intersection_map


def collect_lane_obs(tls_id: str, jx: float, jy: float) -> dict:
    counts = {
        "north_left": 0, "north_through": 0, "north_right": 0,
        "south_left": 0, "south_through": 0, "south_right": 0,
        "east_left":  0, "east_through":  0, "east_right":  0,
        "west_left":  0, "west_through":  0, "west_right":  0,
    }
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)
    seen_lanes = set()
    for link in controlled_links:
        if not link:
            continue
        from_lane = link[0][0]
        if from_lane in seen_lanes:
            continue
        seen_lanes.add(from_lane)
        edge_id  = from_lane.rsplit("_", 1)[0]
        lane_idx = int(from_lane.rsplit("_", 1)[1])
        lane_type = LANE_TYPE.get(lane_idx, "through")
        direction = _classify_edge_direction(edge_id, jx, jy)
        if direction and lane_type:
            key = f"{direction}_{lane_type}"
            if key in counts:
                try:
                    counts[key] += traci.lane.getLastStepVehicleNumber(from_lane)
                except Exception:
                    pass
    return counts


def _log_summary(tls_ids, phase_counts, decision_count, step):
    """Print a phase-distribution table for all junctions."""
    log.info("")
    log.info(f"  ── Decision summary at step {step} "
             f"(total decisions: {decision_count}) ──")
    header = f"  {'Junction':<10}" + "".join(f"{PHASE_NAMES[p]:>14}" for p in range(6))
    log.info(header)
    log.info("  " + "-" * (10 + 14 * 6))
    for tls_id in tls_ids:
        counts = phase_counts[tls_id]
        total  = sum(counts.values()) or 1
        row    = f"  {tls_id:<10}"
        for p in range(6):
            pct = counts[p] / total * 100
            row += f"{counts[p]:>6}({pct:>4.0f}%)"
        log.info(row)
    log.info("")


def main():
    sumo_cfg = _args.sumocfg or os.path.join(os.path.dirname(__file__), "demo.sumocfg")
    if not os.path.exists(sumo_cfg):
        print(f"ERROR: SUMO config not found: {sumo_cfg}")
        sys.exit(1)

    log.info(f"[SUMO] TraFix v6 live  API={API_URL}  log={_LOG_FILE}")

    sumo_bin = "sumo" if _args.no_gui else "sumo-gui"
    traci.start([sumo_bin, "-c", sumo_cfg, "--time-to-teleport", "-1"])

    intersection_map = build_intersection_map()
    tls_ids = sorted(intersection_map.keys())

    log.info(f"  Junctions: {tls_ids}")
    log.info(f"  Phase key: " +
             " | ".join(f"{p}={PHASE_NAMES[p]}" for p in range(6)))
    log.info("")

    # Per-junction phase selection counters
    phase_counts    = {tls_id: defaultdict(int) for tls_id in tls_ids}
    decision_count  = 0

    step = 0
    last_phase_change_step = {tls_id: -MIN_GREEN_THROUGH for tls_id in tls_ids}
    pending_targets:  dict = {}
    yellow_remaining: dict = {}

    # Track last logged phase per junction to detect stagnation
    last_logged_phase = {tls_id: -1 for tls_id in tls_ids}

    # Through-phase starvation override: if a junction hasn't had NS-through (0)
    # or EW-through (3) in the last STARVE_LIMIT decisions, force one.
    # Through phases need exposure or vehicles pile up unserved.
    STARVE_LIMIT = 8   # ~80 simulated seconds without any through phase
    decisions_since_through = {tls_id: 0 for tls_id in tls_ids}

    # Direction-specific starvation: prevents one through-phase from monopolising
    # when the other direction has vehicles waiting. Fires even if the AI keeps
    # picking a through-phase — the original STARVE_LIMIT only catches left-turn
    # monopolies. 10 decisions × 10-step interval = 100 simulated seconds.
    DIRECTION_STARVE_LIMIT = 10
    decisions_since_ns = {tls_id: 0 for tls_id in tls_ids}
    decisions_since_ew = {tls_id: 0 for tls_id in tls_ids}

    # Left-turn starvation: no governor override ever forces left-turn phases, so
    # the model can starve them indefinitely. Track decisions since each left-turn
    # phase was last served; force it when vehicles are waiting too long.
    # 15 decisions × 10-step interval = 150 simulated seconds max wait.
    LEFT_STARVE_LIMIT = 15
    # phase → obs keys that measure demand for that movement
    _LEFT_DEMAND_KEYS = {
        1: ["north_left"],
        2: ["south_left"],
        4: ["east_left"],
        5: ["west_left"],
    }
    decisions_since_left = {tls_id: {p: 0 for p in (1, 2, 4, 5)} for tls_id in tls_ids}

    # Fixed-time fallback: if the backend fails for FALLBACK_THRESHOLD consecutive
    # decision cycles, cycle NS-through / EW-through on a fixed timer so lights
    # never freeze when the API is down.
    FALLBACK_THRESHOLD = 3   # consecutive failures before fallback activates
    FALLBACK_CYCLE     = 40  # simulated seconds per phase in fallback mode
    api_consecutive_failures = 0

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            step += 1

            # Advance yellow transitions
            for tls_id, rem in list(yellow_remaining.items()):
                rem -= 1
                if rem <= 0:
                    try:
                        traci.trafficlight.setPhase(tls_id, pending_targets.pop(tls_id))
                    except Exception:
                        pass
                    yellow_remaining.pop(tls_id, None)
                else:
                    yellow_remaining[tls_id] = rem

            if step % DECISION_INTERVAL != 0:
                continue

            batch_payload = {"step": step, "intersections": []}

            for i, tls_id in enumerate(tls_ids):
                jinfo = intersection_map[tls_id]
                jx, jy = jinfo["jx"], jinfo["jy"]

                lane_counts = collect_lane_obs(tls_id, jx, jy)
                total_queue = sum(lane_counts.values())

                try:
                    curr_sumo_phase = int(traci.trafficlight.getPhase(tls_id))
                    api_phase       = SUMO_TO_MODEL.get(curr_sumo_phase, 0)
                except Exception:
                    api_phase, curr_sumo_phase = 0, 0

                # Track duration manually — setPhase() resets SUMO's internal
                # timer every decision interval, so the SUMO-derived elapsed time
                # never exceeds ~10s. last_phase_change_step is only updated on
                # real switches, so this correctly reflects actual hold time.
                phase_duration = float(
                    step - last_phase_change_step.get(tls_id, -MIN_GREEN_THROUGH)
                )

                batch_payload["intersections"].append({
                    "intersection_id": i,
                    **lane_counts,
                    "queue_length":  min(total_queue * 1.5, 200.0),
                    "current_phase": api_phase,
                    "phase_duration": phase_duration,
                })

            # Log per-lane observation every 50 decisions for diagnostics
            if decision_count % 50 == 0:
                log.info("  [OBS SNAPSHOT]")
                for item in batch_payload["intersections"]:
                    jid = tls_ids[item["intersection_id"]]
                    log.info(
                        f"    {jid}  phase={PHASE_NAMES[item['current_phase']]}({item['phase_duration']:.0f}s)"
                        f"  N(l={item['north_left']} t={item['north_through']} r={item['north_right']})"
                        f"  S(l={item['south_left']} t={item['south_through']} r={item['south_right']})"
                        f"  E(l={item['east_left']} t={item['east_through']} r={item['east_right']})"
                        f"  W(l={item['west_left']} t={item['west_through']} r={item['west_right']})"
                        f"  q={item['queue_length']:.0f}"
                    )

            api_ok = False
            try:
                batch_url = API_URL.replace("/telemetry", "/telemetry_batch")
                res = requests.post(batch_url, json=batch_payload, timeout=0.5)

                if res.status_code == 200:
                    api_ok = True
                    api_consecutive_failures = 0
                    decisions     = res.json().get("decisions", [])
                    decision_count += 1
                    step_log_parts = []

                    for decision in decisions:
                        tls_idx    = decision["intersection_id"]
                        if tls_idx >= len(tls_ids):
                            continue
                        tls_id     = tls_ids[tls_idx]
                        model_phase = int(decision["next_phase"]) % 6
                        confidence  = decision.get("confidence", 0.0)
                        queue       = decision.get("queue_length", 0.0)

                        # Through-phase starvation override
                        if model_phase in (0, 3):
                            decisions_since_through[tls_id] = 0
                        else:
                            decisions_since_through[tls_id] += 1
                            if decisions_since_through[tls_id] >= STARVE_LIMIT:
                                # Pick whichever through phase has more demand
                                obs_item = next(
                                    (o for o in batch_payload["intersections"]
                                     if o["intersection_id"] == tls_idx), None
                                )
                                if obs_item:
                                    ns = obs_item["north_through"] + obs_item["south_through"]
                                    ew = obs_item["east_through"]  + obs_item["west_through"]
                                    forced = 0 if ns >= ew else 3
                                else:
                                    forced = 0
                                log.info(
                                    f"  [STARVE OVERRIDE] {tls_id} forcing "
                                    f"{PHASE_NAMES[forced]} after "
                                    f"{decisions_since_through[tls_id]} left-only decisions"
                                )
                                model_phase = forced
                                decisions_since_through[tls_id] = 0

                        # Direction-specific starvation: if the AI keeps picking
                        # NS-through while EW has vehicles (or vice-versa), force
                        # the starved direction. This fires even during quiet periods
                        # so the counters are already saturated when new cars arrive.
                        if model_phase == 0:
                            decisions_since_ns[tls_id] = 0
                        else:
                            decisions_since_ns[tls_id] += 1

                        if model_phase == 3:
                            decisions_since_ew[tls_id] = 0
                        else:
                            decisions_since_ew[tls_id] += 1

                        obs_item = next(
                            (o for o in batch_payload["intersections"]
                             if o["intersection_id"] == tls_idx), None
                        )
                        if obs_item:
                            ew_queue = (obs_item["east_through"] + obs_item["west_through"]
                                        + obs_item["east_left"]  + obs_item["west_left"]
                                        + obs_item["east_right"] + obs_item["west_right"])
                            ns_queue = (obs_item["north_through"] + obs_item["south_through"]
                                        + obs_item["north_left"]  + obs_item["south_left"]
                                        + obs_item["north_right"] + obs_item["south_right"])

                            if decisions_since_ew[tls_id] >= DIRECTION_STARVE_LIMIT and ew_queue > 0:
                                log.info(
                                    f"  [DIR OVERRIDE] {tls_id} forcing EW-through "
                                    f"(starved {decisions_since_ew[tls_id]} decisions, ew_q={ew_queue})"
                                )
                                model_phase = 3
                                decisions_since_ew[tls_id] = 0

                            elif decisions_since_ns[tls_id] >= DIRECTION_STARVE_LIMIT and ns_queue > 0:
                                log.info(
                                    f"  [DIR OVERRIDE] {tls_id} forcing NS-through "
                                    f"(starved {decisions_since_ns[tls_id]} decisions, ns_q={ns_queue})"
                                )
                                model_phase = 0
                                decisions_since_ns[tls_id] = 0

                        # Left-turn starvation: update counters then check
                        for lp in (1, 2, 4, 5):
                            if model_phase == lp:
                                decisions_since_left[tls_id][lp] = 0
                            else:
                                decisions_since_left[tls_id][lp] += 1

                        if model_phase not in (1, 2, 4, 5):
                            obs_item = obs_item or next(
                                (o for o in batch_payload["intersections"]
                                 if o["intersection_id"] == tls_idx), None
                            )
                            if obs_item:
                                # Find the most starved left-turn phase that has demand
                                best_lp, best_wait = None, 0
                                for lp, keys in _LEFT_DEMAND_KEYS.items():
                                    demand = sum(obs_item.get(k, 0) for k in keys)
                                    wait   = decisions_since_left[tls_id][lp]
                                    if wait >= LEFT_STARVE_LIMIT and demand > 0:
                                        if wait > best_wait:
                                            best_lp, best_wait = lp, wait
                                if best_lp is not None:
                                    log.info(
                                        f"  [LEFT OVERRIDE] {tls_id} forcing "
                                        f"{PHASE_NAMES[best_lp]} "
                                        f"(starved {best_wait} decisions)"
                                    )
                                    model_phase = best_lp
                                    decisions_since_left[tls_id][best_lp] = 0

                        # Count this decision
                        phase_counts[tls_id][model_phase] += 1

                        # Get actual current SUMO phase for context
                        try:
                            curr_sumo = int(traci.trafficlight.getPhase(tls_id))
                            curr_model = SUMO_TO_MODEL.get(curr_sumo, 0)
                        except Exception:
                            curr_model = -1

                        # Build per-junction log string
                        changed = (model_phase != curr_model)
                        arrow   = "->" if changed else "  "
                        step_log_parts.append(
                            f"{tls_id}:{PHASE_NAMES[curr_model]}{arrow}"
                            f"{PHASE_NAMES[model_phase]}"
                            f"(q={queue:.0f},conf={confidence:.2f})"
                        )

                        # Apply the decision
                        if yellow_remaining.get(tls_id, 0) > 0:
                            continue

                        target_sumo_phase = MODEL_TO_SUMO_GREEN[model_phase]

                        try:
                            current_p     = int(traci.trafficlight.getPhase(tls_id))
                            current_green = current_p if current_p % 2 == 0 else current_p - 1

                            if target_sumo_phase == current_green:
                                traci.trafficlight.setPhase(tls_id, current_green)
                                continue

                            current_model_phase = SUMO_TO_MODEL.get(current_green, 0)
                            is_through  = current_model_phase in (0, 3)
                            min_green   = MIN_GREEN_THROUGH if is_through else MIN_GREEN_LEFT

                            if current_p % 2 == 0:
                                if step - last_phase_change_step.get(tls_id, 0) < min_green:
                                    continue

                            # Log actual phase switch
                            log.info(
                                f"  SWITCH  step={step:>5}  {tls_id}  "
                                f"{PHASE_NAMES[current_model_phase]:>12} -> "
                                f"{PHASE_NAMES[model_phase]:<12}  "
                                f"q={queue:.0f}  conf={confidence:.2f}"
                            )

                            yellow_phase = current_green + 1
                            traci.trafficlight.setPhase(tls_id, yellow_phase)
                            pending_targets[tls_id]   = target_sumo_phase
                            yellow_remaining[tls_id]  = YELLOW_STEPS
                            last_phase_change_step[tls_id] = step

                        except Exception:
                            pass

                    # Log the full decision row every decision interval
                    log.info(f"step={step:>5} | " + "  ".join(step_log_parts))

            except Exception:
                pass  # network error — handled below

            if not api_ok:
                api_consecutive_failures += 1
                if api_consecutive_failures == FALLBACK_THRESHOLD:
                    log.info(f"  [FALLBACK] Backend unreachable — switching to fixed-time cycling")
                if api_consecutive_failures >= FALLBACK_THRESHOLD:
                    cycle_pos = (step // FALLBACK_CYCLE) % 2
                    fallback_phase = 0 if cycle_pos == 0 else 3
                    target_green = MODEL_TO_SUMO_GREEN[fallback_phase]
                    for tls_id in tls_ids:
                        if yellow_remaining.get(tls_id, 0) > 0:
                            continue
                        try:
                            current_p     = int(traci.trafficlight.getPhase(tls_id))
                            current_green = current_p if current_p % 2 == 0 else current_p - 1
                            if target_green == current_green:
                                continue
                            if step - last_phase_change_step.get(tls_id, 0) < MIN_GREEN_THROUGH:
                                continue
                            yellow_phase = current_green + 1
                            traci.trafficlight.setPhase(tls_id, yellow_phase)
                            pending_targets[tls_id]  = target_green
                            yellow_remaining[tls_id] = YELLOW_STEPS
                            last_phase_change_step[tls_id] = step
                        except Exception:
                            pass

            # Print summary table every 100 decisions
            if decision_count > 0 and decision_count % 100 == 0:
                _log_summary(tls_ids, phase_counts, decision_count, step)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopped.")
    except traci.exceptions.FatalTraCIError:
        print("\nSUMO closed.")
    finally:
        # Always print final summary
        if decision_count > 0:
            _log_summary(tls_ids, phase_counts, decision_count, step)
        log.info(f"  Log saved to: {_LOG_FILE}")
        traci.close()


if __name__ == "__main__":
    main()
