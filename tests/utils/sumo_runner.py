"""
Core SUMO simulation runner for the TraFix test suite.

Supports two modes:
  'baseline' — SUMO uses its built-in fixed-timing traffic light programs;
               no TraCI intervention on TLS.
  'ai'       — TraFix v6 controls all TLS at every decision_interval seconds.

Both modes:
  • Run the simulation for `sim_duration` seconds with a fixed seed.
  • Write all required SUMO output files to `output_dir`.
  • Track stops-per-vehicle and junction approach waiting times via TraCI
    (not available from SUMO XML), written to inline_metrics.json.
  • Write junction_map.json ({tls_id: [incoming_edge_ids]}) for metric modules.
"""

import json
import math
import os
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

# ── Project path setup ────────────────────────────────────────────────────────
_TESTS_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _TESTS_DIR.parent

for _p in [str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── SUMO TraCI ────────────────────────────────────────────────────────────────
if "SUMO_HOME" in os.environ:
    _sumo_tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if _sumo_tools not in sys.path:
        sys.path.append(_sumo_tools)
else:
    for _candidate in [
        r"C:\Program Files (x86)\Eclipse\Sumo\tools",
        r"C:\Program Files\Eclipse\Sumo\tools",
        "/usr/share/sumo/tools",
        "/usr/local/share/sumo/tools",
    ]:
        if os.path.isdir(_candidate):
            sys.path.append(_candidate)
            break

try:
    import traci
except ImportError as _e:
    raise ImportError("SUMO TraCI not found. Set SUMO_HOME.") from _e

# ── Constants ─────────────────────────────────────────────────────────────────
_NET_FILE     = str(_PROJECT_ROOT / "sumo" / "map.net.xml")
_DEFAULT_CKPT = str(_PROJECT_ROOT / "trafix_v6" / "checkpoints" / "trafix_v6_final.pt")
_T_WINDOW     = 30       # temporal window for v6 model
_NUM_JUNCTIONS = 5
_NUM_PHASES    = 6
_YELLOW_STEPS  = 3
_LANE_TYPE     = {0: "right", 1: "through", 2: "left"}
_MODEL_TO_SUMO_GREEN  = {0: 0, 1: 2, 2: 4, 3: 6, 4: 8, 5: 10}
_SUMO_TO_MODEL_PHASE  = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2,
                          6: 3, 7: 3, 8: 4, 9: 4, 10: 5, 11: 5}

# Gridlock detection: flag if mean network speed < threshold for this many consecutive seconds
_GRIDLOCK_SPEED_THRESHOLD = 0.5   # m/s
_GRIDLOCK_DURATION_S      = 300   # simulated seconds

# Queue metric: a vehicle counts as "queued" if it is crawling below this speed.
# SUMO's getLastStepHaltingNumber only counts < 0.1 m/s (fully stopped), which
# systematically undercounts roundabout congestion (roundabout traffic crawls
# continuously rather than fully stopping). 1.39 m/s = 5 km/h.
_QUEUE_SLOW_SPEED_MS = 1.39

# ── Live-runner safety overrides (ported verbatim from sumo/run_sumo_live.py) ──
# Production applies these starvation overrides ON TOP of the model + governor.
# Without them the bare greedy (argmax) policy can lock a junction onto one phase
# and gridlock at low demand — so to measure the controller that actually ships,
# the AI test path must include them too.
_STARVE_LIMIT           = 8    # consecutive non-through decisions → force a through
_DIRECTION_STARVE_LIMIT = 10   # one through direction monopolises → force the other
_LEFT_STARVE_LIMIT      = 15   # left phase with demand unserved this long → force it
_LEFT_DEMAND_KEYS = {1: ["north_left"], 2: ["south_left"],
                     4: ["east_left"],  5: ["west_left"]}


def _apply_starvation_overrides(tls_id, model_phase, o, dst, dsn, dse, dsl):
    """
    Mirror of the per-decision overrides in run_sumo_live.py. Mutates the counter
    dicts in place and returns the (possibly forced) model phase.
      dst = decisions_since_through, dsn/dse = since NS/EW, dsl = since each left.
    """
    # Through-phase starvation: too many left-only decisions → force busier through
    if model_phase in (0, 3):
        dst[tls_id] = 0
    else:
        dst[tls_id] += 1
        if dst[tls_id] >= _STARVE_LIMIT:
            ns = o["north_through"] + o["south_through"]
            ew = o["east_through"] + o["west_through"]
            model_phase = 0 if ns >= ew else 3
            dst[tls_id] = 0

    # Direction starvation: one through direction monopolises while the other waits
    dsn[tls_id] = 0 if model_phase == 0 else dsn[tls_id] + 1
    dse[tls_id] = 0 if model_phase == 3 else dse[tls_id] + 1
    ew_q = (o["east_through"] + o["west_through"] + o["east_left"]
            + o["west_left"] + o["east_right"] + o["west_right"])
    ns_q = (o["north_through"] + o["south_through"] + o["north_left"]
            + o["south_left"] + o["north_right"] + o["south_right"])
    if dse[tls_id] >= _DIRECTION_STARVE_LIMIT and ew_q > 0:
        model_phase = 3
        dse[tls_id] = 0
    elif dsn[tls_id] >= _DIRECTION_STARVE_LIMIT and ns_q > 0:
        model_phase = 0
        dsn[tls_id] = 0

    # Left-turn starvation: no governor rule ever forces a left, so guard it here
    for lp in (1, 2, 4, 5):
        dsl[tls_id][lp] = 0 if model_phase == lp else dsl[tls_id][lp] + 1
    if model_phase not in (1, 2, 4, 5):
        best_lp, best_wait = None, 0
        for lp, keys in _LEFT_DEMAND_KEYS.items():
            demand = sum(o.get(k, 0) for k in keys)
            wait   = dsl[tls_id][lp]
            if wait >= _LEFT_STARVE_LIMIT and demand > 0 and wait > best_wait:
                best_lp, best_wait = lp, wait
        if best_lp is not None:
            model_phase = best_lp
            dsl[tls_id][best_lp] = 0

    return model_phase


# ── Observation helper (mirrors SumoEnvironment.get_observations) ─────────────

def _get_observations(tls_ids: List[str], phase_held_since: dict, step: int) -> List[dict]:
    obs = []
    for idx, tls_id in enumerate(tls_ids):
        jx, jy = traci.junction.getPosition(tls_id)
        counts = {
            "north_left": 0, "north_through": 0, "north_right": 0,
            "south_left": 0, "south_through": 0, "south_right": 0,
            "east_left":  0, "east_through":  0, "east_right":  0,
            "west_left":  0, "west_through":  0, "west_right":  0,
        }
        seen = set()
        for link_group in traci.trafficlight.getControlledLinks(tls_id):
            for link in link_group:
                if not link:
                    continue
                from_lane = link[0]
                if from_lane in seen:
                    continue
                seen.add(from_lane)
                edge_id = from_lane.rsplit("_", 1)[0]
                lane_idx = int(from_lane.rsplit("_", 1)[1])
                lane_type = _LANE_TYPE.get(lane_idx, "through")
                direction = _classify_direction(edge_id, jx, jy)
                if direction and lane_type:
                    key = f"{direction}_{lane_type}"
                    if key in counts:
                        counts[key] += traci.lane.getLastStepVehicleNumber(from_lane)

        sumo_phase   = traci.trafficlight.getPhase(tls_id)
        model_phase  = _SUMO_TO_MODEL_PHASE.get(sumo_phase, 0)
        elapsed      = float(step - phase_held_since.get(tls_id, step))
        total_queue  = sum(counts.values())

        obs.append({
            "intersection_id": idx,
            **counts,
            "queue_length": min(total_queue * 1.5, 200.0),
            "current_phase": model_phase,
            "phase_duration": elapsed,
        })
    return obs


def _classify_direction(edge_id: str, jx: float, jy: float) -> str:
    try:
        shape = traci.lane.getShape(f"{edge_id}_0")
        if not shape:
            return ""
        x0, y0 = shape[0]
        dx, dy = x0 - jx, y0 - jy
        if abs(dx) > abs(dy):
            return "west" if dx < 0 else "east"
        return "south" if dy < 0 else "north"
    except Exception:
        return ""


# ── Inline metrics tracker ────────────────────────────────────────────────────

class _InlineTracker:
    def __init__(self, tls_ids: List[str], tls_incoming: Dict[str, List[str]]):
        self.tls_ids = tls_ids
        self.tls_incoming = tls_incoming
        self._stop_counts: Dict[str, int] = {}
        self._was_stopped: Dict[str, bool] = {}
        # per-edge accumulations  (list of values, one per step)
        self._edge_wait: Dict[str, Dict[str, List[float]]] = {
            tls_id: {e: [] for e in edges}
            for tls_id, edges in tls_incoming.items()
        }
        self._edge_halt: Dict[str, Dict[str, List[int]]] = {
            tls_id: {e: [] for e in edges}
            for tls_id, edges in tls_incoming.items()
        }
        # crawl-aware queue: count vehicles below _QUEUE_SLOW_SPEED_MS, not just
        # the fully-stopped (< 0.1 m/s) ones SUMO reports as "halting".
        self._edge_slow: Dict[str, Dict[str, List[int]]] = {
            tls_id: {e: [] for e in edges}
            for tls_id, edges in tls_incoming.items()
        }
        # reverse map edge_id -> tls_id, so a single pass over all vehicles can
        # bucket slow cars onto the right junction (cheaper than per-edge queries)
        self._edge_to_tls: Dict[str, str] = {}
        for tls_id, edges in tls_incoming.items():
            for e in edges:
                self._edge_to_tls[e] = tls_id

    def update(self):
        # per-step slow-vehicle counts per tracked incoming edge
        step_slow: Dict[str, int] = {e: 0 for e in self._edge_to_tls}

        # stops per vehicle (+ accumulate slow counts in the same pass)
        for veh_id in traci.vehicle.getIDList():
            speed = traci.vehicle.getSpeed(veh_id)
            is_stopped = speed < 0.1
            if veh_id not in self._stop_counts:
                self._stop_counts[veh_id] = 0
                self._was_stopped[veh_id] = False
            if is_stopped and not self._was_stopped[veh_id]:
                self._stop_counts[veh_id] += 1
            self._was_stopped[veh_id] = is_stopped

            if speed < _QUEUE_SLOW_SPEED_MS:
                try:
                    road = traci.vehicle.getRoadID(veh_id)
                except Exception:
                    road = ""
                if road in step_slow:
                    step_slow[road] += 1

        # junction waiting / queue
        for tls_id, edges in self.tls_incoming.items():
            for edge_id in edges:
                try:
                    self._edge_wait[tls_id][edge_id].append(
                        traci.edge.getWaitingTime(edge_id)
                    )
                    self._edge_halt[tls_id][edge_id].append(
                        traci.edge.getLastStepHaltingNumber(edge_id)
                    )
                    self._edge_slow[tls_id][edge_id].append(
                        step_slow.get(edge_id, 0)
                    )
                except Exception:
                    pass

    def results(self) -> dict:
        total_vehicles = len(self._stop_counts)
        total_stops    = sum(self._stop_counts.values())
        avg_stops      = total_stops / max(total_vehicles, 1)

        junction_waiting  = {}
        junction_queue    = {}
        all_variances     = []

        for tls_id in self.tls_ids:
            edges       = self.tls_incoming.get(tls_id, [])
            per_edge_mw = {}
            per_edge_mq = {}
            per_edge_ms = {}
            for edge_id in edges:
                waits = self._edge_wait[tls_id].get(edge_id, [])
                halts = self._edge_halt[tls_id].get(edge_id, [])
                slows = self._edge_slow[tls_id].get(edge_id, [])
                per_edge_mw[edge_id] = sum(waits) / len(waits) if waits else 0.0
                per_edge_mq[edge_id] = sum(halts) / len(halts) if halts else 0.0
                per_edge_ms[edge_id] = sum(slows) / len(slows) if slows else 0.0

            means = list(per_edge_mw.values())
            if len(means) >= 2:
                grand = sum(means) / len(means)
                var   = sum((m - grand) ** 2 for m in means) / len(means)
            else:
                var = 0.0

            all_variances.append(var)
            junction_waiting[tls_id] = {
                "approach_mean_wait_s": per_edge_mw,
                "variance": var,
                "std_dev": math.sqrt(var),
            }
            junction_queue[tls_id] = {
                "per_approach_mean_halting": per_edge_mq,
                "mean_halting_total": sum(per_edge_mq.values()),
                "per_approach_mean_slow": per_edge_ms,
                "mean_slow_total": sum(per_edge_ms.values()),
                "slow_speed_threshold_ms": _QUEUE_SLOW_SPEED_MS,
            }

        network_fairness_variance = (
            sum(all_variances) / len(all_variances) if all_variances else 0.0
        )

        return {
            "stops": {
                "total_vehicles": total_vehicles,
                "total_stops": total_stops,
                "avg_stops_per_vehicle": avg_stops,
            },
            "junction_waiting": junction_waiting,
            "junction_queue": junction_queue,
            "network_fairness_variance": network_fairness_variance,
        }


# ── Main simulation runner ────────────────────────────────────────────────────

def run_simulation(
    run_id: str,
    route_file: str,
    mode: str,                          # 'baseline' or 'ai'
    sim_duration: int = 3600,
    warmup_steps: int = 50,
    decision_interval: int = 10,
    seed: int = 42,
    checkpoint_path: str = _DEFAULT_CKPT,
    gui: bool = False,
    outputs_base: Optional[str] = None,
    net_file: Optional[str] = None,
    additional_files: Optional[str] = None,
    use_overrides: bool = True,
) -> str:
    """
    Run one SUMO simulation and return the path to its output directory.

    Parameters
    ----------
    run_id          : unique identifier, e.g. "type1_low_baseline"
    route_file      : absolute path to .rou.xml
    mode            : 'baseline' or 'ai'
    sim_duration    : simulated seconds (default 3600)
    warmup_steps    : initial steps before control/tracking starts
    decision_interval : seconds between AI decisions
    seed            : fixed SUMO seed
    checkpoint_path : path to TraFix v6 .pt file (AI mode only)
    gui             : open SUMO-GUI window (for debugging)
    outputs_base    : override base output directory
    """
    if mode not in ("baseline", "ai"):
        raise ValueError(f"mode must be 'baseline' or 'ai', got '{mode}'")

    # ── Output directory ──────────────────────────────────────────────────────
    if outputs_base is None:
        outputs_base = str(_TESTS_DIR / "outputs")
    out_dir = Path(outputs_base) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── AI model setup ────────────────────────────────────────────────────────
    model     = None
    governor  = None
    device    = None
    parse_obs = None

    if mode == "ai":
        import torch
        from trafix_v6.trafix_v6 import TraFixV6
        from trafix_v6.rule_governor import RuleGovernor, sample_governed
        from backend.ai.trafix_v2 import parse_sumo_observations

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = TraFixV6().to(device)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        governor  = RuleGovernor(
            num_junctions=_NUM_JUNCTIONS, num_phases=_NUM_PHASES,
            min_green_s=10.0, max_green_s=90.0,
            flicker_window=2, flicker_penalty=3.0,
            pressure_boost=1.0, pressure_thresh=0.12,   # Step 5: in sync with backend
        )
        parse_obs = parse_sumo_observations

    # ── SUMO command ──────────────────────────────────────────────────────────
    sumo_bin = "sumo-gui" if gui else "sumo"
    _net = net_file if net_file is not None else _NET_FILE
    cmd = [
        sumo_bin,
        "--net-file",     _net,
        "--route-files",  route_file,
        "--seed",         str(seed),
        "--no-warnings",  "true",
        "--step-length",  "1.0",
        "--begin",        "0",
        "--end",          str(sim_duration),
        "--waiting-time-memory",       "1000",
        "--time-to-teleport",          "-1",
        "--time-to-teleport.highways", "-1",
        "--tripinfo-output",    str(out_dir / "tripinfo.xml"),
        "--summary-output",     str(out_dir / "summary.xml"),
        "--queue-output",       str(out_dir / "queue.xml"),
        "--statistic-output",   str(out_dir / "statistics.xml"),
        "--device.emissions.probability", "1.0",
        "--emissions.volumetric-fuel",    "true",
    ]
    if additional_files is not None:
        cmd += ["--additional-files", additional_files]

    # ── Start simulation (retry once if SUMO port still lingering) ───────────
    import time as _time
    for _attempt in range(2):
        try:
            traci.start(cmd)
            break
        except Exception as _e:
            if _attempt == 0:
                _time.sleep(3)
            else:
                raise

    tls_ids = sorted(traci.trafficlight.getIDList())

    # Build junction → incoming edges map
    tls_incoming: Dict[str, List[str]] = {}
    for tls_id in tls_ids:
        edges = set()
        for link_group in traci.trafficlight.getControlledLinks(tls_id):
            for link in link_group:
                if link:
                    edge = link[0].rsplit("_", 1)[0]
                    if not edge.startswith(":"):
                        edges.add(edge)
        tls_incoming[tls_id] = sorted(edges)

    (out_dir / "junction_map.json").write_text(
        json.dumps(tls_incoming, indent=2), encoding="utf-8"
    )

    # ── Warmup ────────────────────────────────────────────────────────────────
    step = 0
    for _ in range(warmup_steps):
        traci.simulationStep()
        step += 1

    phase_held_since: Dict[str, int] = {tls_id: step for tls_id in tls_ids}

    # ── AI-specific init ──────────────────────────────────────────────────────
    window = None
    pending_target: Dict[str, int]  = {}
    yellow_remaining: Dict[str, int] = {}

    if mode == "ai":
        import torch
        obs0  = _get_observations(tls_ids, phase_held_since, step)
        x0    = parse_obs(obs0, device=device)
        window = deque([x0.detach()] * _T_WINDOW, maxlen=_T_WINDOW)
        governor.reset()

    # Starvation-override counters (one set per junction), mirroring run_sumo_live.py
    dec_since_through = {t: 0 for t in tls_ids}
    dec_since_ns      = {t: 0 for t in tls_ids}
    dec_since_ew      = {t: 0 for t in tls_ids}
    dec_since_left    = {t: {p: 0 for p in (1, 2, 4, 5)} for t in tls_ids}

    # ── Inline metrics tracker ────────────────────────────────────────────────
    tracker = _InlineTracker(tls_ids, tls_incoming)

    # ── Gridlock detection ────────────────────────────────────────────────────
    _low_speed_streak = 0
    _gridlock_detected = False
    _gridlock_step     = None

    # ── Main loop ─────────────────────────────────────────────────────────────
    next_decision = step + decision_interval

    while step < sim_duration:
        if traci.simulation.getMinExpectedNumber() == 0:
            break

        tracker.update()

        # Gridlock check: mean speed across all vehicles
        veh_ids = traci.vehicle.getIDList()
        if veh_ids:
            mean_spd = sum(traci.vehicle.getSpeed(v) for v in veh_ids) / len(veh_ids)
            if mean_spd < _GRIDLOCK_SPEED_THRESHOLD:
                _low_speed_streak += 1
                if _low_speed_streak >= _GRIDLOCK_DURATION_S and not _gridlock_detected:
                    _gridlock_detected = True
                    _gridlock_step     = step
            else:
                _low_speed_streak = 0
        else:
            _low_speed_streak = 0

        # ── AI decision ───────────────────────────────────────────────────────
        if mode == "ai" and step >= next_decision:
            import torch
            obs       = _get_observations(tls_ids, phase_held_since, step)
            x         = parse_obs(obs, device=device)
            window.append(x.detach())
            obs_input = torch.stack(list(window)).unsqueeze(0).to(device)  # [1,T,J,D]

            with torch.no_grad():
                logits_list, _ = model.forward(obs_input)
                obs_last        = obs_input[0, -1]          # [J, D]
                # Match production (backend/main.py): stateful governor (incl.
                # anti-flicker) + GREEDY argmax. The deployed backend uses argmax,
                # not sampling — sampling here would measure a different, noisier
                # controller than the one that actually ships.
                masked     = governor.apply(logits_list, obs_last)
                actions_1d = torch.stack(
                    [torch.argmax(l, dim=-1).reshape(()) for l in masked]
                )                                            # [J], deterministic

            # Governor flicker state tracks the governed argmax (matches the
            # backend, which updates state BEFORE the runner's overrides apply).
            governor.update_state(actions_1d)

            # Production safety net: apply the live-runner starvation overrides on
            # top of the model+governor output. Counters update for EVERY junction
            # each decision (even if actuation is blocked by an in-progress yellow),
            # exactly as in run_sumo_live.py.
            final_actions = []
            for i, tls_id in enumerate(tls_ids):
                mp = int(actions_1d[i].item()) % _NUM_PHASES
                if use_overrides:
                    mp = _apply_starvation_overrides(
                        tls_id, mp, obs[i],
                        dec_since_through, dec_since_ns, dec_since_ew, dec_since_left)
                final_actions.append(mp)

            # Apply actions with yellow transitions
            for i, tls_id in enumerate(tls_ids):
                if yellow_remaining.get(tls_id, 0) > 0:
                    continue
                model_action     = final_actions[i]
                target_sumo      = _MODEL_TO_SUMO_GREEN[model_action]
                current_sumo     = traci.trafficlight.getPhase(tls_id)
                if target_sumo == current_sumo:
                    continue
                pending_target[tls_id]   = target_sumo
                yellow_remaining[tls_id] = _YELLOW_STEPS
                yellow_phase = (
                    current_sumo + 1 if current_sumo % 2 == 0 else current_sumo
                )
                traci.trafficlight.setPhase(tls_id, yellow_phase)

            next_decision = step + decision_interval

        # ── Advance yellow transitions (AI mode) ──────────────────────────────
        if mode == "ai":
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

    # ── Capture vehicles still in the network at sim end (NOT in tripinfo) ──────
    # tripinfo.xml only records COMPLETED trips. A controller that gridlocks cars
    # excludes their (huge) waiting/travel times from every average → survivorship
    # bias. We record each still-running vehicle's accumulated time via TraCI so
    # the time-based metrics can fold them back in.
    unfinished = []
    try:
        for vid in traci.vehicle.getIDList():
            try:
                depart = float(traci.vehicle.getDeparture(vid))
                unfinished.append({
                    "id": vid,
                    "depart": depart,
                    "time_in_network_s": float(step) - depart if depart >= 0 else 0.0,
                    "waiting_time_s": float(traci.vehicle.getAccumulatedWaitingTime(vid)),
                    "time_loss_s": float(traci.vehicle.getTimeLoss(vid)),
                })
            except Exception:
                pass
    except Exception:
        pass

    traci.close()
    _time.sleep(1)   # let SUMO process fully exit before next simulation starts

    (out_dir / "unfinished_vehicles.json").write_text(
        json.dumps({"sim_end_step": step, "count": len(unfinished),
                    "vehicles": unfinished}, indent=2),
        encoding="utf-8",
    )

    # ── Save inline metrics ───────────────────────────────────────────────────
    inline = tracker.results()
    (out_dir / "inline_metrics.json").write_text(
        json.dumps(inline, indent=2), encoding="utf-8"
    )

    # ── Save gridlock status ──────────────────────────────────────────────────
    gridlock_result = {
        "gridlocked": _gridlock_detected,
        "gridlock_step": _gridlock_step,
        "threshold_speed_ms": _GRIDLOCK_SPEED_THRESHOLD,
        "threshold_duration_s": _GRIDLOCK_DURATION_S,
    }
    (out_dir / "gridlock.json").write_text(
        json.dumps(gridlock_result, indent=2), encoding="utf-8"
    )
    if _gridlock_detected:
        print(f"[{run_id}] *** GRIDLOCK DETECTED at step {_gridlock_step} ***")

    # ── Teleport sanity check ─────────────────────────────────────────────────
    # SUMO is launched with --time-to-teleport -1, so teleporting is DISABLED and
    # this must be 0 for every run. If it is ever non-zero, the no-teleport
    # guarantee broke and the run is invalid — surface it loudly.
    teleports = 0
    try:
        import xml.etree.ElementTree as _ET
        # statistics.xml carries the AUTHORITATIVE total. (summary.xml's teleports
        # attribute is cumulative-per-step, so summing it overcounts massively.)
        stats = out_dir / "statistics.xml"
        if stats.exists():
            el = _ET.parse(stats).getroot().find("teleports")
            if el is not None:
                teleports = int(el.get("total", 0))
    except Exception:
        pass
    if teleports > 0:
        print(f"[{run_id}] *** WARNING: {teleports} TELEPORTS — teleporting was "
              f"supposed to be disabled (--time-to-teleport -1) ***")
    else:
        print(f"[{run_id}] teleports=0 (OK)")

    print(f"[{run_id}] Done — {len(unfinished)} cars still in network — outputs in {out_dir}")
    return str(out_dir)
