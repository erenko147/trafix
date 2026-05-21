"""
Compare baseline vs ep1000 vs ep2000(final) across all 7 scenarios at 3600 s.

Usage (from project root, venv activated):
    python tests/runners/run_checkpoint_compare.py
    python tests/runners/run_checkpoint_compare.py --sim-duration 600   # quick test
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

_TESTS_DIR    = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _TESTS_DIR.parent
_SCENARIOS    = _TESTS_DIR / "scenarios"
_OUTPUTS      = _TESTS_DIR / "outputs" / "ckpt_compare"
_CKPT         = _PROJECT_ROOT / "trafix_v6" / "checkpoints"

for _p in [str(_TESTS_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.seeds      import apply_all_seeds, load_seeds
from utils.sumo_runner import run_simulation
from analysis.checkpoint_compare import run_checkpoint_analysis

# ── Controllers ───────────────────────────────────────────────────────────────
CONTROLLERS = [
    ("baseline", None),
    ("ep1000",   str(_CKPT / "stage3_ep1000.pt")),
    ("ep2000",   str(_CKPT / "trafix_v6_final.pt")),
]

# ── Scenarios ─────────────────────────────────────────────────────────────────
# type1/type2: in-distribution (overlap with training curriculum — kept for
#              reference so we can compare in-dist vs OOD performance side-by-side)
# unseen:      out-of-distribution — flow levels, temporal shapes, and spatial
#              concentrations the model never encountered during training
ALL_SCENARIOS = [
    ("type1",  "low",                  "type1_low.rou.xml"),
    ("type1",  "medium",               "type1_medium.rou.xml"),
    ("type1",  "high",                 "type1_high.rou.xml"),
    ("type2",  "morning_peak",         "type2_morning_peak.rou.xml"),
    ("type2",  "evening_peak",         "type2_evening_peak.rou.xml"),
    ("type2",  "incident",             "type2_incident.rou.xml"),
    ("type2",  "pulse",                "type2_pulse.rou.xml"),
    # ── Unseen / OOD ──────────────────────────────────────────────────────────
    ("unseen", "supersaturation",      "unseen_supersaturation.rou.xml"),
    ("unseen", "stadium_exit",         "unseen_stadium_exit.rou.xml"),
    ("unseen", "peak_plus_incident",   "unseen_peak_plus_incident.rou.xml"),
    ("unseen", "oscillating",          "unseen_oscillating.rou.xml"),
    ("unseen", "tidal_ramp",           "unseen_tidal_ramp.rou.xml"),
    ("unseen", "bidirectional_peak",   "unseen_bidirectional_peak.rou.xml"),
]


def _route(filename: str) -> str:
    p = _SCENARIOS / filename
    if not p.exists():
        sys.exit(f"Route file missing: {p}\nRun: python tests/scenarios/generate_scenarios.py")
    return str(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-duration", type=int, default=3600)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    seeds = load_seeds()
    apply_all_seeds(seeds["python_seed"])

    cfg = dict(
        sim_duration      = args.sim_duration,
        warmup_steps      = 50,
        decision_interval = 10,
        seed              = seeds["sumo_seed"],
    )

    _OUTPUTS.mkdir(parents=True, exist_ok=True)
    manifest = []

    total = len(CONTROLLERS) * len(ALL_SCENARIOS)
    done  = 0

    for test_type, level, route_fn in ALL_SCENARIOS:
        for ctrl_name, ckpt_path in CONTROLLERS:
            done += 1
            run_id = f"cc_{test_type}_{level}_{ctrl_name}"
            out_path = _OUTPUTS / run_id
            # Skip if already completed (tripinfo.xml present from a previous run)
            if (out_path / "tripinfo.xml").exists():
                print(f"\n[{done}/{total}] {run_id} — skipping (already done)")
                manifest.append(dict(
                    test_type=test_type, scenario=level,
                    controller=ctrl_name, output_dir=str(out_path),
                ))
                continue
            print(f"\n[{done}/{total}] {run_id}")
            mode = "baseline" if ctrl_name == "baseline" else "ai"
            out_dir = run_simulation(
                run_id            = run_id,
                route_file        = _route(route_fn),
                mode              = mode,
                sim_duration      = cfg["sim_duration"],
                warmup_steps      = cfg["warmup_steps"],
                decision_interval = cfg["decision_interval"],
                seed              = cfg["seed"],
                checkpoint_path   = ckpt_path or "",
                gui               = args.gui,
                outputs_base      = str(_OUTPUTS),
            )
            manifest.append(dict(
                test_type   = test_type,
                scenario    = level,
                controller  = ctrl_name,
                output_dir  = out_dir,
            ))

    manifest_path = _OUTPUTS / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest saved: {manifest_path}")

    print("\n=== Generating analysis & figures ===")
    run_checkpoint_analysis(manifest, sim_duration_s=cfg["sim_duration"])
    print("\nDone.")


if __name__ == "__main__":
    main()
