"""
Test Type 2: Roundabout vs Standard junction comparison (4-way).

Controllers compared at each of 3 traffic levels:
  roundabout_fixed  — 5 Turkish-style roundabouts, Webster 86 s fixed-timing TLS  (new runs)
  standard_fixed    — 5 standard cross-intersections, SUMO built-in fixed timing  (reuse Type 1)
  standard_ep1000   — standard cross-intersections, AI TraFix v6 at ep1000        (reuse ckpt compare)
  standard_ep2000   — standard cross-intersections, AI TraFix v6 at ep2000/final  (reuse Type 1 / ckpt compare)

Reuse priority for standard runs:
  1. tests/outputs/ckpt_compare/cc_type1_{level}_{ctrl}/   (checkpoint-compare runner)
  2. tests/outputs/type1_{level}_{ctrl}/                   (run_all.py)
  3. Fresh simulation (if neither exists)

Usage (from project root, venv activated):
    python tests/runners/run_test_type_2.py
    python tests/runners/run_test_type_2.py --sim-duration 600   # quick smoke test
    python tests/runners/run_test_type_2.py --analysis-only      # skip simulations
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

_TESTS_DIR    = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _TESTS_DIR.parent
_SCENARIO_DIR = _TESTS_DIR / "scenarios" / "test_type_2_junction_comparison"
_OUTPUTS      = _TESTS_DIR / "outputs"
_T2_OUTPUTS   = _OUTPUTS / "test_type_2"
_REPORTS      = _TESTS_DIR / "reports"  / "test_type_2"
_CKPT_DIR     = _PROJECT_ROOT / "trafix_v6" / "checkpoints"

_NET_A  = str(_SCENARIO_DIR / "junction_a_roundabout" / "network.net.xml")
_ADD_A  = str(_SCENARIO_DIR / "junction_a_roundabout" / "tls_fixed.add.xml")
_NET_B  = str(_PROJECT_ROOT / "sumo" / "map.net.xml")

for _p in [str(_TESTS_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.seeds       import apply_all_seeds, load_seeds
from utils.sumo_runner import run_simulation
from analysis.junction_compare import run_junction_analysis


TRAFFIC_LEVELS = ["low", "medium", "high"]

# ── Controller definitions ─────────────────────────────────────────────────────
# Each entry: (controller_key, sumo_mode, checkpoint_path_or_None, net_file, additional_files)
# standard_* controllers are resolved via _find_standard_output() — no fresh sim unless missing.

_STANDARD_CONTROLLERS = [
    # key             sumo_mode    ckpt_suffix     run_all_suffix
    ("standard_fixed",  "baseline", "baseline",     "baseline"),
    ("standard_ep1000", "ai",       "ep1000",        None),
    ("standard_ep2000", "ai",       "ep2000",        "ai"),
]

_CKPT_PATH = {
    "ep1000": str(_CKPT_DIR / "stage3_ep1000.pt"),
    "ep2000": str(_CKPT_DIR / "trafix_v6_final.pt"),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _route_file(level: str) -> Path:
    p = _SCENARIO_DIR / f"{level}.rou.xml"
    if not p.exists():
        sys.exit(
            f"Route file missing: {p}\n"
            "Run: python tests/scenarios/test_type_2_junction_comparison/build_roundabout_net.py"
        )
    return p


def _demand_hash(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _find_standard_output(level: str, ckpt_suffix: str, run_all_suffix: str | None) -> Path | None:
    """
    Try to locate an existing standard-junction output directory.
    Search order:
      1. ckpt_compare output  (cc_type1_{level}_{ckpt_suffix})
      2. run_all output       (type1_{level}_{run_all_suffix})   — only if run_all_suffix given
    Returns the Path if it contains tripinfo.xml, else None.
    """
    candidates = [_OUTPUTS / "ckpt_compare" / f"cc_type1_{level}_{ckpt_suffix}"]
    if run_all_suffix:
        candidates.append(_OUTPUTS / f"type1_{level}_{run_all_suffix}")

    for candidate in candidates:
        if (candidate / "tripinfo.xml").exists():
            return candidate
    return None


def _check_prerequisites():
    missing = []
    for path, label in [(_NET_A, "Junction A net"), (_ADD_A, "Junction A TLS"), (_NET_B, "Junction B net")]:
        if not Path(path).exists():
            missing.append(f"{label}: {path}")
    for level in TRAFFIC_LEVELS:
        if not (_SCENARIO_DIR / f"{level}.rou.xml").exists():
            missing.append(f"Route symlink: {_SCENARIO_DIR / level}.rou.xml")
    if missing:
        print("ERROR — missing prerequisites:")
        for m in missing:
            print(f"  {m}")
        sys.exit(
            "\nBuild network first:\n"
            "  python tests/scenarios/test_type_2_junction_comparison/build_roundabout_net.py"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test Type 2: Roundabout vs Standard (4-way)")
    parser.add_argument("--sim-duration", type=int, default=3600)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Skip simulations; load manifest from disk and re-run analysis")
    parser.add_argument("--no-reuse", action="store_true",
                        help="Run every standard controller fresh (do NOT reuse "
                             "outputs from other suites). Guarantees a fully fresh, "
                             "self-consistent Test Type 2 report.")
    args = parser.parse_args()

    _check_prerequisites()

    seeds = load_seeds()
    apply_all_seeds(seeds["python_seed"])

    sim_cfg = dict(
        sim_duration      = args.sim_duration,
        warmup_steps      = 50,
        decision_interval = 10,
        seed              = seeds["sumo_seed"],
    )

    _T2_OUTPUTS.mkdir(parents=True, exist_ok=True)
    _REPORTS.mkdir(parents=True, exist_ok=True)

    manifest = []
    demand_hashes = {level: _demand_hash(_route_file(level)) for level in TRAFFIC_LEVELS}

    if args.analysis_only:
        manifest_path = _T2_OUTPUTS / "manifest.json"
        if not manifest_path.exists():
            sys.exit(f"No manifest at {manifest_path}. Run simulations first.")
        manifest = json.loads(manifest_path.read_text())
        print(f"Loaded manifest: {len(manifest)} entries")
    else:
        total_fresh = 0   # count only fresh simulations

        for level in TRAFFIC_LEVELS:
            route = str(_route_file(level))

            # ── Roundabout (always a fresh run) ──────────────────────────────
            run_id = f"t2_{level}_roundabout_fixed"
            out    = _T2_OUTPUTS / run_id
            if (out / "tripinfo.xml").exists():
                print(f"[SKIP] {run_id} — already done")
            else:
                total_fresh += 1
                print(f"\n[RUN] {run_id}")
                run_simulation(
                    run_id           = run_id,
                    route_file       = route,
                    mode             = "baseline",
                    net_file         = _NET_A,
                    additional_files = _ADD_A,
                    **sim_cfg,
                    gui              = args.gui,
                    outputs_base     = str(_T2_OUTPUTS),
                )

            manifest.append(dict(
                controller    = "roundabout_fixed",
                traffic_level = level,
                output_dir    = str(out),
                demand_hash   = demand_hashes[level],
                source        = "fresh",
            ))

            # ── Standard controllers (reuse if possible) ───────────────────────
            for ctrl_key, sumo_mode, ckpt_suffix, run_all_suffix in _STANDARD_CONTROLLERS:
                existing = None if args.no_reuse else _find_standard_output(
                    level, ckpt_suffix, run_all_suffix)

                if existing:
                    source = "reused"
                    out_dir = existing
                    print(f"[REUSE] {ctrl_key} / {level} → {existing}")
                else:
                    # Run fresh
                    run_id = f"t2_{level}_{ctrl_key}"
                    out_dir = _T2_OUTPUTS / run_id
                    if (out_dir / "tripinfo.xml").exists():
                        source = "fresh"
                        print(f"[SKIP] {run_id} — already done")
                    else:
                        total_fresh += 1
                        source = "fresh"
                        print(f"\n[RUN] {run_id}")
                        ckpt = _CKPT_PATH.get(ckpt_suffix) if sumo_mode == "ai" else None
                        run_simulation(
                            run_id           = str(run_id),
                            route_file       = route,
                            mode             = sumo_mode,
                            net_file         = _NET_B,
                            checkpoint_path  = ckpt or "",
                            **sim_cfg,
                            gui              = args.gui,
                            outputs_base     = str(_T2_OUTPUTS),
                        )

                manifest.append(dict(
                    controller    = ctrl_key,
                    traffic_level = level,
                    output_dir    = str(out_dir),
                    demand_hash   = demand_hashes[level],
                    source        = source,
                ))

        print(f"\n{total_fresh} fresh simulations run.")

    # Save manifest
    manifest_path = _T2_OUTPUTS / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")

    # ── Demand-hash parity check ──────────────────────────────────────────────
    print("\n=== Demand-hash parity check ===")
    for level in TRAFFIC_LEVELS:
        hashes = {e["demand_hash"] for e in manifest if e["traffic_level"] == level and "demand_hash" in e}
        if len(hashes) > 1:
            print(f"  [FAIL] {level}: mismatched hashes — {hashes}")
        else:
            print(f"  [OK]   {level}: {next(iter(hashes), 'N/A')[:16]}...")

    # ── Gridlock summary ──────────────────────────────────────────────────────
    print("\n=== Gridlock check ===")
    any_gridlock = False
    for entry in manifest:
        gl_file = Path(entry["output_dir"]) / "gridlock.json"
        if gl_file.exists() and json.loads(gl_file.read_text())["gridlocked"]:
            print(f"  *** GRIDLOCK: {entry['controller']} / {entry['traffic_level']} ***")
            any_gridlock = True
    if not any_gridlock:
        print("  No gridlock detected.")

    # ── Analysis ──────────────────────────────────────────────────────────────
    print("\n=== Generating analysis & figures ===")
    run_junction_analysis(manifest, sim_duration_s=sim_cfg["sim_duration"], reports_dir=_REPORTS)
    print(f"\nDone. Reports: {_REPORTS}")


if __name__ == "__main__":
    main()
