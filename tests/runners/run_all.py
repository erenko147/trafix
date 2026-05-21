"""
TraFix v6 — Full test suite runner.

Usage (from project root, using project venv):
    python tests/runners/run_all.py                          # full suite
    python tests/runners/run_all.py --type1-only             # Test Type 1 only
    python tests/runners/run_all.py --type2-only             # Test Type 2 only
    python tests/runners/run_all.py --repro-check            # reproducibility only
    python tests/runners/run_all.py --gui                    # open SUMO-GUI (debug)

Prerequisites:
    1. python tests/scenarios/generate_scenarios.py   (run once)
    2. SUMO installed and SUMO_HOME set
    3. Project venv activated

Seed used for all runs: 42 (from tests/config/seeds.yaml)
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

_TESTS_DIR    = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _TESTS_DIR.parent
_SCENARIOS    = _TESTS_DIR / "scenarios"
_OUTPUTS      = _TESTS_DIR / "outputs"
_CKPT         = _PROJECT_ROOT / "trafix_v6" / "checkpoints" / "trafix_v6_final.pt"

for _p in [str(_TESTS_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.seeds import apply_all_seeds, load_seeds
from utils.sumo_runner import run_simulation
from analysis.compare import run_analysis


# ── Test definitions ──────────────────────────────────────────────────────────

TYPE1_LEVELS = [
    ("low",    "type1_low.rou.xml"),
    ("medium", "type1_medium.rou.xml"),
    ("high",   "type1_high.rou.xml"),
]

TYPE2_SCENARIOS = [
    ("morning_peak",  "type2_morning_peak.rou.xml"),
    ("evening_peak",  "type2_evening_peak.rou.xml"),
    ("incident",      "type2_incident.rou.xml"),
    ("pulse",         "type2_pulse.rou.xml"),
]


def _route(filename: str) -> str:
    path = _SCENARIOS / filename
    if not path.exists():
        sys.exit(
            f"Route file not found: {path}\n"
            f"Run: python tests/scenarios/generate_scenarios.py"
        )
    return str(path)


# ── Hash helper for reproducibility check ────────────────────────────────────

def _hash_tripinfo(path: str) -> str:
    """Hash only the <tripinfo> entries, skipping SUMO's wall-clock header comment."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    # Re-serialise the root so the timestamp comment is gone
    content = ET.tostring(tree.getroot(), encoding="unicode")
    return hashlib.sha256(content.encode()).hexdigest()


def _hash_run(output_dir: str) -> str:
    """Hash deterministic parts of a run: tripinfo data + inline metrics."""
    parts = []
    tripinfo = Path(output_dir) / "tripinfo.xml"
    if tripinfo.exists():
        parts.append(_hash_tripinfo(str(tripinfo)))
    inline = Path(output_dir) / "inline_metrics.json"
    if inline.exists():
        parts.append(inline.read_text(encoding="utf-8"))
    return hashlib.sha256("".join(parts).encode()).hexdigest()


# ── Individual run helper ─────────────────────────────────────────────────────

def _run_pair(scenario: str, level: str, route_file: str, cfg: dict, gui: bool):
    """Run baseline then AI for one (scenario, level) pair."""
    manifest = []
    for ctrl in ("baseline", "ai"):
        run_id = f"{scenario}_{level}_{ctrl}"
        print(f"\n{'='*60}")
        print(f"  {run_id}")
        print(f"{'='*60}")
        out_dir = run_simulation(
            run_id=run_id,
            route_file=route_file,
            mode=ctrl,
            sim_duration=cfg["sim_duration"],
            warmup_steps=cfg["warmup_steps"],
            decision_interval=cfg["decision_interval"],
            seed=cfg["seed"],
            checkpoint_path=str(_CKPT),
            gui=gui,
            outputs_base=str(_OUTPUTS),
        )
        manifest.append({
            "scenario": scenario,
            "traffic_level": level,
            "controller": ctrl,
            "output_dir": out_dir,
        })
    return manifest


# ── Reproducibility check ─────────────────────────────────────────────────────

def _repro_check(cfg: dict, gui: bool):
    """Run the medium-baseline scenario twice and assert identical outputs."""
    print("\n" + "="*60)
    print("  REPRODUCIBILITY CHECK  (type1_medium_baseline × 2)")
    print("="*60)
    route = _route("type1_medium.rou.xml")
    out1 = run_simulation(
        run_id="repro_run1",
        route_file=route,
        mode="baseline",
        sim_duration=cfg["sim_duration"],
        warmup_steps=cfg["warmup_steps"],
        decision_interval=cfg["decision_interval"],
        seed=cfg["seed"],
        gui=gui,
        outputs_base=str(_OUTPUTS),
    )
    out2 = run_simulation(
        run_id="repro_run2",
        route_file=route,
        mode="baseline",
        sim_duration=cfg["sim_duration"],
        warmup_steps=cfg["warmup_steps"],
        decision_interval=cfg["decision_interval"],
        seed=cfg["seed"],
        gui=gui,
        outputs_base=str(_OUTPUTS),
    )
    h1, h2 = _hash_run(out1), _hash_run(out2)
    if h1 == h2:
        print(f"\n  PASS — outputs are byte-identical (sha256={h1[:16]}...)")
    else:
        print(f"\n  FAIL — outputs differ!")
        print(f"    run1 hash: {h1}")
        print(f"    run2 hash: {h2}")
        print("  Check SUMO seed, Python seed, and torch determinism flags.")
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TraFix v6 — full SUMO test suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--type1-only",   action="store_true")
    parser.add_argument("--type2-only",   action="store_true")
    parser.add_argument("--repro-check",  action="store_true",
                        help="Only run reproducibility verification (2 baseline runs)")
    parser.add_argument("--gui",          action="store_true",
                        help="Open SUMO-GUI for each run (slow, debug only)")
    parser.add_argument("--sim-duration", type=int, default=3600)
    parser.add_argument("--no-analysis",  action="store_true",
                        help="Skip report/chart generation")
    args = parser.parse_args()

    # Load seeds and config
    seeds = load_seeds()
    apply_all_seeds(seeds["python_seed"])

    cfg = {
        "seed":              seeds["sumo_seed"],
        "sim_duration":      args.sim_duration,
        "warmup_steps":      50,
        "decision_interval": 10,
    }

    print(f"\nTraFix v6 Test Suite")
    print(f"  SUMO seed       : {cfg['seed']}")
    print(f"  Sim duration    : {cfg['sim_duration']} s")
    print(f"  Checkpoint      : {_CKPT}")

    if args.repro_check:
        _repro_check(cfg, args.gui)
        return

    manifest = []

    # ── Test Type 1 — variable traffic load ──────────────────────────────────
    if not args.type2_only:
        print("\n\n### Test Type 1 — Variable Traffic Load ###")
        for level, route_fn in TYPE1_LEVELS:
            manifest += _run_pair("type1", level, _route(route_fn), cfg, args.gui)

    # ── Test Type 2 — scenario robustness ────────────────────────────────────
    if not args.type1_only:
        print("\n\n### Test Type 2 — Scenario Robustness ###")
        for scenario, route_fn in TYPE2_SCENARIOS:
            manifest += _run_pair("type2", scenario, _route(route_fn), cfg, args.gui)

    # ── Reproducibility check (included in full suite) ────────────────────────
    if not args.type1_only and not args.type2_only:
        _repro_check(cfg, args.gui)

    # Save manifest
    manifest_path = _OUTPUTS / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n  Manifest saved: {manifest_path}")

    # ── Analysis ──────────────────────────────────────────────────────────────
    if not args.no_analysis and manifest:
        print("\n\n### Generating Reports ###")
        run_analysis(manifest, sim_duration_s=cfg["sim_duration"])

    print("\n\nDone. All runs complete.")


if __name__ == "__main__":
    main()
