"""
High-traffic no-regression reference
=====================================
Runs the AI controller (TraFixV6 + governor + starvation overrides — the full
production safety net, via tests/utils/sumo_runner.run_simulation) on
type1_high and prints the key must-not-regress metrics:
  throughput (arrived/hr), completion %, mean slow-queue, mean travel time.

Usage:
  .venv/bin/python tests/diagnostics/high_traffic_reference.py \
      --checkpoint trafix_v6/checkpoints/trafix_v6_final.pt --tag before
"""

import argparse
import os
import sys
from pathlib import Path

_DIAG_DIR = Path(__file__).resolve().parent
_TESTS_DIR = _DIAG_DIR.parent
_PROJECT_ROOT = _TESTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if "SUMO_HOME" not in os.environ and os.path.isdir("/usr/share/sumo/tools"):
    os.environ["SUMO_HOME"] = "/usr/share/sumo"

from tests.utils.sumo_runner import run_simulation  # noqa: E402
from tests.metrics.throughput import compute_throughput  # noqa: E402
from tests.metrics.travel_time import compute_travel_time  # noqa: E402
from tests.metrics.queue_length import compute_queue_length  # noqa: E402

_DEFAULT_CKPT = str(_PROJECT_ROOT / "trafix_v6" / "checkpoints" / "trafix_v6_final.pt")
_ROUTE = str(_TESTS_DIR / "scenarios" / "type1_high.rou.xml")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=_DEFAULT_CKPT)
    p.add_argument("--tag", default="ref", help="label for the run_id / output dir")
    p.add_argument("--sim-duration", type=int, default=3600)
    p.add_argument("--no-overrides", action="store_true", help="run the raw AI model without safety overrides")
    args = p.parse_args()

    run_id = f"hi_ref_{args.tag}"
    out_base = str(_DIAG_DIR / "high_traffic_outputs")
    # Always a fresh run — wipe any stale output for this tag.
    import shutil
    stale = Path(out_base) / run_id
    if stale.exists():
        shutil.rmtree(stale)

    out_dir = run_simulation(
        run_id=run_id,
        route_file=_ROUTE,
        mode="ai",
        sim_duration=args.sim_duration,
        checkpoint_path=args.checkpoint,
        outputs_base=out_base,
        use_overrides=not args.no_overrides,
    )

    thr = compute_throughput(out_dir, sim_duration_s=args.sim_duration)
    tt = compute_travel_time(out_dir)
    q = compute_queue_length(out_dir)

    print("\n" + "=" * 56)
    print(f"  HIGH-TRAFFIC REFERENCE  [{args.tag}]")
    print(f"  checkpoint: {args.checkpoint}")
    print("=" * 56)
    print(f"  arrived_vehicles      : {thr['arrived_vehicles']}")
    print(f"  vehicles_still_running: {thr['vehicles_still_running']}")
    print(f"  not_inserted          : {thr['not_inserted']}")
    print(f"  completion_rate       : {thr['completion_rate']*100:.1f}%")
    print(f"  mean_travel_time_s    : {tt['mean_travel_time_s']:.1f}")
    print(f"  mean_halting_queue    : {q['mean_halting_per_junction']:.3f}")
    if 'mean_slow_per_junction' in q:
        print(f"  mean_slow_queue       : {q['mean_slow_per_junction']:.3f}")
    print("=" * 56)


if __name__ == "__main__":
    main()
