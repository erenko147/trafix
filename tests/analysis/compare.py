"""
Statistical comparison and report generation.

Reads all per-run metrics, builds a master CSV, a markdown summary report,
and bar-chart visualisations (one per metric × traffic level).
"""

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

_TESTS_DIR = Path(__file__).resolve().parents[1]
_REPORTS   = _TESTS_DIR / "reports"
_CHARTS    = _REPORTS / "charts"

sys.path.insert(0, str(_TESTS_DIR))

from metrics.travel_time     import compute_travel_time
from metrics.emissions       import compute_emissions
from metrics.waiting_time    import compute_waiting_time
from metrics.queue_length    import compute_queue_length
from metrics.throughput      import compute_throughput
from metrics.network_speed   import compute_network_speed
from metrics.teleports       import compute_teleports
from metrics.time_loss       import compute_time_loss
from metrics.fuel_consumption import compute_fuel_consumption
from metrics.pollutants      import compute_pollutants
from metrics.stops_per_vehicle import compute_stops_per_vehicle
from metrics.junction_fairness import compute_junction_fairness


# ── Metric extractors ─────────────────────────────────────────────────────────
# Each returns a flat {metric_name: value} dict for a single run directory.

def _extract_all_metrics(out_dir: str, sim_duration_s: int = 3600) -> Dict[str, float]:
    d = Path(out_dir)
    m: Dict[str, float] = {}

    r = compute_travel_time(d)
    m["travel_time_s"]           = r["mean_travel_time_s"]

    r = compute_emissions(d)
    m["co2_mg_per_vehicle"]      = r["mean_CO2_per_vehicle_mg"]
    m["co2_total_mg"]            = r["total_CO2_mg"]

    r = compute_waiting_time(d)
    m["waiting_time_s"]          = r["mean_waiting_time_s"]

    r = compute_queue_length(d)
    m["queue_length_vehicles"]   = r["mean_halting_per_junction"]

    r = compute_throughput(d, sim_duration_s)
    m["throughput_veh_hr"]       = r["throughput_veh_per_hr"]

    r = compute_network_speed(d)
    m["network_speed_ms"]        = r["mean_speed_ms"]

    r = compute_teleports(d)
    m["teleports"]               = float(r["total_teleports"])

    r = compute_time_loss(d)
    m["time_loss_s"]             = r["mean_time_loss_s"]

    r = compute_fuel_consumption(d, volumetric=True)
    m["fuel_per_vehicle_L"]      = r["mean_fuel_per_vehicle_L"]

    r = compute_pollutants(d)
    m["NOx_total_mg"]            = r["totals_mg"]["NOx"]
    m["PMx_total_mg"]            = r["totals_mg"]["PMx"]
    m["HC_total_mg"]             = r["totals_mg"]["HC"]
    m["CO_total_mg"]             = r["totals_mg"]["CO"]

    r = compute_stops_per_vehicle(d)
    m["stops_per_vehicle"]       = r["avg_stops_per_vehicle"]

    r = compute_junction_fairness(d)
    m["fairness_variance"]       = r["network_wide_variance"]

    return m


# ── lower-is-better flag per metric ──────────────────────────────────────────
_LOWER_IS_BETTER = {
    "travel_time_s": True, "co2_mg_per_vehicle": True, "co2_total_mg": True,
    "waiting_time_s": True, "queue_length_vehicles": True,
    "throughput_veh_hr": False, "network_speed_ms": False,
    "teleports": True, "time_loss_s": True, "fuel_per_vehicle_L": True,
    "NOx_total_mg": True, "PMx_total_mg": True, "HC_total_mg": True, "CO_total_mg": True,
    "stops_per_vehicle": True, "fairness_variance": True,
}


def _pct_change(baseline: float, ai: float, lower_is_better: bool) -> float:
    """Positive = AI improved, negative = AI regressed."""
    if baseline == 0:
        return 0.0
    raw = (baseline - ai) / baseline * 100.0
    return raw if lower_is_better else -raw


# ── CSV export ────────────────────────────────────────────────────────────────

def write_results_csv(runs: List[dict], out_path: Path = None):
    """
    runs: list of {scenario, traffic_level, controller, metric, value}
    """
    out_path = out_path or (_REPORTS / "results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "traffic_level",
                                                "controller", "metric", "value"])
        writer.writeheader()
        writer.writerows(runs)
    print(f"  CSV written: {out_path}")


# ── Markdown report ───────────────────────────────────────────────────────────

def write_summary_report(
    grouped: Dict,          # {(scenario, level): {"baseline": metrics, "ai": metrics}}
    out_path: Path = None,
):
    out_path = out_path or (_REPORTS / "summary.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# TraFix v6 — Test Suite Results\n"]

    improved, regressed = [], []

    for (scenario, level), controllers in sorted(grouped.items()):
        baseline = controllers.get("baseline", {})
        ai       = controllers.get("ai", {})
        if not baseline or not ai:
            continue

        lines.append(f"\n## {scenario} | {level}\n")
        lines.append(f"| Metric | Baseline | AI (v6) | Δ% |")
        lines.append(f"|--------|----------|---------|-----|")

        for metric in sorted(baseline.keys()):
            bval = baseline.get(metric, 0.0)
            aval = ai.get(metric, 0.0)
            lib  = _LOWER_IS_BETTER.get(metric, True)
            pct  = _pct_change(bval, aval, lib)
            sign = "✓" if pct >= 0 else "✗"
            lines.append(
                f"| {metric} | {bval:.4g} | {aval:.4g} | {sign} {pct:+.1f}% |"
            )
            if pct > 0:
                improved.append(f"{metric} ({scenario}/{level})")
            elif pct < 0:
                regressed.append(f"{metric} ({scenario}/{level})")

    # Verdict
    lines.append("\n## Verdict\n")
    if improved:
        lines.append("**Metrics where AI improved over baseline:**")
        for item in improved[:20]:
            lines.append(f"- {item}")
    if regressed:
        lines.append("\n**Metrics where AI regressed vs baseline:**")
        for item in regressed[:20]:
            lines.append(f"- {item}")
    net = len(improved) - len(regressed)
    lines.append(
        f"\n**Overall:** AI improved {len(improved)} metric-scenarios, "
        f"regressed {len(regressed)}.  "
        f"Net score: {net:+d}.  "
        + ("Recommend deploying AI controller." if net > 0
           else "Further training recommended.")
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report written: {out_path}")


# ── Bar-chart visualisations ──────────────────────────────────────────────────

def write_charts(grouped: Dict, charts_dir: Path = None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("  [skip charts] matplotlib not installed")
        return

    charts_dir = charts_dir or _CHARTS
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Collect data by metric
    metrics_data: Dict[str, Dict] = {}
    for (scenario, level), controllers in grouped.items():
        baseline = controllers.get("baseline", {})
        ai       = controllers.get("ai", {})
        for metric in baseline:
            if metric not in metrics_data:
                metrics_data[metric] = {"labels": [], "baseline": [], "ai": []}
            metrics_data[metric]["labels"].append(f"{scenario}\n{level}")
            metrics_data[metric]["baseline"].append(baseline.get(metric, 0.0))
            metrics_data[metric]["ai"].append(ai.get(metric, 0.0))

    for metric, data in metrics_data.items():
        labels   = data["labels"]
        baseline = data["baseline"]
        ai_vals  = data["ai"]
        n        = len(labels)
        x        = list(range(n))

        fig, ax = plt.subplots(figsize=(max(6, n * 1.5), 5))
        w = 0.35
        ax.bar([xi - w / 2 for xi in x], baseline, w, label="Baseline", color="#4c72b0")
        ax.bar([xi + w / 2 for xi in x], ai_vals,  w, label="AI v6",    color="#dd8452")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} — Baseline vs AI (v6)")
        ax.legend()
        plt.tight_layout()
        chart_path = charts_dir / f"{metric}.png"
        plt.savefig(chart_path, dpi=100)
        plt.close(fig)

    print(f"  Charts written to {charts_dir}")


# ── Public entry point ────────────────────────────────────────────────────────

def run_analysis(run_manifest: List[dict], sim_duration_s: int = 3600):
    """
    Parameters
    ----------
    run_manifest : list of {scenario, traffic_level, controller, output_dir}
    """
    csv_rows  = []
    grouped   = {}

    for entry in run_manifest:
        scenario  = entry["scenario"]
        level     = entry["traffic_level"]
        ctrl      = entry["controller"]
        out_dir   = entry["output_dir"]

        try:
            metrics = _extract_all_metrics(out_dir, sim_duration_s)
        except Exception as e:
            print(f"  [WARN] Could not extract metrics from {out_dir}: {e}")
            continue

        key = (scenario, level)
        grouped.setdefault(key, {})
        grouped[key][ctrl] = metrics

        for metric, value in metrics.items():
            csv_rows.append({
                "scenario": scenario, "traffic_level": level,
                "controller": ctrl,  "metric": metric, "value": value,
            })

    write_results_csv(csv_rows)
    write_summary_report(grouped)
    write_charts(grouped)
    return grouped
