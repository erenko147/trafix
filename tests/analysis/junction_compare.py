"""
Data extraction and CSV output for Test Type 2: 4-way junction comparison.
Figure generation is handled by make_figures_type2.py.

Manifest entry format:
  {"controller": str, "traffic_level": str, "output_dir": str,
   "demand_hash": str, "source": "fresh"|"reused"}
"""

import csv
import json
from pathlib import Path
from typing import Dict, Optional

_TESTS_DIR = Path(__file__).resolve().parents[1]


def _extract(out_dir: str, sim_s: int = 3600) -> Dict[str, float]:
    d = Path(out_dir)
    from metrics.travel_time       import compute_travel_time
    from metrics.emissions         import compute_emissions
    from metrics.waiting_time      import compute_waiting_time
    from metrics.queue_length      import compute_queue_length
    from metrics.throughput        import compute_throughput
    from metrics.network_speed     import compute_network_speed
    from metrics.teleports         import compute_teleports
    from metrics.time_loss         import compute_time_loss
    from metrics.fuel_consumption  import compute_fuel_consumption
    from metrics.pollutants        import compute_pollutants
    from metrics.stops_per_vehicle import compute_stops_per_vehicle
    from metrics.junction_fairness import compute_junction_fairness

    m: Dict[str, float] = {}
    m["travel_time_s"]       = compute_travel_time(d)["mean_travel_time_s"]
    m["waiting_time_s"]      = compute_waiting_time(d)["mean_waiting_time_s"]
    m["time_loss_s"]         = compute_time_loss(d)["mean_time_loss_s"]
    # crawl-aware queue (< 5 km/h), fair to roundabouts which never fully stop
    m["queue_length"]        = compute_queue_length(d)["mean_slow_per_junction"]
    tp = compute_throughput(d, sim_s)
    m["arrived_vehicles"]    = float(tp["arrived_vehicles"])
    m["vehicles_still_running"] = float(tp["vehicles_still_running"])
    m["not_inserted"]        = float(tp["not_inserted"])
    m["cars_not_completed"]  = float(tp["cars_not_completed"])
    m["completion_rate_pct"] = float(tp["completion_rate"]) * 100.0
    m["throughput_veh_hr"]   = tp["throughput_veh_per_hr"]
    m["network_speed_ms"]    = compute_network_speed(d)["mean_speed_ms"]
    m["teleports"]           = float(compute_teleports(d)["total_teleports"])
    m["co2_per_vehicle_mg"]  = compute_emissions(d)["mean_CO2_per_vehicle_mg"]
    m["fuel_per_vehicle_L"]  = compute_fuel_consumption(d, volumetric=True)["mean_fuel_per_vehicle_L"]
    m["NOx_total_mg"]        = compute_pollutants(d)["totals_mg"]["NOx"]
    m["stops_per_vehicle"]   = compute_stops_per_vehicle(d)["avg_stops_per_vehicle"]
    m["fairness_variance"]   = compute_junction_fairness(d)["network_wide_variance"]
    return m


def _gridlock_flag(out_dir: str) -> bool:
    p = Path(out_dir) / "gridlock.json"
    if not p.exists():
        return False
    return json.loads(p.read_text())["gridlocked"]


# ── Markdown report generation ─────────────────────────────────────────────────
# (Previously these .md files were stale, hand-edited artifacts with no generator;
#  they are now produced from the fresh CSV every run.)

_CTRL_ORDER = ["roundabout_fixed", "standard_fixed", "standard_ep1000", "standard_ep2000"]
_LEVELS     = ["low", "medium", "high"]

# (metric_key, label, lower_is_better) — completion headline first
_REPORT_METRICS = [
    ("completion_rate_pct",     "Trip Completion (%)",          False),
    ("cars_not_completed",      "Cars NOT Completed",           True),
    ("arrived_vehicles",        "Cars Completed Trip",          False),
    ("vehicles_still_running",  "Cars Stuck in Network",        True),
    ("not_inserted",            "Cars Never Inserted",          True),
    ("throughput_veh_hr",       "Throughput (veh/hr)",          False),
    ("waiting_time_s",          "Waiting Time (s)",             True),
    ("travel_time_s",           "Travel Time (s)",              True),
    ("time_loss_s",             "Time Loss (s)",                True),
    ("queue_length",            "Queue (slow veh/junction)",    True),
    ("network_speed_ms",        "Network Speed (m/s)",          False),
    ("stops_per_vehicle",       "Stops per Vehicle",            True),
    ("co2_per_vehicle_mg",      "CO₂ per Vehicle (mg)",    True),
    ("fuel_per_vehicle_L",      "Fuel per Vehicle (L)",         True),
    ("NOx_total_mg",            "NOx Total (mg)",               True),
    ("fairness_variance",       "Fairness Variance",            True),
]


def _fmt(v: float) -> str:
    if v is None:
        return "—"
    a = abs(v)
    if a >= 1e4:
        return f"{v:.4g}"
    if a >= 100:
        return f"{v:.1f}"
    return f"{v:.4g}"


def _winner(vals: dict, lower_better: bool):
    valid = {c: x for c, x in vals.items() if x is not None}
    if not valid:
        return None
    return (min if lower_better else max)(valid, key=valid.get)


def _write_summary_md(data: dict, path: Path):
    L = ["# Test Type 2 — Junction & Controller Comparison",
         "Simulation: 3600 s  |  Route files: shared (type1_low/medium/high.rou.xml)",
         "",
         "> Metrics now include cars still stuck in the network at sim end (no more",
         "> survivorship bias), queue counts crawling cars (< 5 km/h), and AI uses",
         "> greedy argmax (matches production). Teleporting disabled in every run.",
         "",
         "| Key | Controller |", "|-----|-----------|",
         "| roundabout_fixed | Turkish-style roundabout × 5 — Webster 86 s fixed TLS |",
         "| standard_fixed   | Standard cross-intersection × 5 — SUMO fixed timing |",
         "| standard_ep1000  | Standard cross-intersection × 5 — AI TraFix v6 @ ep1000 |",
         "| standard_ep2000  | Standard cross-intersection × 5 — AI TraFix v6 @ ep2000 (final) |",
         ""]
    wins = {c: 0 for c in _CTRL_ORDER}
    for level in _LEVELS:
        lvl = data.get(level, {})
        L += [f"---", f"## Traffic Level: {level.capitalize()}", "",
              "| Metric | " + " | ".join(_CTRL_ORDER) + " | Winner |",
              "|--------|" + "|".join(["--------"] * len(_CTRL_ORDER)) + "|------|"]
        for key, label, lb in _REPORT_METRICS:
            vals = {c: lvl.get(c, {}).get(key) for c in _CTRL_ORDER}
            win = _winner(vals, lb)
            if win:
                wins[win] += 1
            cells = []
            for c in _CTRL_ORDER:
                star = " ★" if c == win else ""
                cells.append(f"{_fmt(vals[c])}{star}")
            L.append(f"| {label} | " + " | ".join(cells) + f" | {win or '—'} |")
        L.append("")
    L += ["---", "## Wins Tally (across all traffic levels)", "",
          "| Controller | Metrics Won |", "|-----------|------------|"]
    for c in _CTRL_ORDER:
        L.append(f"| {c} | {wins[c]} |")
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"  Report: {path}")


def _write_delta_md(data: dict, path: Path, ref: str, compares: list, title: str, note: str):
    L = [f"# Test Type 2 — {title}", "Simulation: 3600 s", "",
         note, ""]
    for level in _LEVELS:
        lvl = data.get(level, {})
        hdr = "| Metric | " + f"{ref} (ref) | "
        sep = "|--------|----------|"
        for c in compares:
            hdr += f"{c} | Δ% | "
            sep += " --- | --- |"
        L += [f"---", f"## {level.capitalize()} traffic", "", hdr.rstrip(), sep.rstrip()]
        for key, label, lb in _REPORT_METRICS:
            rv = lvl.get(ref, {}).get(key)
            row = f"| {label} | {_fmt(rv)} | "
            for c in compares:
                cv = lvl.get(c, {}).get(key)
                if rv in (None, 0) or cv is None:
                    delta = 0.0
                else:
                    raw = (rv - cv) / abs(rv) * 100.0
                    delta = raw if lb else -raw
                mark = "✓" if delta >= 0 else "✗"
                row += f"{_fmt(cv)} | {mark} {delta:+.1f}% | "
            L.append(row.rstrip())
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"  Report: {path}")


def write_markdown_reports(data: dict, reports_dir: Path):
    _write_summary_md(data, reports_dir / "summary.md")
    _write_delta_md(
        data, reports_dir / "comparison_fixed_vs_ai.md",
        ref="roundabout_fixed",
        compares=["standard_fixed", "standard_ep1000", "standard_ep2000"],
        title="Fixed-Timing vs AI Improvement",
        note="Δ% = % by which the controller beats the **roundabout_fixed** reference. Positive = better than roundabout.")
    _write_delta_md(
        data, reports_dir / "comparison_1000_vs_2000.md",
        ref="standard_ep1000",
        compares=["standard_ep2000"],
        title="AI ep1000 vs ep2000 (final)",
        note="Δ% = % by which ep2000 (final) beats ep1000. Positive = final model improved.")


def run_junction_analysis(
    manifest: list,
    sim_duration_s: int = 3600,
    reports_dir: Optional[Path] = None,
) -> Dict:
    """Extract metrics, write CSV, call figure generator. Returns data dict."""
    import sys
    sys.path.insert(0, str(_TESTS_DIR))

    if reports_dir is None:
        reports_dir = _TESTS_DIR / "reports" / "test_type_2"
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # data[level][controller] = {metric: value}
    data: Dict = {}
    csv_rows   = []

    for entry in manifest:
        ctrl   = entry["controller"]
        level  = entry["traffic_level"]
        odir   = entry["output_dir"]
        source = entry.get("source", "unknown")
        gl     = _gridlock_flag(odir)

        try:
            metrics = _extract(odir, sim_duration_s)
        except Exception as e:
            print(f"  [WARN] {ctrl}/{level}: {e}")
            metrics = {}

        data.setdefault(level, {})[ctrl] = metrics

        for metric, value in metrics.items():
            csv_rows.append(dict(
                controller    = ctrl,
                traffic_level = level,
                metric        = metric,
                value         = value,
                gridlock      = int(gl),
                source        = source,
            ))

    # Write CSV
    csv_path = reports_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["controller", "traffic_level", "metric",
                           "value", "gridlock", "source"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  CSV: {csv_path}")

    # Markdown reports (summary + comparisons) — regenerated fresh from this data
    write_markdown_reports(data, reports_dir)

    # Generate figures
    from analysis.make_figures_type2 import make_all_figures
    make_all_figures(data, reports_dir / "charts")

    return data
