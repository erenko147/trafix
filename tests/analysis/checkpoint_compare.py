"""
Analysis and matplotlib figure generation for the 3-controller comparison:
  baseline  vs  ep1000  vs  ep2000 (final)
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

_TESTS_DIR = Path(__file__).resolve().parents[1]
_REPORTS   = _TESTS_DIR / "reports" / "ckpt_compare"

sys.path.insert(0, str(_TESTS_DIR))

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


# ── Metric extraction ─────────────────────────────────────────────────────────

def _extract(out_dir: str, sim_s: int = 3600) -> Dict[str, float]:
    d = Path(out_dir)
    m: Dict[str, float] = {}
    m["travel_time_s"]         = compute_travel_time(d)["mean_travel_time_s"]
    m["waiting_time_s"]        = compute_waiting_time(d)["mean_waiting_time_s"]
    m["time_loss_s"]           = compute_time_loss(d)["mean_time_loss_s"]
    m["queue_length"]          = compute_queue_length(d)["mean_halting_per_junction"]
    tp                         = compute_throughput(d, sim_s)
    m["arrived_vehicles"]      = float(tp["arrived_vehicles"])
    m["vehicles_still_running"]= float(tp["vehicles_still_running"])
    m["total_departed"]        = float(tp["total_departed"])
    m["throughput_veh_hr"]     = tp["throughput_veh_per_hr"]
    m["network_speed_ms"]      = compute_network_speed(d)["mean_speed_ms"]
    m["teleports"]             = float(compute_teleports(d)["total_teleports"])
    m["co2_per_vehicle_mg"]    = compute_emissions(d)["mean_CO2_per_vehicle_mg"]
    m["fuel_per_vehicle_L"]    = compute_fuel_consumption(d, volumetric=True)["mean_fuel_per_vehicle_L"]
    p = compute_pollutants(d)
    m["NOx_total_mg"]          = p["totals_mg"]["NOx"]
    m["stops_per_vehicle"]     = compute_stops_per_vehicle(d)["avg_stops_per_vehicle"]
    m["fairness_variance"]     = compute_junction_fairness(d)["network_wide_variance"]
    return m


# ── Figure generation ─────────────────────────────────────────────────────────

# Which direction is "better" for each metric
_LOWER_BETTER = {
    "travel_time_s": True,  "waiting_time_s": True,  "time_loss_s": True,
    "queue_length": True,   "throughput_veh_hr": False, "network_speed_ms": False,
    "teleports": True,      "co2_per_vehicle_mg": True, "fuel_per_vehicle_L": True,
    "NOx_total_mg": True,   "stops_per_vehicle": True,  "fairness_variance": True,
    "arrived_vehicles": False, "vehicles_still_running": True, "total_departed": False,
}

_METRIC_LABEL = {
    "travel_time_s":         "Travel Time (s)",
    "waiting_time_s":        "Waiting Time (s)",
    "time_loss_s":           "Time Loss (s)",
    "queue_length":          "Queue Length (veh/junction)",
    "throughput_veh_hr":     "Throughput (veh/hr)",
    "arrived_vehicles":      "Cars Completed Trip",
    "vehicles_still_running":"Cars Still in Network at End",
    "total_departed":        "Total Cars Departed",
    "network_speed_ms":      "Network Speed (m/s)",
    "teleports":             "Teleports",
    "co2_per_vehicle_mg":    "CO₂ per Vehicle (mg)",
    "fuel_per_vehicle_L":    "Fuel per Vehicle (L)",
    "NOx_total_mg":          "NOx Total (mg)",
    "stops_per_vehicle":     "Stops per Vehicle",
    "fairness_variance":     "Fairness Variance",
}

# Colours per controller
_COLOURS = {
    "baseline": "#4c72b0",
    "ep1000":   "#dd8452",
    "ep2000":   "#55a868",
}

_CTRL_ORDER  = ["baseline", "ep1000", "ep2000"]
_CTRL_LABELS = {"baseline": "Baseline\n(fixed)", "ep1000": "AI ep1000", "ep2000": "AI ep2000\n(final)"}


def _make_figures(data: Dict, charts_dir: Path):
    """
    data layout:
      data[scenario_key][controller] = {metric: value}
    scenario_key = "type1_low", "type2_morning_peak", etc.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    charts_dir.mkdir(parents=True, exist_ok=True)
    scenarios = sorted(data.keys())
    n_scen    = len(scenarios)
    n_ctrl    = len(_CTRL_ORDER)
    x         = np.arange(n_scen)
    w         = 0.25

    for metric, ylabel in _METRIC_LABEL.items():
        fig, ax = plt.subplots(figsize=(max(10, n_scen * 1.8), 5))

        for i, ctrl in enumerate(_CTRL_ORDER):
            vals = [data[sc].get(ctrl, {}).get(metric, 0.0) for sc in scenarios]
            offset = (i - 1) * w
            bars = ax.bar(x + offset, vals, w,
                          label=_CTRL_LABELS[ctrl],
                          color=_COLOURS[ctrl],
                          edgecolor="white", linewidth=0.5)
            # value labels on top
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.01,
                            f"{v:.3g}", ha="center", va="bottom",
                            fontsize=6.5, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{ylabel} — Baseline vs AI ep1000 vs AI ep2000",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        # shade "better" direction
        lb = _LOWER_BETTER[metric]
        ax.annotate("← lower is better" if lb else "↑ higher is better",
                    xy=(0.01, 0.97), xycoords="axes fraction",
                    fontsize=7, color="#888888", va="top")

        plt.tight_layout()
        plt.savefig(charts_dir / f"{metric}.png", dpi=130)
        plt.close(fig)
        print(f"  {metric}.png")

    # ── Summary overview figure ───────────────────────────────────────────────
    # % improvement of each AI over baseline, averaged across scenarios
    key_metrics = ["waiting_time_s", "travel_time_s", "time_loss_s",
                   "queue_length", "throughput_veh_hr", "network_speed_ms",
                   "co2_per_vehicle_mg", "fuel_per_vehicle_L",
                   "stops_per_vehicle", "fairness_variance"]

    ai_ctrls = ["ep1000", "ep2000"]
    improvements: Dict[str, List[float]] = {c: [] for c in ai_ctrls}

    for metric in key_metrics:
        lb = _LOWER_BETTER[metric]
        for ctrl in ai_ctrls:
            pcts = []
            for sc in scenarios:
                bval = data[sc].get("baseline", {}).get(metric, 0.0)
                aval = data[sc].get(ctrl, {}).get(metric, 0.0)
                if bval == 0:
                    continue
                raw = (bval - aval) / bval * 100.0
                pcts.append(raw if lb else -raw)
            improvements[ctrl].append(
                sum(pcts) / len(pcts) if pcts else 0.0
            )

    fig, ax = plt.subplots(figsize=(13, 5))
    xi = np.arange(len(key_metrics))
    w2 = 0.35
    for i, ctrl in enumerate(ai_ctrls):
        vals = improvements[ctrl]
        offset = (i - 0.5) * w2
        bars = ax.bar(xi + offset, vals, w2,
                      label=_CTRL_LABELS[ctrl],
                      color=_COLOURS[ctrl],
                      edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.3 if v >= 0 else -1.2),
                    f"{v:+.1f}%", ha="center", va="bottom",
                    fontsize=7, color="#333333")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xi)
    ax.set_xticklabels([_METRIC_LABEL[m].split(" (")[0] for m in key_metrics],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Mean % improvement vs baseline\n(positive = better)", fontsize=9)
    ax.set_title("TraFix v6 — Average Improvement over Fixed-Timing Baseline\n"
                 "ep1000 vs ep2000 (final), across all 7 scenarios",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(charts_dir / "_summary_improvement.png", dpi=130)
    plt.close(fig)
    print("  _summary_improvement.png")


# ── Markdown report ───────────────────────────────────────────────────────────

def _write_report(data: Dict, out_path: Path, sim_s: int):
    lines = [
        "# TraFix v6 — Checkpoint Comparison",
        f"Simulation duration: {sim_s} s | Controllers: baseline · ep1000 · ep2000 (final)\n",
    ]
    for sc in sorted(data.keys()):
        ctrls = data[sc]
        lines.append(f"\n## {sc.replace('_', ' ')}\n")
        metrics = sorted(next(iter(ctrls.values())).keys())
        header = "| Metric | Baseline | ep1000 | ep2000 | ep1000 Δ% | ep2000 Δ% |"
        lines.append(header)
        lines.append("|--------|----------|--------|--------|-----------|-----------|")
        for metric in metrics:
            bval = ctrls.get("baseline", {}).get(metric, 0.0)
            v10  = ctrls.get("ep1000",   {}).get(metric, 0.0)
            v20  = ctrls.get("ep2000",   {}).get(metric, 0.0)
            lb   = _LOWER_BETTER.get(metric, True)
            def pct(a):
                if bval == 0: return 0.0
                raw = (bval - a) / bval * 100.0
                return raw if lb else -raw
            p10, p20 = pct(v10), pct(v20)
            s10 = f"{'✓' if p10 >= 0 else '✗'} {p10:+.1f}%"
            s20 = f"{'✓' if p20 >= 0 else '✗'} {p20:+.1f}%"
            lines.append(
                f"| {metric} | {bval:.4g} | {v10:.4g} | {v20:.4g} | {s10} | {s20} |"
            )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {out_path}")


# ── CSV ───────────────────────────────────────────────────────────────────────

def _write_csv(rows: list, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["test_type","scenario","controller","metric","value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV: {out_path}")


# ── Public entry point ────────────────────────────────────────────────────────

def run_checkpoint_analysis(manifest: list, sim_duration_s: int = 3600):
    _REPORTS.mkdir(parents=True, exist_ok=True)

    data: Dict = {}    # data[scenario_key][controller] = {metric: value}
    csv_rows   = []

    for entry in manifest:
        sc_key = f"{entry['test_type']}_{entry['scenario']}"
        ctrl   = entry["controller"]
        try:
            metrics = _extract(entry["output_dir"], sim_duration_s)
        except Exception as e:
            print(f"  [WARN] {sc_key}/{ctrl}: {e}")
            continue

        data.setdefault(sc_key, {})[ctrl] = metrics
        for metric, value in metrics.items():
            csv_rows.append(dict(
                test_type  = entry["test_type"],
                scenario   = entry["scenario"],
                controller = ctrl,
                metric     = metric,
                value      = value,
            ))

    _write_csv(csv_rows, _REPORTS / "results.csv")
    _write_report(data, _REPORTS / "summary.md", sim_duration_s)
    _make_figures(data, _REPORTS / "charts")
    return data
