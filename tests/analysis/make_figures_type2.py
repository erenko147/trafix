"""
Generate all comparison figures for Test Type 2: Roundabout vs Standard junction comparison.

4 controllers compared at 3 traffic levels (low / medium / high):
  roundabout_fixed  — Turkish-style roundabout, Webster 86 s fixed-timing TLS
  standard_fixed    — Standard cross-intersection, SUMO built-in fixed timing
  standard_ep1000   — Standard cross-intersection, AI TraFix v6 @ ep1000
  standard_ep2000   — Standard cross-intersection, AI TraFix v6 @ ep2000 (final)

Run standalone:
    python tests/analysis/make_figures_type2.py

Figures produced in tests/reports/test_type_2/charts/:
  {metric}.png                  — per-metric grouped bar chart  (12 figures)
  _heatmap_vs_roundabout.png    — % improvement heatmap vs roundabout baseline
  _summary_improvement.png      — average % improvement bar chart
  _cars_completed.png           — completed vs still-in-network stacked bars
  _winner_map.png               — which controller wins per metric × traffic level
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

_TESTS_DIR    = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _TESTS_DIR.parent
sys.path.insert(0, str(_TESTS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.junction_compare import _extract

MANIFEST_PATH = _TESTS_DIR / "outputs" / "test_type_2" / "manifest.json"
CHARTS_DIR    = _TESTS_DIR / "reports"  / "test_type_2" / "charts"

# ── Style constants ────────────────────────────────────────────────────────────

C = {
    "roundabout_fixed":  "#8c4e1e",
    "standard_fixed":    "#5b6f8a",
    "standard_ep1000":   "#e07b39",
    "standard_ep2000":   "#3a9e6e",
}

CTRL_LABEL = {
    "roundabout_fixed":  "Roundabout (Webster fixed)",
    "standard_fixed":    "Standard (fixed timing)",
    "standard_ep1000":   "Standard AI ep1000",
    "standard_ep2000":   "Standard AI ep2000 (final)",
}

CTRL_ORDER = ["roundabout_fixed", "standard_fixed", "standard_ep1000", "standard_ep2000"]

LEVELS = ["low", "medium", "high"]
LEVEL_LABEL = {
    "low":    "Low traffic\n(~200 veh/hr)",
    "medium": "Medium traffic\n(~500 veh/hr)",
    "high":   "High traffic\n(~950 veh/hr)",
}

LOWER_BETTER = {
    "waiting_time_s": True, "travel_time_s": True, "time_loss_s": True,
    "queue_length": True,   "throughput_veh_hr": False, "network_speed_ms": False,
    "co2_per_vehicle_mg": True, "fuel_per_vehicle_L": True,
    "stops_per_vehicle": True, "fairness_variance": True,
    "arrived_vehicles": False, "vehicles_still_running": True,
    "not_inserted": True, "cars_not_completed": True, "completion_rate_pct": False,
    "NOx_total_mg": True, "teleports": True,
}

METRIC_LABEL = {
    "waiting_time_s":       "Mean Waiting Time (s) — incl. stuck cars",
    "travel_time_s":        "Mean Travel Time (s) — incl. stuck cars",
    "time_loss_s":          "Mean Time Loss (s) — incl. stuck cars",
    "queue_length":         "Queue Length (slow-moving veh / junction, < 5 km/h)",
    "throughput_veh_hr":    "Throughput (veh / hr)",
    "network_speed_ms":     "Mean Network Speed (m/s)",
    "co2_per_vehicle_mg":   "CO₂ per Vehicle (mg)",
    "fuel_per_vehicle_L":   "Fuel per Vehicle (L)",
    "stops_per_vehicle":    "Stops per Vehicle",
    "fairness_variance":    "Fairness Variance (waiting time)",
    "arrived_vehicles":     "Cars Completed Trip",
    "vehicles_still_running": "Cars Still in Network at End",
    "not_inserted":         "Cars Never Inserted (fringe gridlock)",
    "cars_not_completed":   "Cars NOT Completed (stuck + never inserted)",
    "completion_rate_pct":  "Trip Completion Rate (%)",
    "NOx_total_mg":         "NOx Total (mg)",
    "teleports":            "Teleports",
}

KEY_METRICS = [
    "completion_rate_pct", "cars_not_completed",
    "waiting_time_s", "travel_time_s", "time_loss_s",
    "queue_length", "network_speed_ms", "throughput_veh_hr",
    "arrived_vehicles", "vehicles_still_running",
    "co2_per_vehicle_mg", "fuel_per_vehicle_L",
    "stops_per_vehicle", "fairness_variance",
]


# ── Load data ──────────────────────────────────────────────────────────────────

def load_data(sim_s: int = 3600) -> dict:
    """Returns data[level][controller] = {metric: value}"""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    data = {}
    for entry in manifest:
        level = entry["traffic_level"]
        ctrl  = entry["controller"]
        try:
            metrics = _extract(entry["output_dir"], sim_s)
            data.setdefault(level, {})[ctrl] = metrics
        except Exception as e:
            print(f"  [WARN] {level}/{ctrl}: {e}")
    return data


# ── Helper ─────────────────────────────────────────────────────────────────────

def pct_improvement(ref_val, ctr_val, lower_better):
    """% by which ctr_val beats ref_val (positive = better than reference)."""
    if ref_val == 0:
        return 0.0
    raw = (ref_val - ctr_val) / abs(ref_val) * 100.0
    return raw if lower_better else -raw


# ── Figure 1: Per-metric grouped bar charts ────────────────────────────────────

def fig_per_metric(data: dict, charts_dir: Path):
    """One figure per metric, 4 bars (controllers) × 3 groups (traffic levels)."""
    xi     = np.arange(len(LEVELS))
    xlbls  = [LEVEL_LABEL[lv] for lv in LEVELS]
    n_ctrl = len(CTRL_ORDER)
    w      = 0.20

    for metric in KEY_METRICS:
        label = METRIC_LABEL.get(metric, metric)
        fig, ax = plt.subplots(figsize=(10, 5))

        for i, ctrl in enumerate(CTRL_ORDER):
            vals   = [data.get(lv, {}).get(ctrl, {}).get(metric, 0.0) for lv in LEVELS]
            offset = (i - (n_ctrl - 1) / 2) * w
            bars   = ax.bar(xi + offset, vals, w,
                            label=CTRL_LABEL[ctrl], color=C[ctrl],
                            edgecolor="white", linewidth=0.4, zorder=3)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.008,
                            f"{v:.3g}", ha="center", va="bottom",
                            fontsize=6, color="#333", zorder=4)

        ax.set_xticks(xi)
        ax.set_xticklabels(xlbls, fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(
            f"{label}\nRoundabout (fixed) vs Standard (fixed) vs Standard AI ep1000 vs ep2000",
            fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.yaxis.grid(True, ls="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        lb = LOWER_BETTER.get(metric, True)
        ax.annotate("↓ lower is better" if lb else "↑ higher is better",
                    xy=(0.99, 0.97), xycoords="axes fraction",
                    fontsize=7, color="#888", va="top", ha="right")

        plt.tight_layout()
        plt.savefig(charts_dir / f"{metric}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {metric}.png")


# ── Figure 2: % improvement heatmap vs roundabout_fixed ───────────────────────

def fig_improvement_heatmap(data: dict, charts_dir: Path):
    """
    Rows = metrics, Columns = traffic levels.
    Each cell = % improvement of that controller vs roundabout_fixed.
    One heatmap per non-roundabout controller, all on one figure.
    """
    compare_ctrls = ["standard_fixed", "standard_ep1000", "standard_ep2000"]
    metrics  = [m for m in KEY_METRICS
                if m not in ("arrived_vehicles", "vehicles_still_running", "teleports")]
    m_labels = [METRIC_LABEL[m].replace(" (", "\n(") for m in metrics]

    fig, axes = plt.subplots(1, len(compare_ctrls),
                             figsize=(6 * len(compare_ctrls), len(metrics) * 0.55 + 2),
                             sharey=True)

    for ax, ctrl in zip(axes, compare_ctrls):
        matrix = np.zeros((len(metrics), len(LEVELS)))
        for j, lv in enumerate(LEVELS):
            ref = data.get(lv, {}).get("roundabout_fixed", {})
            ctr = data.get(lv, {}).get(ctrl, {})
            for i, metric in enumerate(metrics):
                r = ref.get(metric, 0.0)
                c = ctr.get(metric, 0.0)
                matrix[i, j] = pct_improvement(r, c, LOWER_BETTER.get(metric, True))

        vmax = min(80, max(10, float(np.percentile(np.abs(matrix), 95))))
        im   = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(LEVELS)))
        ax.set_xticklabels([lv.capitalize() for lv in LEVELS], fontsize=9)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(m_labels, fontsize=8)
        ax.set_title(CTRL_LABEL[ctrl], fontsize=10, fontweight="bold",
                     color=C[ctrl], pad=6)

        for i in range(len(metrics)):
            for j in range(len(LEVELS)):
                v = matrix[i, j]
                ax.text(j, i, f"{v:+.0f}%", ha="center", va="center",
                        fontsize=7.5,
                        color="black" if abs(v) < vmax * 0.6 else "white",
                        fontweight="bold" if abs(v) > 20 else "normal")

        plt.colorbar(im, ax=ax, label="% improvement\nvs roundabout", shrink=0.7)

    fig.suptitle(
        "% Improvement vs Roundabout (Webster Fixed) — green = better, red = worse",
        fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(charts_dir / "_heatmap_vs_roundabout.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  _heatmap_vs_roundabout.png")


# ── Figure 3: Average % improvement summary bar chart ────────────────────────

def fig_summary_improvement(data: dict, charts_dir: Path):
    """
    For each non-roundabout controller, show mean % improvement over roundabout_fixed
    averaged across the 3 traffic levels, for each key metric.
    Y-axis is clamped to ±120 %; outliers are annotated with their true value.
    """
    metrics   = ["waiting_time_s", "time_loss_s", "travel_time_s",
                 "queue_length", "network_speed_ms", "throughput_veh_hr",
                 "co2_per_vehicle_mg", "fuel_per_vehicle_L",
                 "stops_per_vehicle", "fairness_variance"]
    m_labels  = [METRIC_LABEL[m].split(" (")[0] for m in metrics]
    compare_ctrls = ["standard_fixed", "standard_ep1000", "standard_ep2000"]
    CLAMP = 120.0   # y-axis limit; values beyond this are clamped + annotated

    def avg_improvement(ctrl):
        results = []
        for metric in metrics:
            lb   = LOWER_BETTER.get(metric, True)
            pcts = []
            for lv in LEVELS:
                ref = data.get(lv, {}).get("roundabout_fixed", {}).get(metric, 0.0)
                ctr = data.get(lv, {}).get(ctrl,              {}).get(metric, 0.0)
                pcts.append(pct_improvement(ref, ctr, lb))
            results.append(np.mean(pcts))
        return results

    fig, ax = plt.subplots(figsize=(14, 6))
    xi = np.arange(len(metrics))
    w  = 0.24
    nc = len(compare_ctrls)

    for i, ctrl in enumerate(compare_ctrls):
        raw_vals = avg_improvement(ctrl)
        offset   = (i - (nc - 1) / 2) * w
        for xi_j, (raw_v) in enumerate(raw_vals):
            clamped  = max(-CLAMP, min(CLAMP, raw_v))
            bar = ax.bar(xi[xi_j] + offset, clamped, w,
                         color=C[ctrl], edgecolor="white",
                         label=CTRL_LABEL[ctrl] if xi_j == 0 else "_nolegend_")
            # Label: show true value (not clamped display value)
            is_clamped = abs(raw_v) > CLAMP
            label_v = clamped + (1.5 if clamped >= 0 else -4.0)
            ax.text(xi[xi_j] + offset, label_v,
                    f"{raw_v:+.0f}%{'*' if is_clamped else ''}",
                    ha="center", va="bottom", fontsize=6.5,
                    fontweight="bold" if is_clamped else "normal",
                    color="#cc2222" if is_clamped and raw_v < 0 else
                          "#1a7a3a" if is_clamped and raw_v > 0 else "black")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(-CLAMP * 1.15, CLAMP * 1.15)
    ax.set_xticks(xi)
    ax.set_xticklabels(m_labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Mean % improvement vs roundabout baseline\n(avg across low / medium / high)", fontsize=9)
    ax.set_title(
        "Standard Intersection (Fixed & AI) vs Turkish Roundabout Baseline\n"
        "Average improvement across all traffic levels  (* = axis clamped to ±120 %)",
        fontsize=11, fontweight="bold")
    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=9)
    ax.yaxis.grid(True, ls="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(charts_dir / "_summary_improvement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  _summary_improvement.png")


# ── Figure 4: Cars completed — stacked bars ────────────────────────────────────

def fig_cars_completed(data: dict, charts_dir: Path):
    xi     = np.arange(len(LEVELS))
    xlbls  = [LEVEL_LABEL[lv] for lv in LEVELS]
    n_ctrl = len(CTRL_ORDER)
    w      = 0.20

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, ctrl in enumerate(CTRL_ORDER):
        arrived  = [data.get(lv, {}).get(ctrl, {}).get("arrived_vehicles", 0)        for lv in LEVELS]
        still_in = [data.get(lv, {}).get(ctrl, {}).get("vehicles_still_running", 0)  for lv in LEVELS]
        never_in = [data.get(lv, {}).get(ctrl, {}).get("not_inserted", 0)            for lv in LEVELS]
        offset   = (i - (n_ctrl - 1) / 2) * w

        ax.bar(xi + offset, arrived, w,
               label=f"{CTRL_LABEL[ctrl]} — completed",
               color=C[ctrl], edgecolor="white", zorder=3)
        ax.bar(xi + offset, still_in, w, bottom=arrived,
               color=C[ctrl], alpha=0.35, edgecolor="white", hatch="//", zorder=3)
        bottom2 = [a + s for a, s in zip(arrived, still_in)]
        ax.bar(xi + offset, never_in, w, bottom=bottom2,
               color=C[ctrl], alpha=0.18, edgecolor="white", hatch="xx", zorder=3)

        for j in range(len(LEVELS)):
            total = arrived[j] + still_in[j] + never_in[j]
            if total > 0:
                pct = arrived[j] / total * 100.0
                ax.text(xi[j] + offset, total + total * 0.01,
                        f"{int(arrived[j])}\n{pct:.0f}%", ha="center", va="bottom",
                        fontsize=5.5, color=C[ctrl], fontweight="bold", zorder=4)

    ax.set_xticks(xi)
    ax.set_xticklabels(xlbls, fontsize=9)
    ax.set_ylabel("Vehicles", fontsize=10)
    ax.set_title(
        "Cars Completed (solid) vs Stuck in Network (//) vs Never Inserted (xx)\n"
        "Numbers above bars = completed count and completion %",
        fontsize=10, fontweight="bold")

    handles = [mpatches.Patch(color=C[c], label=CTRL_LABEL[c]) for c in CTRL_ORDER]
    handles += [mpatches.Patch(facecolor="grey", alpha=0.35, hatch="//", label="Stuck in network"),
                mpatches.Patch(facecolor="grey", alpha=0.18, hatch="xx", label="Never inserted")]
    ax.legend(handles=handles, fontsize=8, loc="upper left", ncol=2)
    ax.yaxis.grid(True, ls="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(charts_dir / "_cars_completed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  _cars_completed.png")


# ── Figure 5: Winner map ───────────────────────────────────────────────────────

def fig_winner_map(data: dict, charts_dir: Path):
    """
    Grid: metric (rows) × traffic level (cols).
    Each cell coloured and labelled by the winning controller.
    """
    metrics  = [m for m in KEY_METRICS if m not in ("teleports",)]
    m_labels = [METRIC_LABEL[m].split(" (")[0] for m in metrics]

    # Map controller → index for colormap
    ctrl_idx = {c: i for i, c in enumerate(CTRL_ORDER)}
    cmap     = mcolors.ListedColormap([C[c] for c in CTRL_ORDER])

    abbrev = {
        "roundabout_fixed":  "RND",
        "standard_fixed":    "STD",
        "standard_ep1000":   "AI-1K",
        "standard_ep2000":   "AI-2K",
    }

    grid = np.full((len(metrics), len(LEVELS)), np.nan)
    for mi, metric in enumerate(metrics):
        lb = LOWER_BETTER.get(metric, True)
        for li, lv in enumerate(LEVELS):
            level_data = data.get(lv, {})
            candidates = {c: level_data[c][metric]
                          for c in CTRL_ORDER
                          if c in level_data and metric in level_data[c]}
            if not candidates:
                continue
            winner = min(candidates, key=candidates.__getitem__) if lb \
                else max(candidates, key=candidates.__getitem__)
            grid[mi, li] = ctrl_idx[winner]

    fig, ax = plt.subplots(figsize=(8, len(metrics) * 0.58 + 1.8))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=len(CTRL_ORDER) - 1,
              aspect="auto", interpolation="nearest")

    for mi in range(len(metrics)):
        for li in range(len(LEVELS)):
            v = grid[mi, li]
            if not np.isnan(v):
                ctrl = CTRL_ORDER[int(v)]
                ax.text(li, mi, abbrev[ctrl],
                        ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")

    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels([lv.capitalize() for lv in LEVELS], fontsize=10)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(m_labels, fontsize=8)
    ax.set_xlabel("Traffic Level", fontsize=9)
    ax.set_title("Best Controller per Metric × Traffic Level", fontsize=11, fontweight="bold")

    patches = [mpatches.Patch(color=C[c], label=CTRL_LABEL[c]) for c in CTRL_ORDER]
    ax.legend(handles=patches, fontsize=8, loc="upper right",
              bbox_to_anchor=(1.55, 1.0), framealpha=0.9)

    plt.tight_layout()
    plt.savefig(charts_dir / "_winner_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  _winner_map.png")


# ── Public entry point ─────────────────────────────────────────────────────────

def make_all_figures(data: dict, charts_dir: Path):
    charts_dir.mkdir(parents=True, exist_ok=True)

    print("Per-metric bar charts:")
    fig_per_metric(data, charts_dir)

    print("\nSummary figures:")
    fig_improvement_heatmap(data, charts_dir)
    fig_summary_improvement(data, charts_dir)
    fig_cars_completed(data, charts_dir)
    fig_winner_map(data, charts_dir)

    charts = sorted(charts_dir.glob("*.png"))
    print(f"\n{len(charts)} figures written to {charts_dir}")


# ── Standalone ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading manifest: {MANIFEST_PATH}")
    data = load_data(sim_s=3600)
    levels_found = sorted(data.keys())
    ctrls_found  = sorted({c for lv in data.values() for c in lv})
    print(f"Loaded: levels={levels_found}  controllers={ctrls_found}")
    make_all_figures(data, CHARTS_DIR)
