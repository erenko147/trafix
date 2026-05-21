"""
Generate all comparison figures for the TraFix v6 checkpoint evaluation.
Covers all 13 scenarios: 7 in-distribution + 6 unseen OOD.

Run:
    python tests/analysis/make_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_TESTS_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _TESTS_DIR.parent
sys.path.insert(0, str(_TESTS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.checkpoint_compare import _extract

MANIFEST_PATH = _TESTS_DIR / "outputs" / "ckpt_compare" / "manifest.json"
CHARTS_DIR    = _TESTS_DIR / "reports" / "ckpt_compare" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Style constants ────────────────────────────────────────────────────────────
C = {
    "baseline": "#5b6f8a",
    "ep1000":   "#e07b39",
    "ep2000":   "#3a9e6e",
}
CTRL_LABEL = {
    "baseline": "Baseline (fixed timing)",
    "ep1000":   "AI ep1000",
    "ep2000":   "AI ep2000 (final)",
}

# Scenario display order and labels
IN_DIST = [
    ("type1_low",          "Low traffic\n(200 veh/hr)"),
    ("type1_medium",       "Medium traffic\n(500 veh/hr)"),
    ("type1_high",         "High traffic\n(950 veh/hr)"),
    ("type2_morning_peak", "Morning peak\n(inbound heavy)"),
    ("type2_evening_peak", "Evening peak\n(outbound heavy)"),
    ("type2_incident",     "Incident\n(junction closed)"),
    ("type2_pulse",        "Pulse\n(demand burst)"),
]
UNSEEN = [
    ("unseen_supersaturation",     "Supersaturation\n(1500 veh/hr)"),
    ("unseen_stadium_exit",        "Stadium exit\n(J4 concentrated)"),
    ("unseen_peak_plus_incident",  "Peak + Incident\n(combined stress)"),
    ("unseen_oscillating",         "Oscillating\n(flip every 10 min)"),
    ("unseen_tidal_ramp",          "Tidal ramp\n(3-phase gradual)"),
    ("unseen_bidirectional_peak",  "Bidirectional\n(equal in+out)"),
]
ALL_SCENARIOS = IN_DIST + UNSEEN

LOWER_BETTER = {
    "waiting_time_s": True, "travel_time_s": True, "time_loss_s": True,
    "queue_length": True,   "throughput_veh_hr": False, "network_speed_ms": False,
    "co2_per_vehicle_mg": True, "fuel_per_vehicle_L": True,
    "stops_per_vehicle": True, "fairness_variance": True,
    "arrived_vehicles": False, "vehicles_still_running": True,
    "NOx_total_mg": True, "teleports": True, "total_departed": False,
}

METRIC_LABEL = {
    "waiting_time_s":      "Mean Waiting Time (s)",
    "travel_time_s":       "Mean Travel Time (s)",
    "time_loss_s":         "Mean Time Loss (s)",
    "queue_length":        "Queue Length (halting veh / junction)",
    "throughput_veh_hr":   "Throughput (veh / hr)",
    "network_speed_ms":    "Mean Network Speed (m/s)",
    "co2_per_vehicle_mg":  "CO₂ per Vehicle (mg)",
    "fuel_per_vehicle_L":  "Fuel per Vehicle (L)",
    "stops_per_vehicle":   "Stops per Vehicle",
    "fairness_variance":   "Fairness Variance (waiting time)",
    "arrived_vehicles":    "Cars Completed Trip",
    "vehicles_still_running": "Cars Still in Network at End",
    "NOx_total_mg":        "NOx Total (mg)",
    "teleports":           "Teleports",
}

# Key metrics shown in the per-metric bar charts
KEY_METRICS = [
    "waiting_time_s", "travel_time_s", "time_loss_s",
    "queue_length", "network_speed_ms", "throughput_veh_hr",
    "arrived_vehicles", "vehicles_still_running",
    "co2_per_vehicle_mg", "fuel_per_vehicle_L",
    "stops_per_vehicle", "fairness_variance",
]


# ── Load data ──────────────────────────────────────────────────────────────────

def load_data(sim_s: int = 3600) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    data = {}
    for entry in manifest:
        key = f"{entry['test_type']}_{entry['scenario']}"
        ctrl = entry["controller"]
        try:
            metrics = _extract(entry["output_dir"], sim_s)
            data.setdefault(key, {})[ctrl] = metrics
        except Exception as e:
            print(f"  [WARN] {key}/{ctrl}: {e}")
    return data


# ── Helper ─────────────────────────────────────────────────────────────────────

def pct_improvement(baseline, ai, lower_better):
    if baseline == 0:
        return 0.0
    raw = (baseline - ai) / baseline * 100.0
    return raw if lower_better else -raw


def _divider_x(ax, x_pos, label="OOD →"):
    ax.axvline(x=x_pos - 0.5, color="#cc3333", lw=1.5, ls="--", alpha=0.7)
    ax.text(x_pos - 0.48, ax.get_ylim()[1] * 0.97, label,
            color="#cc3333", fontsize=8, va="top", ha="left")


# ── Figure 1: Per-metric grouped bar charts (13 scenarios) ────────────────────

def fig_per_metric(data: dict):
    sc_keys  = [k for k, _ in ALL_SCENARIOS]
    sc_labels = [l for _, l in ALL_SCENARIOS]
    n = len(sc_keys)
    n_in = len(IN_DIST)

    for metric in KEY_METRICS:
        label = METRIC_LABEL.get(metric, metric)
        fig, ax = plt.subplots(figsize=(16, 5))

        w = 0.26
        xi = np.arange(n)

        for i, ctrl in enumerate(["baseline", "ep1000", "ep2000"]):
            vals = [data.get(sc, {}).get(ctrl, {}).get(metric, 0.0) for sc in sc_keys]
            offset = (i - 1) * w
            bars = ax.bar(xi + offset, vals, w,
                          label=CTRL_LABEL[ctrl], color=C[ctrl],
                          edgecolor="white", linewidth=0.4, zorder=3)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.008,
                            f"{v:.3g}", ha="center", va="bottom",
                            fontsize=5.5, color="#333", zorder=4)

        # Shade unseen region
        ax.axvspan(n_in - 0.5, n - 0.5, alpha=0.07, color="#cc3333", zorder=0)
        ax.axvline(x=n_in - 0.5, color="#cc3333", lw=1.5, ls="--", alpha=0.8)
        ax.text(n_in - 0.3, ax.get_ylim()[1], "← in-dist  |  unseen OOD →",
                color="#cc3333", fontsize=8, va="top", ha="left")

        ax.set_xticks(xi)
        ax.set_xticklabels(sc_labels, fontsize=7.5)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"{label}\nBaseline vs AI ep1000 vs AI ep2000  |  grey = in-distribution  |  red shading = unseen OOD",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")
        ax.yaxis.grid(True, ls="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

        lb = LOWER_BETTER.get(metric, True)
        ax.annotate("↓ lower is better" if lb else "↑ higher is better",
                    xy=(0.99, 0.97), xycoords="axes fraction",
                    fontsize=7, color="#888", va="top", ha="right")

        plt.tight_layout()
        safe = metric.replace("/", "_")
        plt.savefig(CHARTS_DIR / f"{safe}.png", dpi=130)
        plt.close(fig)
        print(f"  {safe}.png")


# ── Figure 2: % improvement heatmap — ep2000 vs baseline ─────────────────────

def fig_improvement_heatmap(data: dict):
    sc_keys   = [k for k, _ in ALL_SCENARIOS]
    sc_labels = [l.replace("\n", " ") for _, l in ALL_SCENARIOS]
    metrics   = [m for m in KEY_METRICS if m not in ("arrived_vehicles", "vehicles_still_running", "teleports")]
    m_labels  = [METRIC_LABEL[m].replace(" (", "\n(") for m in metrics]

    matrix = np.zeros((len(metrics), len(sc_keys)))
    for j, sc in enumerate(sc_keys):
        for i, metric in enumerate(metrics):
            bval = data.get(sc, {}).get("baseline", {}).get(metric, 0.0)
            aval = data.get(sc, {}).get("ep2000",   {}).get(metric, 0.0)
            matrix[i, j] = pct_improvement(bval, aval, LOWER_BETTER.get(metric, True))

    fig, ax = plt.subplots(figsize=(17, 6))
    vmax = min(80, np.percentile(np.abs(matrix), 95))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(sc_keys)))
    ax.set_xticklabels(sc_labels, fontsize=8, rotation=20, ha="right")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(m_labels, fontsize=8)

    # Annotate cells
    for i in range(len(metrics)):
        for j in range(len(sc_keys)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:+.0f}%", ha="center", va="center",
                    fontsize=7, color="black" if abs(v) < vmax * 0.6 else "white",
                    fontweight="bold" if abs(v) > 20 else "normal")

    # OOD divider
    n_in = len(IN_DIST)
    ax.axvline(x=n_in - 0.5, color="#333", lw=2)
    ax.text(n_in - 0.4, -0.8, "← In-dist", fontsize=8, color="#333",
            transform=ax.get_xaxis_transform(), ha="right")
    ax.text(n_in + 0.4, -0.8, "Unseen OOD →", fontsize=8, color="#cc3333",
            transform=ax.get_xaxis_transform(), ha="left")

    plt.colorbar(im, ax=ax, label="% improvement vs baseline (green=better, red=worse)", shrink=0.8)
    ax.set_title("ep2000 vs Fixed-Timing Baseline — % Improvement per Metric × Scenario\n"
                 "Right of divider = scenarios never seen during training",
                 fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "_heatmap_ep2000_vs_baseline.png", dpi=130)
    plt.close(fig)
    print("  _heatmap_ep2000_vs_baseline.png")


# ── Figure 3: In-dist vs OOD average improvement bar chart ───────────────────

def fig_indist_vs_ood(data: dict):
    metrics = ["waiting_time_s", "time_loss_s", "travel_time_s",
               "queue_length", "network_speed_ms", "throughput_veh_hr",
               "co2_per_vehicle_mg", "fuel_per_vehicle_L",
               "stops_per_vehicle", "fairness_variance"]
    m_labels = [METRIC_LABEL[m].split(" (")[0] for m in metrics]

    in_keys  = [k for k, _ in IN_DIST]
    ood_keys = [k for k, _ in UNSEEN]

    def avg_improvement(sc_keys, ctrl):
        results = []
        for metric in metrics:
            pcts = []
            for sc in sc_keys:
                bval = data.get(sc, {}).get("baseline", {}).get(metric, 0.0)
                aval = data.get(sc, {}).get(ctrl,       {}).get(metric, 0.0)
                pcts.append(pct_improvement(bval, aval, LOWER_BETTER.get(metric, True)))
            results.append(np.mean(pcts))
        return results

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    xi = np.arange(len(metrics))
    w  = 0.35

    for ax, sc_keys, title_tag in [
        (axes[0], in_keys,  "In-Distribution (training curriculum)"),
        (axes[1], ood_keys, "Unseen OOD (never trained on)"),
    ]:
        for i, ctrl in enumerate(["ep1000", "ep2000"]):
            vals = avg_improvement(sc_keys, ctrl)
            offset = (i - 0.5) * w
            bars = ax.bar(xi + offset, vals, w,
                          label=CTRL_LABEL[ctrl], color=C[ctrl],
                          edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.5 if v >= 0 else -2.0),
                        f"{v:+.1f}%", ha="center", va="bottom", fontsize=7)

        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(xi)
        ax.set_xticklabels(m_labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Mean % improvement vs baseline", fontsize=9)
        ax.set_title(f"{title_tag}\nAverage across scenarios",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, ls="--", alpha=0.4)
        ax.set_axisbelow(True)

    plt.suptitle("TraFix v6 — ep1000 vs ep2000: In-Distribution vs Unseen OOD Performance",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "_indist_vs_ood_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  _indist_vs_ood_comparison.png")


# ── Figure 4: Cars completed — all 13 scenarios ───────────────────────────────

def fig_cars_completed(data: dict):
    sc_keys   = [k for k, _ in ALL_SCENARIOS]
    sc_labels = [l for _, l in ALL_SCENARIOS]
    n = len(sc_keys)
    n_in = len(IN_DIST)

    fig, ax = plt.subplots(figsize=(16, 5))
    w  = 0.26
    xi = np.arange(n)

    for i, ctrl in enumerate(["baseline", "ep1000", "ep2000"]):
        arrived  = [data.get(sc, {}).get(ctrl, {}).get("arrived_vehicles", 0) for sc in sc_keys]
        still_in = [data.get(sc, {}).get(ctrl, {}).get("vehicles_still_running", 0) for sc in sc_keys]
        offset   = (i - 1) * w
        ax.bar(xi + offset, arrived,  w, label=f"{CTRL_LABEL[ctrl]} — completed",
               color=C[ctrl], edgecolor="white", zorder=3)
        ax.bar(xi + offset, still_in, w, bottom=arrived,
               label=f"{CTRL_LABEL[ctrl]} — still in network",
               color=C[ctrl], alpha=0.3, edgecolor="white", hatch="//", zorder=3)
        for j, (arr, sti) in enumerate(zip(arrived, still_in)):
            total = arr + sti
            if total > 0:
                ax.text(xi[j] + offset, total + total * 0.01,
                        str(int(arr)), ha="center", va="bottom", fontsize=6, color=C[ctrl],
                        fontweight="bold", zorder=4)

    ax.axvspan(n_in - 0.5, n - 0.5, alpha=0.06, color="#cc3333", zorder=0)
    ax.axvline(x=n_in - 0.5, color="#cc3333", lw=1.5, ls="--", alpha=0.8)
    ax.text(n_in - 0.3, ax.get_ylim()[1] * 0.99,
            "← in-dist  |  unseen OOD →", color="#cc3333", fontsize=8, va="top")

    ax.set_xticks(xi)
    ax.set_xticklabels(sc_labels, fontsize=7.5)
    ax.set_ylabel("Vehicles", fontsize=10)
    ax.set_title("Cars Completed Trip (solid) vs Still in Network at Sim End (hatched)\n"
                 "Numbers above bars = completed count",
                 fontsize=10, fontweight="bold")
    # Compact legend
    handles = [
        mpatches.Patch(color=C[c], label=CTRL_LABEL[c]) for c in ["baseline", "ep1000", "ep2000"]
    ]
    handles += [mpatches.Patch(facecolor="grey", alpha=0.3, hatch="//", label="Still in network")]
    ax.legend(handles=handles, fontsize=8, loc="upper left", ncol=2)
    ax.yaxis.grid(True, ls="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "_cars_completed.png", dpi=130)
    plt.close(fig)
    print("  _cars_completed.png")


# ── Figure 5: OOD deep-dive — key metrics for 6 unseen scenarios ──────────────

def fig_ood_deepdive(data: dict):
    ood_keys   = [k for k, _ in UNSEEN]
    ood_labels = [l for _, l in UNSEEN]
    focus = ["waiting_time_s", "time_loss_s", "queue_length", "network_speed_ms"]
    f_labels = [METRIC_LABEL[m] for m in focus]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, metric, mlabel in zip(axes, focus, f_labels):
        w  = 0.26
        xi = np.arange(len(ood_keys))
        for i, ctrl in enumerate(["baseline", "ep1000", "ep2000"]):
            vals = [data.get(sc, {}).get(ctrl, {}).get(metric, 0.0) for sc in ood_keys]
            bars = ax.bar(xi + (i - 1) * w, vals, w,
                          label=CTRL_LABEL[ctrl], color=C[ctrl],
                          edgecolor="white", zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f"{v:.3g}", ha="center", va="bottom", fontsize=7, zorder=4)
        ax.set_xticks(xi)
        ax.set_xticklabels(ood_labels, fontsize=8)
        ax.set_ylabel(mlabel, fontsize=9)
        ax.set_title(mlabel, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.yaxis.grid(True, ls="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        lb = LOWER_BETTER.get(metric, True)
        ax.annotate("↓ lower is better" if lb else "↑ higher is better",
                    xy=(0.99, 0.97), xycoords="axes fraction",
                    fontsize=7, color="#888", va="top", ha="right")

    plt.suptitle("OOD Generalisation Deep-Dive — 6 Unseen Scenarios\n"
                 "Baseline vs AI ep1000 vs AI ep2000",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "_ood_deepdive.png", dpi=130)
    plt.close(fig)
    print("  _ood_deepdive.png")


# ── Figure 6: Winner map — ep1000 vs ep2000 per scenario ─────────────────────

def fig_winner_map(data: dict):
    sc_keys   = [k for k, _ in ALL_SCENARIOS]
    sc_labels = [l.replace("\n", " ") for _, l in ALL_SCENARIOS]
    metrics   = [m for m in KEY_METRICS if m not in ("teleports", "total_departed")]
    m_labels  = [METRIC_LABEL[m].split(" (")[0] for m in metrics]

    # +1 = ep2000 better, -1 = ep1000 better, 0 = tie
    winner = np.zeros((len(metrics), len(sc_keys)))
    for j, sc in enumerate(sc_keys):
        for i, metric in enumerate(metrics):
            lb   = LOWER_BETTER.get(metric, True)
            v10  = data.get(sc, {}).get("ep1000", {}).get(metric, 0.0)
            v20  = data.get(sc, {}).get("ep2000", {}).get(metric, 0.0)
            if abs(v10 - v20) < 1e-6:
                winner[i, j] = 0
            elif (lb and v20 < v10) or (not lb and v20 > v10):
                winner[i, j] = 1   # ep2000 wins
            else:
                winner[i, j] = -1  # ep1000 wins

    fig, ax = plt.subplots(figsize=(17, 6))
    cmap = matplotlib.colors.ListedColormap(["#e07b39", "#dddddd", "#3a9e6e"])
    im = ax.imshow(winner, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    for i in range(len(metrics)):
        for j in range(len(sc_keys)):
            v = winner[i, j]
            txt = "ep2000" if v == 1 else ("ep1000" if v == -1 else "tie")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    color="white" if v != 0 else "#555", fontweight="bold")

    ax.set_xticks(range(len(sc_keys)))
    ax.set_xticklabels(sc_labels, fontsize=8, rotation=20, ha="right")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(m_labels, fontsize=8)

    n_in = len(IN_DIST)
    ax.axvline(x=n_in - 0.5, color="#333", lw=2)
    ax.text(n_in - 0.4, -0.8, "← In-dist", fontsize=8, color="#333",
            transform=ax.get_xaxis_transform(), ha="right")
    ax.text(n_in + 0.4, -0.8, "Unseen OOD →", fontsize=8, color="#cc3333",
            transform=ax.get_xaxis_transform(), ha="left")

    legend_handles = [
        mpatches.Patch(color="#3a9e6e", label="ep2000 better"),
        mpatches.Patch(color="#e07b39", label="ep1000 better"),
        mpatches.Patch(color="#dddddd", label="Tie"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    ax.set_title("ep1000 vs ep2000 — Which Model Wins per Metric × Scenario?",
                 fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "_winner_map.png", dpi=130)
    plt.close(fig)
    print("  _winner_map.png")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading data from {MANIFEST_PATH}")
    data = load_data(sim_s=3600)
    print(f"Loaded {len(data)} scenarios, generating figures to {CHARTS_DIR}\n")

    print("Per-metric bar charts (13 scenarios each):")
    fig_per_metric(data)

    print("\nSummary figures:")
    fig_improvement_heatmap(data)
    fig_indist_vs_ood(data)
    fig_cars_completed(data)
    fig_ood_deepdive(data)
    fig_winner_map(data)

    charts = sorted(CHARTS_DIR.glob("*.png"))
    print(f"\nDone — {len(charts)} figures written to {CHARTS_DIR}")
