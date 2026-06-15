"""
Big Showcase — AI (TraFix v6) vs Fixed-Timing Lights, across EVERY test scenario.
==================================================================================
Reads the master results CSV (tests/reports/results.csv), decides who won each
metric in each scenario, and renders ONE big figure that answers the headline
question: "Across all our tests, who won — our model or the fixed lights?"

The figure contains:
  • A winner heatmap (scenario x metric): green = AI won, red = fixed won,
    grey = tie. Each cell is annotated with the % improvement of AI over fixed.
  • A bottom bar showing, per scenario, how many metrics the AI won.
  • A grand tally in the title.

Mandatory unit/FR/NFR tests are NOT included — this only covers the
performance scenarios in tests/reports/results.csv.

Run:
    .venv/bin/python tests/analysis/big_showcase.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

_TESTS_DIR = Path(__file__).resolve().parents[1]
_CSV       = _TESTS_DIR / "reports" / "results.csv"
_OUT       = _TESTS_DIR / "reports" / "charts" / "_big_showcase.png"

# lower-is-better flag per metric (kept identical to analysis/compare.py)
LOWER_IS_BETTER = {
    "travel_time_s": True, "co2_mg_per_vehicle": True, "co2_total_mg": True,
    "waiting_time_s": True, "queue_length_vehicles": True,
    "throughput_veh_hr": False, "network_speed_ms": False,
    "cars_not_completed": True, "completion_rate_pct": False, "vehicles_still_running": True,
    "teleports": True, "time_loss_s": True, "fuel_per_vehicle_L": True,
    "NOx_total_mg": True, "PMx_total_mg": True, "HC_total_mg": True, "CO_total_mg": True,
    "stops_per_vehicle": True, "fairness_variance": True,
}

# Human-friendly metric labels, ordered the way they should appear (top = most
# headline-worthy, bottom = secondary emission breakdowns).
METRIC_ORDER = [
    ("travel_time_s",          "Travel time"),
    ("waiting_time_s",         "Waiting time"),
    ("time_loss_s",            "Time loss"),
    ("queue_length_vehicles",  "Queue length"),
    ("network_speed_ms",       "Network speed"),
    ("throughput_veh_hr",      "Throughput"),
    ("stops_per_vehicle",      "Stops / vehicle"),
    ("completion_rate_pct",    "Completion rate"),
    ("cars_not_completed",     "Cars not completed"),
    ("vehicles_still_running", "Cars still running"),
    ("fairness_variance",      "Fairness variance"),
    ("co2_mg_per_vehicle",     "CO₂ / vehicle"),
    ("co2_total_mg",           "CO₂ total"),
    ("fuel_per_vehicle_L",     "Fuel / vehicle"),
    ("NOx_total_mg",           "NOx total"),
    ("CO_total_mg",            "CO total"),
    ("HC_total_mg",            "HC total"),
    ("PMx_total_mg",           "PMx total"),
    ("teleports",              "Teleports"),
]

# Scenario display order + friendly labels
SCEN_ORDER = [
    (("type1", "low"),            "Low\n(200/hr)"),
    (("type1", "medium"),         "Medium\n(500/hr)"),
    (("type1", "high"),           "High\n(950/hr)"),
    (("type2", "morning_peak"),   "Morning\npeak"),
    (("type2", "evening_peak"),   "Evening\npeak"),
    (("type2", "incident"),       "Incident"),
    (("type2", "pulse"),          "Pulse"),
]

TIE_EPS = 0.05   # |% improvement| below this counts as a tie


def load_csv(path: Path):
    """Returns {(scenario, level): {controller: {metric: value}}}."""
    data: dict = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key  = (row["scenario"], row["traffic_level"])
            ctrl = row["controller"]
            data.setdefault(key, {}).setdefault(ctrl, {})[row["metric"]] = float(row["value"])
    return data


def pct_improvement(baseline: float, ai: float, lower_better: bool) -> float:
    """Positive => AI better than fixed lights."""
    if baseline == 0:
        # No baseline magnitude to divide by: judge by raw direction.
        if abs(ai) < 1e-9:
            return 0.0
        better = (ai < baseline) if lower_better else (ai > baseline)
        return 5.0 if better else -5.0
    raw = (baseline - ai) / abs(baseline) * 100.0
    return raw if lower_better else -raw


def main():
    if not _CSV.exists():
        raise SystemExit(f"results CSV not found: {_CSV}")

    data = load_csv(_CSV)

    scen_keys   = [k for k, _ in SCEN_ORDER if k in data]
    scen_labels = [lbl for k, lbl in SCEN_ORDER if k in data]
    metrics     = [(m, lbl) for m, lbl in METRIC_ORDER]
    n_m, n_s    = len(metrics), len(scen_keys)

    pct    = np.full((n_m, n_s), np.nan)   # signed % improvement
    winner = np.zeros((n_m, n_s))          # +1 AI, -1 fixed, 0 tie/missing

    ai_wins = fixed_wins = ties = 0
    for j, sk in enumerate(scen_keys):
        ctrls = data[sk]
        base  = ctrls.get("baseline", {})
        ai    = ctrls.get("ai", {})
        for i, (metric, _) in enumerate(metrics):
            if metric not in base or metric not in ai:
                continue
            p = pct_improvement(base[metric], ai[metric], LOWER_IS_BETTER.get(metric, True))
            pct[i, j] = p
            if p > TIE_EPS:
                winner[i, j] = 1;  ai_wins += 1
            elif p < -TIE_EPS:
                winner[i, j] = -1; fixed_wins += 1
            else:
                winner[i, j] = 0;  ties += 1

    total = ai_wins + fixed_wins + ties

    # ── Figure layout: heatmap on top, per-scenario win bar below ──────────────
    fig = plt.figure(figsize=(13, 13))
    gs  = GridSpec(2, 1, height_ratios=[10, 1.6], hspace=0.18, figure=fig)
    ax  = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1])

    cmap = mcolors.ListedColormap(["#c0392b", "#dddddd", "#27ae60"])  # fixed / tie / AI
    norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    ax.imshow(winner, cmap=cmap, norm=norm, aspect="auto")

    # annotate each cell with the % improvement
    for i in range(n_m):
        for j in range(n_s):
            if np.isnan(pct[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="#888")
                continue
            v = pct[i, j]
            ax.text(j, i, f"{v:+.0f}%", ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color="white" if winner[i, j] != 0 else "#555")

    ax.set_xticks(range(n_s))
    ax.set_xticklabels(scen_labels, fontsize=9)
    ax.set_yticks(range(n_m))
    ax.set_yticklabels([lbl for _, lbl in metrics], fontsize=9)
    ax.set_xticks(np.arange(-0.5, n_s, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_m, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    # vertical divider between type1 (synthetic levels) and type2 (named profiles)
    n_t1 = sum(1 for (s, _l), _ in [(k, lbl) for k, lbl in SCEN_ORDER] if s == "type1" and (s, _l) in data)
    if 0 < n_t1 < n_s:
        ax.axvline(n_t1 - 0.5, color="#222", lw=2)

    pct_ai = 100.0 * ai_wins / total if total else 0.0
    verdict = ("OUR MODEL WINS" if ai_wins > fixed_wins
               else "FIXED LIGHTS WIN" if fixed_wins > ai_wins else "TIED")
    ax.set_title(
        "TraFix v6 (AI)  vs  Fixed-Timing Lights — Who Won Every Test?\n"
        f"AI won {ai_wins} / {total} metric-scenarios ({pct_ai:.0f}%)   "
        f"•   Fixed won {fixed_wins}   •   Ties {ties}      →  {verdict}\n"
        "cell = AI's % improvement over fixed lights  (green = AI better, red = fixed better)",
        fontsize=12, fontweight="bold", pad=14,
    )

    legend_handles = [
        mpatches.Patch(color="#27ae60", label="AI (our model) won"),
        mpatches.Patch(color="#c0392b", label="Fixed lights won"),
        mpatches.Patch(color="#dddddd", label="Tie"),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.005, 1.0), fontsize=9, frameon=False)

    # ── bottom bar: AI wins per scenario ──────────────────────────────────────
    per_scen_ai = (winner == 1).sum(axis=0)
    judged      = (~np.isnan(pct)).sum(axis=0)
    bars = axb.bar(range(n_s), per_scen_ai, color="#27ae60",
                   edgecolor="white", zorder=3)
    for j, b in enumerate(bars):
        axb.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                 f"{per_scen_ai[j]}/{judged[j]}", ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color="#1a5e3a")
    axb.set_xticks(range(n_s))
    axb.set_xticklabels([l.replace("\n", " ") for l in scen_labels], fontsize=8)
    axb.set_ylim(0, n_m)
    axb.set_ylabel("Metrics\nAI won", fontsize=9)
    axb.set_title("AI metric-wins per scenario", fontsize=10, fontweight="bold")
    axb.yaxis.grid(True, ls="--", alpha=0.4, zorder=0)
    axb.set_axisbelow(True)

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"AI won {ai_wins}/{total} ({pct_ai:.0f}%), fixed won {fixed_wins}, ties {ties}")
    print(f"Verdict: {verdict}")
    print(f"Figure written: {_OUT}")


if __name__ == "__main__":
    main()
