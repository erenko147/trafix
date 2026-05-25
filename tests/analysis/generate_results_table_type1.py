"""
Test Type 1 Results Table Generator
=====================================
Reads tests/reports/results.csv and renders a publication-quality matplotlib
table PNG comparing AI v6 vs Fixed Baseline across all scenarios.

Output: tests/reports/charts/results_table_type1.png

Run: python tests/analysis/generate_results_table_type1.py
"""
import sys, pathlib, csv, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_CSV = pathlib.Path(__file__).resolve().parents[1] / "reports" / "results.csv"
OUT_PNG     = RESULTS_CSV.parent / "charts" / "results_table_type1.png"

# Scenarios in display order
SCENARIOS = [
    ("type1", "low",          "Type 1\nLow"),
    ("type1", "medium",       "Type 1\nMedium"),
    ("type1", "high",         "Type 1\nHigh"),
    ("type2", "morning_peak", "Type 2\nMorn. Peak"),
    ("type2", "evening_peak", "Type 2\nEve. Peak"),
    ("type2", "incident",     "Type 2\nIncident"),
    ("type2", "pulse",        "Type 2\nPulse"),
]

# (key, display_name, unit, lower_is_better)
METRICS = [
    ("waiting_time_s",       "Mean Waiting Time",   "s",      True),
    ("travel_time_s",        "Mean Travel Time",    "s",      True),
    ("time_loss_s",          "Mean Time Loss",      "s",      True),
    ("queue_length_vehicles","Queue Length",         "veh",    True),
    ("network_speed_ms",     "Network Speed",       "m/s",    False),
    ("throughput_veh_hr",    "Throughput",          "veh/hr", False),
    ("co2_mg_per_vehicle",   "CO₂ per Vehicle",    "mg",     True),
    ("fuel_per_vehicle_L",   "Fuel per Vehicle",    "L",      True),
    ("stops_per_vehicle",    "Stops per Vehicle",   "stops",  True),
    ("fairness_variance",    "Fairness Variance",   "σ²",     True),
]

HEADER_BG   = "#2c2c2c"
HEADER_FG   = "white"
SUBHDR_BG   = "#1a5276"
ROW_ALT     = "#f7f7f7"
ROW_NORM    = "white"
AI_COLOR    = "#1a7a3a"
BASE_COLOR  = "#8b0000"
IMPR_COLOR  = "#c6efce"
REGR_COLOR  = "#ffc7ce"
NEUT_COLOR  = "#ffffff"


def load_data():
    data = {}
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            key = (row["scenario"], row["traffic_level"], row["controller"], row["metric"])
            data[key] = float(row["value"])
    return data


def fmt(v, metric_key):
    if metric_key == "co2_mg_per_vehicle":
        return f"{v/1000:.1f}k"
    if metric_key == "fairness_variance":
        return f"{v:.1f}" if v < 1000 else f"{v/1000:.1f}k"
    if abs(v) >= 1000:
        return f"{v:.0f}"
    if abs(v) >= 100:
        return f"{v:.1f}"
    return f"{v:.2f}"


def delta_pct(base, ai, lower_is_better):
    if base == 0:
        return 0.0
    raw = (base - ai) / abs(base) * 100 if lower_is_better else (ai - base) / abs(base) * 100
    return raw


def make_table():
    data = load_data()

    n_metrics   = len(METRICS)
    n_scenarios = len(SCENARIOS)

    # Columns: Metric | Unit | [per scenario: Baseline | AI | Δ%]
    # That's 2 + 3*7 = 23 columns — too wide. Use 2 cols per scenario: AI value + Δ%
    # Columns: Metric | Unit | [s0: AI | Δ] | [s1: AI | Δ] ... = 2 + 2*7 = 16

    scenario_col_w = 0.108   # width per scenario group (2 sub-cols each)
    metric_col_w   = 0.17
    unit_col_w     = 0.045
    total_w = metric_col_w + unit_col_w + scenario_col_w * n_scenarios

    sub_col_w = scenario_col_w / 2   # AI value | Δ%

    col_x_metric = 0.02
    col_x_unit   = col_x_metric + metric_col_w
    col_x_scen   = [col_x_unit + unit_col_w + i * scenario_col_w for i in range(n_scenarios)]

    row_h   = 0.055
    fig_h   = 2.2 + n_metrics * 0.62 + 0.6
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def _rect(x, y, w, h, fc, ec="#cccccc", lw=0.4, **kw):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y - h), w, h,
            boxstyle="square,pad=0",
            facecolor=fc, edgecolor=ec, linewidth=lw,
            transform=fig.transFigure, figure=fig, **kw
        ))

    def _text(x, y, s, ha="center", color="black", size=7.5, bold=False):
        fig.text(x, y, s, ha=ha, va="center", fontsize=size,
                 color=color, fontweight="bold" if bold else "normal",
                 multialignment="center")

    # Title
    fig.text(0.5, 0.97,
             "TraFix v6 — Test Type 1 & 2: AI Controller vs Fixed Baseline",
             ha="center", va="top", fontsize=13, fontweight="bold", color="#1a1a1a")
    fig.text(0.5, 0.935,
             "Values shown: AI v6 result  |  Δ% = improvement over baseline  "
             "(green = AI better, red = AI worse)",
             ha="center", va="top", fontsize=8.5, color="#555555", fontstyle="italic")

    table_t = 0.89

    # ── Scenario group headers ────────────────────────────────────────────────
    _rect(col_x_metric, table_t, total_w, row_h, fc=HEADER_BG, ec=HEADER_BG)
    _text(col_x_metric + metric_col_w / 2, table_t - row_h / 2,
          "Metric", color="white", bold=True, size=8.5)
    _text(col_x_unit + unit_col_w / 2, table_t - row_h / 2,
          "Unit", color="white", bold=True, size=8.5)

    for si, (scenario, level, label) in enumerate(SCENARIOS):
        sx = col_x_scen[si]
        sc = "#1a5276" if scenario == "type1" else "#4a235a"
        _rect(sx, table_t, scenario_col_w, row_h, fc=sc, ec=sc)
        _text(sx + scenario_col_w / 2, table_t - row_h / 2,
              label, color="white", bold=True, size=7.5)

    # Sub-header: AI | Δ%
    sub_y = table_t - row_h
    _rect(col_x_metric, sub_y, total_w, row_h * 0.65, fc="#444444", ec="#444444")
    _text(col_x_metric + metric_col_w / 2, sub_y - row_h * 0.65 / 2,
          "", color="white", size=7)
    for si in range(n_scenarios):
        sx = col_x_scen[si]
        _text(sx + sub_col_w / 2, sub_y - row_h * 0.65 / 2,
              "AI v6", color="white", bold=True, size=7)
        _text(sx + sub_col_w + sub_col_w / 2, sub_y - row_h * 0.65 / 2,
              "Δ%", color="#aaffaa", bold=True, size=7)

    data_y0 = sub_y - row_h * 0.65

    # Win/loss counters
    wins = losses = ties = 0

    # ── Data rows ─────────────────────────────────────────────────────────────
    for r_idx, (metric_key, metric_name, unit, lower_is_better) in enumerate(METRICS):
        y = data_y0 - r_idx * row_h
        bg = ROW_ALT if r_idx % 2 else ROW_NORM
        _rect(col_x_metric, y, total_w, row_h, fc=bg)

        _text(col_x_metric + 0.006, y - row_h / 2, metric_name, ha="left", size=8)
        _text(col_x_unit + unit_col_w / 2, y - row_h / 2, unit, size=7, color="#666666")

        for si, (scenario, level, _) in enumerate(SCENARIOS):
            sx = col_x_scen[si]
            ai_v   = data.get((scenario, level, "ai",       metric_key))
            base_v = data.get((scenario, level, "baseline", metric_key))

            if ai_v is None or base_v is None:
                _text(sx + scenario_col_w / 2, y - row_h / 2, "N/A", size=7, color="#aaaaaa")
                continue

            dp = delta_pct(base_v, ai_v, lower_is_better)

            # Background colour for AI value cell
            if abs(dp) < 0.5:
                cell_bg = bg
                wins += 0; ties += 1
            elif dp > 0:
                cell_bg = IMPR_COLOR
                wins += 1
            else:
                cell_bg = REGR_COLOR
                losses += 1

            # AI value cell
            ai_cx = sx + 0.002
            ai_cw = sub_col_w - 0.003
            _rect(ai_cx, y - 0.004, ai_cw, row_h - 0.008, fc=cell_bg, ec="#dddddd", lw=0.3)
            _text(ai_cx + ai_cw / 2, y - row_h / 2, fmt(ai_v, metric_key),
                  size=7.5, bold=(dp > 0 and abs(dp) > 1))

            # Δ% cell
            d_cx = sx + sub_col_w + 0.002
            d_cw = sub_col_w - 0.003
            d_color = "#1a7a3a" if dp > 0.5 else ("#cc2222" if dp < -0.5 else "#888888")
            d_sign  = "+" if dp > 0 else ""
            _rect(d_cx, y - 0.004, d_cw, row_h - 0.008, fc=bg, ec="#eeeeee", lw=0.3)
            _text(d_cx + d_cw / 2, y - row_h / 2,
                  f"{d_sign}{dp:.1f}%", size=7, color=d_color,
                  bold=(abs(dp) > 10))

    # ── Summary footer ────────────────────────────────────────────────────────
    footer_y = data_y0 - n_metrics * row_h - row_h * 0.3
    _rect(col_x_metric, footer_y, total_w, row_h, fc="#e8e8e8", ec="#aaaaaa", lw=0.8)
    _text(col_x_metric + 0.006, footer_y - row_h / 2,
          f"AI v6 wins: {wins}   Regressions: {losses}   Ties: {ties}   "
          f"Net improvement: {wins - losses:+d} / {wins + losses + ties} metric-scenarios",
          ha="left", size=8.5, bold=True, color="#1a1a1a")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_y = footer_y - row_h - 0.015
    patches = [
        ("#1a5276", "Type 1 scenarios (standard traffic)"),
        ("#4a235a", "Type 2 scenarios (complex patterns)"),
        (IMPR_COLOR, "AI improves over baseline"),
        (REGR_COLOR, "AI regresses vs baseline"),
    ]
    lx = col_x_metric
    for col, label in patches:
        _rect(lx, legend_y, 0.012, 0.018, fc=col, ec="#999999")
        _text(lx + 0.016, legend_y - 0.009, label, ha="left", size=7.5, color="#444444")
        lx += len(label) * 0.007 + 0.025

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {OUT_PNG}")
    return OUT_PNG, wins, losses, ties


if __name__ == "__main__":
    print("Generating Test Type 1 results table …")
    out, w, l, t = make_table()
    print(f"  Wins: {w}  Regressions: {l}  Ties: {t}")
    print(f"  Done → {out}")
