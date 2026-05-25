"""
Test Type 2 Results Table Generator
=====================================
Reads tests/reports/test_type_2/results.csv and renders a publication-quality
matplotlib table PNG comparing Roundabout vs Standard Intersection configurations.

Output: tests/reports/test_type_2/charts/results_table_type2.png

Run: python tests/analysis/generate_results_table_type2.py
"""
import sys, pathlib, csv, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Configuration ─────────────────────────────────────────────────────────────

RESULTS_CSV = pathlib.Path(__file__).resolve().parents[1] / "reports" / "test_type_2" / "results.csv"
OUT_PNG     = RESULTS_CSV.parent / "charts" / "results_table_type2.png"

CONTROLLERS = [
    ("roundabout_fixed", "Roundabout\n(Fixed)"),
    ("standard_fixed",   "Standard\n(Fixed)"),
    ("standard_ep1000",  "Standard AI\n(ep1000)"),
    ("standard_ep2000",  "Standard AI\n(ep2000)"),
]
CTRL_IDS = [c[0] for c in CONTROLLERS]
CTRL_LABELS = [c[1] for c in CONTROLLERS]

TRAFFIC_LEVELS = ["low", "medium", "high"]
TRAFFIC_LABELS = {"low": "Low", "medium": "Medium", "high": "High"}

# (metric_key, display_name, unit, lower_is_better)
METRICS = [
    ("waiting_time_s",    "Mean Waiting Time",    "s",         True),
    ("travel_time_s",     "Mean Travel Time",     "s",         True),
    ("time_loss_s",       "Mean Time Loss",       "s",         True),
    ("queue_length",      "Queue Length",         "veh",       True),
    ("network_speed_ms",  "Network Speed",        "m/s",       False),
    ("throughput_veh_hr", "Throughput",           "veh/hr",    False),
    ("co2_per_vehicle_mg","CO₂ per Vehicle",      "mg",        True),
    ("fuel_per_vehicle_L","Fuel per Vehicle",     "L",         True),
    ("stops_per_vehicle", "Stops per Vehicle",    "stops",     True),
]

# Visual constants
HEADER_BG   = "#2c2c2c"
HEADER_FG   = "white"
SUBHDR_BG   = "#444444"
ROW_ALT     = "#f7f7f7"
ROW_NORM    = "white"
BEST_COLOR  = "#c6efce"   # light green
WORST_COLOR = "#ffc7ce"   # light red
MID_COLOR   = "#ffeb9c"   # light yellow
CTRL_COLORS = {
    "roundabout_fixed": "#6c757d",
    "standard_fixed":   "#0d6efd",
    "standard_ep1000":  "#fd7e14",
    "standard_ep2000":  "#198754",
}


# ── Load data ─────────────────────────────────────────────────────────────────

def load_data():
    data = {}
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            key = (row["controller"], row["traffic_level"], row["metric"])
            data[key] = float(row["value"])
    return data


def avg_across_traffic(data, controller, metric):
    vals = [data.get((controller, t, metric)) for t in TRAFFIC_LEVELS]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def get_cell_value(data, controller, traffic, metric):
    return data.get((controller, traffic, metric))


# ── Figure ────────────────────────────────────────────────────────────────────

def make_table():
    data = load_data()

    # Layout: metric name | [low: 4 ctrl] | [medium: 4 ctrl] | [high: 4 ctrl] | avg winner
    # Columns: 1 + 4 + 4 + 4 + 1 = 14, but we'll use grouped layout
    # Simpler: metric | avg per ctrl (4) | winner
    # We'll show averages across traffic levels with colored cells

    n_metrics = len(METRICS)
    n_ctrl    = len(CONTROLLERS)

    # Column layout: metric_name | ctrl0 | ctrl1 | ctrl2 | ctrl3 | winner
    col_labels = ["Metric", "Unit"] + CTRL_LABELS + ["Winner"]
    col_widths = [0.23, 0.06] + [0.14] * n_ctrl + [0.13]
    col_x = [0.02]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w)

    total_w = sum(col_widths)
    row_h   = 0.062
    fig_h   = 2.0 + n_metrics * 0.70 + 1.0
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    table_t = 0.89   # top of table in normalised figure coords

    def _rect(x, y, w, h, fc, ec="#cccccc", lw=0.5, **kw):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y - h), w, h,
            boxstyle="square,pad=0",
            facecolor=fc, edgecolor=ec, linewidth=lw,
            transform=fig.transFigure, figure=fig, **kw
        ))

    def _text(x, y, s, ha="center", color="black", size=8.5, bold=False, wrap=False):
        fig.text(x, y, s, ha=ha, va="center", fontsize=size,
                 color=color, fontweight="bold" if bold else "normal",
                 multialignment="center")

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.97,
             "TraFix v6 — Test Type 2: Roundabout vs Standard Intersection",
             ha="center", va="top", fontsize=14, fontweight="bold", color="#1a1a1a")
    fig.text(0.5, 0.93,
             "Average metrics across Low / Medium / High traffic levels  "
             "(best per row = green, worst = red)",
             ha="center", va="top", fontsize=9, color="#555555", fontstyle="italic")

    header_y = table_t

    # ── Header row ────────────────────────────────────────────────────────────
    _rect(col_x[0], header_y, total_w, row_h, fc=HEADER_BG, ec=HEADER_BG)
    for i, (label, cx, cw) in enumerate(zip(col_labels, col_x, col_widths)):
        _text(cx + cw / 2, header_y - row_h / 2, label,
              color=HEADER_FG, bold=True, size=8.5)

    # Controller color pills in sub-header
    subhdr_y = header_y - row_h
    _rect(col_x[0], subhdr_y, total_w, row_h * 0.55, fc=SUBHDR_BG, ec=SUBHDR_BG)
    for ci, ctrl_id in enumerate(CTRL_IDS):
        col_i = 2 + ci
        pill_x = col_x[col_i] + 0.005
        pill_w = col_widths[col_i] - 0.010
        _rect(pill_x, subhdr_y - 0.004, pill_w, row_h * 0.55 - 0.008,
              fc=CTRL_COLORS[ctrl_id], ec=CTRL_COLORS[ctrl_id])

    data_start_y = subhdr_y - row_h * 0.55

    # ── Win counters ─────────────────────────────────────────────────────────
    win_count = {cid: 0 for cid in CTRL_IDS}

    # ── Data rows ────────────────────────────────────────────────────────────
    for r_idx, (metric_key, metric_name, unit, lower_is_better) in enumerate(METRICS):
        y = data_start_y - r_idx * row_h
        bg = ROW_ALT if r_idx % 2 else ROW_NORM
        _rect(col_x[0], y, total_w, row_h, fc=bg)

        # Metric name
        _text(col_x[0] + 0.008, y - row_h / 2, metric_name, ha="left", size=8.5)
        _text(col_x[1] + col_widths[1] / 2, y - row_h / 2, unit, size=8, color="#666666")

        # Compute averages for each controller
        avgs = {}
        for ctrl_id in CTRL_IDS:
            v = avg_across_traffic(data, ctrl_id, metric_key)
            avgs[ctrl_id] = v

        valid_vals = {k: v for k, v in avgs.items() if v is not None}
        if valid_vals:
            best_ctrl  = min(valid_vals, key=lambda k: valid_vals[k]) if lower_is_better \
                         else max(valid_vals, key=lambda k: valid_vals[k])
            worst_ctrl = max(valid_vals, key=lambda k: valid_vals[k]) if lower_is_better \
                         else min(valid_vals, key=lambda k: valid_vals[k])
            win_count[best_ctrl] += 1
        else:
            best_ctrl = worst_ctrl = None

        # Draw each controller's average cell
        for ci, ctrl_id in enumerate(CTRL_IDS):
            col_i = 2 + ci
            v = avgs.get(ctrl_id)
            if v is None:
                cell_txt = "N/A"
                cell_bg  = bg
            else:
                # Format value
                if metric_key in ("throughput_veh_hr",):
                    cell_txt = f"{v:.0f}"
                elif metric_key in ("co2_per_vehicle_mg",):
                    cell_txt = f"{v/1000:.1f}k"
                elif abs(v) >= 100:
                    cell_txt = f"{v:.1f}"
                else:
                    cell_txt = f"{v:.2f}"

                if ctrl_id == best_ctrl:
                    cell_bg = BEST_COLOR
                elif ctrl_id == worst_ctrl:
                    cell_bg = WORST_COLOR
                else:
                    cell_bg = bg

            cx = col_x[col_i] + 0.003
            cw = col_widths[col_i] - 0.006
            _rect(cx, y - 0.005, cw, row_h - 0.010, fc=cell_bg, ec="#dddddd", lw=0.4)
            bold = (ctrl_id == best_ctrl)
            _text(cx + cw / 2, y - row_h / 2, cell_txt, bold=bold, size=8.5)

        # Winner column
        if best_ctrl:
            wx = col_x[-1] + 0.004
            ww = col_widths[-1] - 0.008
            winner_short = {
                "roundabout_fixed": "Roundabout",
                "standard_fixed":   "Std Fixed",
                "standard_ep1000":  "AI ep1000",
                "standard_ep2000":  "AI ep2000",
            }[best_ctrl]
            _rect(wx, y - 0.005, ww, row_h - 0.010,
                  fc=CTRL_COLORS[best_ctrl], ec=CTRL_COLORS[best_ctrl])
            _text(wx + ww / 2, y - row_h / 2, winner_short,
                  color="white", bold=True, size=8)

    # ── Win-count summary row ─────────────────────────────────────────────────
    summary_y = data_start_y - n_metrics * row_h - row_h * 0.3
    _rect(col_x[0], summary_y, total_w, row_h, fc="#e8e8e8", ec="#aaaaaa", lw=1.0)
    _text(col_x[0] + 0.008, summary_y - row_h / 2, "Metric Wins (best avg)",
          ha="left", bold=True, size=8.5)
    _text(col_x[1] + col_widths[1] / 2, summary_y - row_h / 2, "", size=8)
    for ci, ctrl_id in enumerate(CTRL_IDS):
        col_i = 2 + ci
        wins = win_count[ctrl_id]
        cx = col_x[col_i] + 0.003
        cw = col_widths[col_i] - 0.006
        badge_fc = CTRL_COLORS[ctrl_id] if wins > 0 else "#cccccc"
        _rect(cx, summary_y - 0.005, cw, row_h - 0.010, fc=badge_fc, ec=badge_fc)
        _text(cx + cw / 2, summary_y - row_h / 2,
              f"{wins} win{'s' if wins != 1 else ''}",
              color="white", bold=(wins > 0), size=9)
    # blank winner cell
    wx = col_x[-1] + 0.004
    ww = col_widths[-1] - 0.008
    _rect(wx, summary_y - 0.005, ww, row_h - 0.010, fc="#e8e8e8", ec="#aaaaaa")

    # ── Traffic-level breakdown panel (small) ─────────────────────────────────
    panel_y = summary_y - row_h - 0.05
    panel_h = len(TRAFFIC_LEVELS) * row_h * 0.85 + 0.06
    _rect(col_x[0], panel_y, total_w, panel_h, fc="#f0f7ff", ec="#3a7ebf", lw=1.2)

    fig.text(col_x[0] + 0.01, panel_y - 0.018,
             "Mean Waiting Time by Traffic Level (s)",
             va="top", fontsize=9, fontweight="bold", color="#3a7ebf")

    sub_row_h = row_h * 0.75
    for ti, traffic in enumerate(TRAFFIC_LEVELS):
        ry = panel_y - 0.048 - ti * sub_row_h
        _text(col_x[0] + 0.008, ry - sub_row_h / 2,
              TRAFFIC_LABELS[traffic], ha="left", size=8.5, bold=True)
        _text(col_x[1] + col_widths[1] / 2, ry - sub_row_h / 2, "", size=8)

        vals = {cid: data.get((cid, traffic, "waiting_time_s")) for cid in CTRL_IDS}
        valid = {k: v for k, v in vals.items() if v is not None}
        best_v  = min(valid.values()) if valid else None
        worst_v = max(valid.values()) if valid else None

        for ci, ctrl_id in enumerate(CTRL_IDS):
            col_i = 2 + ci
            v = vals.get(ctrl_id)
            cx = col_x[col_i] + 0.003
            cw = col_widths[col_i] - 0.006
            if v is None:
                _text(cx + cw / 2, ry - sub_row_h / 2, "N/A", size=8)
            else:
                if v == best_v:
                    cell_fc = BEST_COLOR
                elif v == worst_v:
                    cell_fc = WORST_COLOR
                else:
                    cell_fc = "#f0f7ff"
                _rect(cx, ry - 0.004, cw, sub_row_h - 0.008, fc=cell_fc, ec="#dde8f5", lw=0.4)
                _text(cx + cw / 2, ry - sub_row_h / 2, f"{v:.1f}", size=8.5,
                      bold=(v == best_v))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_y = panel_y - panel_h - 0.025
    fig.text(col_x[0], legend_y, "Configuration key:", va="top", fontsize=8, color="#666666")
    lx = col_x[0] + 0.14
    for ctrl_id, label in CONTROLLERS:
        short = label.replace("\n", " ")
        fig.text(lx, legend_y, f"■ {short}", va="top", fontsize=8,
                 color=CTRL_COLORS[ctrl_id], fontweight="bold")
        lx += 0.20

    fig.text(col_x[0] + total_w, legend_y,
             "■ Best   ■ Worst",
             va="top", fontsize=8, ha="right", color="#666666")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {OUT_PNG}")
    return OUT_PNG


if __name__ == "__main__":
    print("Generating Test Type 2 results table …")
    out = make_table()
    print(f"  Done → {out}")
