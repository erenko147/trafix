"""
Test Report Table Generator
============================
Runs all mandatory tests, captures pass/fail counts, measures actual
inference latency, and renders a publication-quality matplotlib table PNG.

Output: tests/mandatory/test_report_table.png

Run: python tests/mandatory/generate_test_report_table.py
"""
import sys, pathlib, time, statistics, unittest, io
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

# ── Latency measurement (runs before importing heavy test modules) ─────────────

def _measure_latency(n_calls=50, warmup=5):
    from trafix_v6.trafix_v6 import TraFixV6
    from trafix_v6.rule_governor import RuleGovernor
    from backend.ai.trafix_v2 import parse_sumo_observations

    J, T, D, P = 5, 30, 20, 6
    model = TraFixV6(obs_dim=D, num_phases=P)
    model.eval()
    gov = RuleGovernor(num_junctions=J, num_phases=P)

    def _obs_list():
        return [{
            "intersection_id": i, "north_left": 3, "north_through": 10,
            "north_right": 2, "south_left": 1, "south_through": 8,
            "south_right": 0, "east_left": 4, "east_through": 12,
            "east_right": 3, "west_left": 0, "west_through": 5,
            "west_right": 1, "queue_length": 40.0,
            "current_phase": i % 6, "phase_duration": 15.0,
        } for i in range(J)]

    def _one_call():
        obs = parse_sumo_observations(_obs_list())
        window = torch.stack([obs] * T).unsqueeze(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, _ = model(window)
            logits = gov.apply(logits, window[0, -1])
        return (time.perf_counter() - t0) * 1000

    for _ in range(warmup):
        _one_call()
    samples = sorted(_one_call() for _ in range(n_calls))
    return {
        "mean":  round(statistics.mean(samples), 2),
        "p50":   round(samples[n_calls // 2], 2),
        "p99":   round(samples[int(0.99 * n_calls)], 2),
        "limit": 1000,
    }


# ── Test runner ───────────────────────────────────────────────────────────────

_HERE = pathlib.Path(__file__).resolve().parent

_TEST_FILES = [
    ("test_unit_model.py",          "Unit",        "AI Model (GRU / GATConv / PPO)"),
    ("test_unit_rule_governor.py",  "Unit",        "Rule Governor"),
    ("test_unit_observation.py",    "Unit",        "Observation Parser & Normalisation"),
    ("test_unit_preemption.py",     "Unit",        "Emergency Preemption State Machine"),
    ("test_nfr.py",                 "NFR",         "NFR-01 Latency · NFR-02 Fallback · NFR-05 Platform"),
    ("test_fr.py",                  "FR",          "FR-01 Telemetry · FR-02 AI · FR-03 Preemption · FR-04 DB · FR-06 Yellow"),
    ("test_metrics_modules.py",     "Metrics",     "12 Simulation Metric Modules"),
]


def _run_file(path: pathlib.Path):
    loader = unittest.TestLoader()
    suite  = loader.discover(str(path.parent), pattern=path.name)
    buf    = io.StringIO()
    runner = unittest.TextTestRunner(stream=buf, verbosity=0)
    result = runner.run(suite)
    return result.testsRun, len(result.failures) + len(result.errors)


def collect_results():
    rows = []
    for filename, category, description in _TEST_FILES:
        path = _HERE / filename
        total, failed = _run_file(path)
        passed = total - failed
        rows.append((category, description, total, passed, failed))
    return rows


# ── Figure ────────────────────────────────────────────────────────────────────

CATEGORY_COLORS = {
    "Unit":    "#3a7ebf",
    "NFR":     "#e07b39",
    "FR":      "#3a9e6e",
    "Metrics": "#8c4e1e",
}

PASS_COLOR = "#d4edda"
FAIL_COLOR = "#f8d7da"
HEADER_BG  = "#2c2c2c"
HEADER_FG  = "white"
ROW_ALT    = "#f5f5f5"
ROW_NORM   = "white"


def make_table(rows, latency):
    n_data = len(rows)

    fig_h = 1.6 + n_data * 0.52 + 1.4   # header + rows + latency panel
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.set_position([0, 0, 1, 1])        # axes fills whole figure
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.98, "TraFix v6 — Mandatory Test Suite Results",
             ha="center", va="top", fontsize=15, fontweight="bold", color="#1a1a1a")
    fig.text(0.5, 0.94, "tests/mandatory/  |  run: python -m pytest tests/mandatory/ -v",
             ha="center", va="top", fontsize=9, color="#555555", fontstyle="italic")

    # ── Table coordinates ─────────────────────────────────────────────────────
    col_labels  = ["Category", "Test Suite / Description", "Total", "Pass", "Fail", "Status"]
    col_widths  = [0.12,        0.48,                        0.07,   0.07,   0.06,   0.10]
    col_x       = [0.05]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w)

    row_h   = 0.052
    table_t = 0.87
    header_y = table_t

    total_w = sum(col_widths)

    # Use fig.add_artist so patches are never clipped by axes bounds
    def _rect(x, y, w, h, fc, ec="#cccccc", lw=0.5, **kw):
        p = mpatches.FancyBboxPatch(
            (x, y - h), w, h,
            boxstyle="square,pad=0",
            facecolor=fc, edgecolor=ec, linewidth=lw,
            transform=fig.transFigure, clip_on=False,
        )
        fig.add_artist(p)

    def _text(x, y, s, ha="center", color="black", size=9, bold=False, **kw):
        fig.text(x, y, s, ha=ha, va="center", fontsize=size,
                 color=color, fontweight="bold" if bold else "normal", **kw)

    # Header row
    _rect(col_x[0], header_y, total_w, row_h, fc=HEADER_BG, ec=HEADER_BG)
    for i, (label, cx, cw) in enumerate(zip(col_labels, col_x, col_widths)):
        _text(cx + cw / 2, header_y - row_h / 2, label,
              color=HEADER_FG, bold=True, size=9)

    # Data rows
    grand_total = grand_pass = grand_fail = 0
    prev_cat = None
    cat_start_row = 0
    cat_rows_seen = {}

    for r_idx, (cat, desc, total, passed, failed) in enumerate(rows):
        y = header_y - (r_idx + 1) * row_h
        bg = ROW_ALT if r_idx % 2 else ROW_NORM
        _rect(col_x[0], y, total_w, row_h, fc=bg)

        # Category pill
        cat_col = CATEGORY_COLORS.get(cat, "#888888")
        pill_x = col_x[0] + 0.005
        pill_w = col_widths[0] - 0.01
        _rect(pill_x, y - 0.005, pill_w, row_h - 0.010, fc=cat_col, ec=cat_col)
        _text(pill_x + pill_w / 2, y - row_h / 2, cat,
              color="white", bold=True, size=8)

        _text(col_x[1] + 0.005, y - row_h / 2, desc, ha="left", size=8.5)
        _text(col_x[2] + col_widths[2] / 2, y - row_h / 2, str(total), size=9)
        _text(col_x[3] + col_widths[3] / 2, y - row_h / 2, str(passed),
              color="#1a7a3a", bold=True, size=9)

        fail_col = "#cc2222" if failed else "#888888"
        _text(col_x[4] + col_widths[4] / 2, y - row_h / 2, str(failed),
              color=fail_col, bold=(failed > 0), size=9)

        status_txt = "PASS" if failed == 0 else "FAIL"
        status_bg  = "#22c55e" if failed == 0 else "#ef4444"
        sx = col_x[5] + 0.01
        sw = col_widths[5] - 0.02
        _rect(sx, y - 0.006, sw, row_h - 0.012, fc=status_bg, ec=status_bg)
        _text(sx + sw / 2, y - row_h / 2, status_txt,
              color="white", bold=True, size=9)

        grand_total += total
        grand_pass  += passed
        grand_fail  += failed

    # Total row
    y_total = header_y - (n_data + 1) * row_h
    _rect(col_x[0], y_total, total_w, row_h, fc="#e8e8e8", ec="#aaaaaa", lw=1.0)
    _text(col_x[0] + col_widths[0] / 2, y_total - row_h / 2, "TOTAL",
          color="#1a1a1a", bold=True, size=9)
    _text(col_x[1] + 0.005, y_total - row_h / 2, "All mandatory test suites",
          ha="left", size=8.5, color="#444444")
    _text(col_x[2] + col_widths[2] / 2, y_total - row_h / 2,
          str(grand_total), bold=True, size=9)
    _text(col_x[3] + col_widths[3] / 2, y_total - row_h / 2,
          str(grand_pass), color="#1a7a3a", bold=True, size=9)
    _text(col_x[4] + col_widths[4] / 2, y_total - row_h / 2,
          str(grand_fail), color=("#cc2222" if grand_fail else "#888888"),
          bold=(grand_fail > 0), size=9)
    final_bg = "#22c55e" if grand_fail == 0 else "#ef4444"
    final_txt = "ALL PASS" if grand_fail == 0 else f"{grand_fail} FAILED"
    sx = col_x[5] + 0.01
    sw = col_widths[5] - 0.02
    _rect(sx, y_total - 0.006, sw, row_h - 0.012, fc=final_bg, ec=final_bg)
    _text(sx + sw / 2, y_total - row_h / 2, final_txt,
          color="white", bold=True, size=8.5)

    # ── Latency panel ─────────────────────────────────────────────────────────
    lat_y = y_total - row_h - 0.05
    lat_h = 0.14

    _rect(col_x[0], lat_y, total_w, lat_h, fc="#f0f7ff", ec="#3a7ebf", lw=1.2)

    fig.text(col_x[0] + 0.01, lat_y - 0.018,
             "NFR-01  AI Inference Latency  (50 calls, CPU, no GPU)",
             va="top", fontsize=9, fontweight="bold", color="#3a7ebf")

    metrics = [
        ("Mean",  f"{latency['mean']} ms"),
        ("p50",   f"{latency['p50']} ms"),
        ("p99",   f"{latency['p99']} ms"),
        ("Limit", f"{latency['limit']} ms"),
        ("Margin", f"{latency['limit'] - latency['p99']:.1f} ms headroom"),
    ]
    bar_x0  = col_x[0] + 0.01
    metric_w = total_w / len(metrics)
    for mi, (label, value) in enumerate(metrics):
        mx = bar_x0 + mi * metric_w
        fig.text(mx, lat_y - 0.055, label,
                 va="top", fontsize=8, color="#555555")
        color = "#22c55e" if "headroom" in value or label == "Margin" else "#1a1a1a"
        fig.text(mx, lat_y - 0.082, value,
                 va="top", fontsize=10, fontweight="bold", color=color)

    # bar: p99 proportion of limit
    bar_left  = col_x[0] + 0.01
    bar_right = col_x[0] + total_w - 0.01
    bar_bw    = bar_right - bar_left
    bar_by    = lat_y - lat_h + 0.025
    bar_bh    = 0.018
    _rect(bar_left, bar_by, bar_bw, bar_bh, fc="#dde8f5", ec="#aaccee")
    fill_w = bar_bw * min(latency["p99"] / latency["limit"], 1.0)
    _rect(bar_left, bar_by, fill_w, bar_bh, fc="#3a7ebf", ec="#3a7ebf")
    fig.text(bar_left + fill_w + 0.005, bar_by - bar_bh / 2,
             f"p99 = {latency['p99']} ms  /  limit = {latency['limit']} ms",
             va="center", fontsize=7.5, color="#444444")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_y = lat_y - lat_h - 0.025
    fig.text(col_x[0], legend_y, "Category key:", va="top",
             fontsize=8, color="#666666")
    lx = col_x[0] + 0.09
    for cat, col in CATEGORY_COLORS.items():
        patch = mpatches.Patch(facecolor=col, label=cat)
        fig.text(lx, legend_y, f"■ {cat}", va="top", fontsize=8,
                 color=col, fontweight="bold")
        lx += 0.09

    out = _HERE / "test_report_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out, grand_total, grand_pass, grand_fail


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("Measuring inference latency …")
    latency = _measure_latency()
    print(f"  mean={latency['mean']} ms  p50={latency['p50']} ms  "
          f"p99={latency['p99']} ms  limit={latency['limit']} ms")

    print("Running test suites …")
    rows = collect_results()

    print("Generating table …")
    out, total, passed, failed = make_table(rows, latency)

    print(f"\n{'='*50}")
    print(f"  Tests: {total}  |  Passed: {passed}  |  Failed: {failed}")
    print(f"  Figure: {out}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
