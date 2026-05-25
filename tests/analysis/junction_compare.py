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
    m["queue_length"]        = compute_queue_length(d)["mean_halting_per_junction"]
    tp = compute_throughput(d, sim_s)
    m["arrived_vehicles"]    = float(tp["arrived_vehicles"])
    m["vehicles_still_running"] = float(tp["vehicles_still_running"])
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

    # Generate figures
    from analysis.make_figures_type2 import make_all_figures
    make_all_figures(data, reports_dir / "charts")

    return data
