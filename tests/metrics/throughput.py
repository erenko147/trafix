"""
Metric: Throughput — vehicles completing routes per simulated hour
Source: tripinfo.xml — count of <tripinfo> elements (arrived vehicles)
        summary.xml   — loaded / inserted / running at sim end
        unfinished_vehicles.json — exact count of cars still in the network

Distinguishes three failure modes that a raw 'arrived' count hides:
  • arrived           — completed their trip
  • still_running     — in the network at sim end (departed, not arrived)
  • not_inserted      — wanted to depart but could never enter (gridlock at fringe)
'cars_not_completed' = still_running + not_inserted = demand the controller failed
to serve. completion_rate = arrived / total_demand.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union


def compute_throughput(sumo_output_dir: Union[str, Path],
                       sim_duration_s: int = 3600) -> dict:
    out = Path(sumo_output_dir)
    tree = ET.parse(out / "tripinfo.xml")
    arrived = sum(1 for _ in tree.getroot().iter("tripinfo"))
    hours = sim_duration_s / 3600.0

    # summary.xml last step: cumulative loaded/inserted + current running
    loaded = inserted = running = 0
    summary_path = out / "summary.xml"
    if summary_path.exists():
        try:
            steps = ET.parse(summary_path).getroot().findall("step")
            if steps:
                last = steps[-1]
                loaded   = int(last.get("loaded", 0))
                inserted = int(last.get("inserted", arrived))
                running  = int(last.get("running", 0))
        except Exception:
            pass

    # Prefer the exact still-running count captured via TraCI at sim end
    still_running = running
    unf = out / "unfinished_vehicles.json"
    if unf.exists():
        try:
            still_running = int(json.loads(unf.read_text(encoding="utf-8")).get("count", running))
        except Exception:
            pass

    not_inserted = max(0, loaded - inserted)              # never entered the net
    total_demand = loaded if loaded > 0 else (arrived + still_running + not_inserted)
    cars_not_completed = max(0, still_running + not_inserted)
    completion_rate = (arrived / total_demand) if total_demand > 0 else 0.0

    return {
        "arrived_vehicles":       arrived,
        "vehicles_still_running": still_running,
        "not_inserted":           not_inserted,
        "cars_not_completed":     cars_not_completed,
        "total_demand":           total_demand,
        "completion_rate":        completion_rate,
        "total_departed":         inserted,
        "throughput_veh_per_hr":  arrived / hours if hours > 0 else 0.0,
        "sim_duration_s":         sim_duration_s,
    }


if __name__ == "__main__":
    from pathlib import Path
    fixture = Path(__file__).parents[1] / "utils" / "fixtures"
    r = compute_throughput(fixture, sim_duration_s=3600)
    assert r["arrived_vehicles"] == 3
    print("throughput: OK —", r["throughput_veh_per_hr"], "veh/hr")
