"""
Metric: Throughput — vehicles completing routes per simulated hour
Source: tripinfo.xml — count of <tripinfo> elements (arrived vehicles)
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union


def compute_throughput(sumo_output_dir: Union[str, Path],
                       sim_duration_s: int = 3600) -> dict:
    """
    Returns
    -------
    {
        "arrived_vehicles": int,       # vehicles that completed their route
        "vehicles_still_running": int, # departed but not yet arrived at sim end
        "total_departed": int,         # arrived + still_running
        "throughput_veh_per_hr": float,
        "sim_duration_s": int,
    }
    """
    tripinfo = Path(sumo_output_dir) / "tripinfo.xml"
    tree = ET.parse(tripinfo)
    arrived = sum(1 for _ in tree.getroot().iter("tripinfo"))
    hours = sim_duration_s / 3600.0

    # summary.xml last step has cumulative departed and arrived counts
    summary_path = Path(sumo_output_dir) / "summary.xml"
    total_departed = arrived
    if summary_path.exists():
        try:
            steps = ET.parse(summary_path).getroot().findall("step")
            if steps:
                last = steps[-1]
                total_departed = int(last.get("inserted", arrived))
        except Exception:
            pass

    return {
        "arrived_vehicles":    arrived,
        "vehicles_still_running": max(0, total_departed - arrived),
        "total_departed":      total_departed,
        "throughput_veh_per_hr": arrived / hours if hours > 0 else 0.0,
        "sim_duration_s":      sim_duration_s,
    }


if __name__ == "__main__":
    from pathlib import Path
    fixture = Path(__file__).parents[1] / "utils" / "fixtures"
    r = compute_throughput(fixture, sim_duration_s=3600)
    assert r["arrived_vehicles"] == 3
    print("throughput: OK —", r["throughput_veh_per_hr"], "veh/hr")
