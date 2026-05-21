"""
Metric: Average waiting time at junctions
Source: tripinfo.xml — <tripinfo waitingTime="..."/>
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union


def compute_waiting_time(sumo_output_dir: Union[str, Path]) -> dict:
    """
    Returns
    -------
    {
        "mean_waiting_time_s": float,
        "total_waiting_time_s": float,
        "total_trips": int,
        "raw": list[float],
    }
    """
    tripinfo = Path(sumo_output_dir) / "tripinfo.xml"
    tree = ET.parse(tripinfo)
    waits = [
        float(el.get("waitingTime", 0))
        for el in tree.getroot().iter("tripinfo")
    ]
    if not waits:
        return {"mean_waiting_time_s": 0.0, "total_waiting_time_s": 0.0,
                "total_trips": 0, "raw": []}
    return {
        "mean_waiting_time_s": sum(waits) / len(waits),
        "total_waiting_time_s": sum(waits),
        "total_trips": len(waits),
        "raw": waits,
    }


if __name__ == "__main__":
    from pathlib import Path
    fixture = Path(__file__).parents[1] / "utils" / "fixtures"
    r = compute_waiting_time(fixture)
    assert abs(r["mean_waiting_time_s"] - (15 + 20 + 5) / 3) < 0.01
    print("waiting_time: OK")
