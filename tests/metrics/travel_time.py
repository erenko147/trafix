"""
Metric: Average travel time
Source: tripinfo.xml — <tripinfo duration="..."/>
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union


def compute_travel_time(sumo_output_dir: Union[str, Path]) -> dict:
    """
    Returns
    -------
    {
        "mean_travel_time_s": float,   # mean trip duration (seconds)
        "min_s": float,
        "max_s": float,
        "total_trips": int,
        "raw": list[float],            # per-vehicle durations
    }
    """
    tripinfo = Path(sumo_output_dir) / "tripinfo.xml"
    tree = ET.parse(tripinfo)
    durations = [
        float(el.get("duration", 0))
        for el in tree.getroot().iter("tripinfo")
    ]
    if not durations:
        return {"mean_travel_time_s": 0.0, "min_s": 0.0, "max_s": 0.0,
                "total_trips": 0, "raw": []}
    return {
        "mean_travel_time_s": sum(durations) / len(durations),
        "min_s": min(durations),
        "max_s": max(durations),
        "total_trips": len(durations),
        "raw": durations,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    fixture = Path(__file__).parents[1] / "utils" / "fixtures"
    result = compute_travel_time(fixture)
    print(result)
    assert abs(result["mean_travel_time_s"] - (60 + 70 + 40) / 3) < 0.01
    print("travel_time: OK")
