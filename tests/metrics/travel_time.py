"""
Metric: Average travel time
Source: tripinfo.xml — <tripinfo duration="..."/> (completed trips)
        + unfinished_vehicles.json — cars still in network at sim end (their
          time-in-network), so a controller that gridlocks cars does NOT get to
          exclude their long travel times from the average (survivorship bias).
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union


def _load_unfinished(out_dir: Path, key: str) -> list:
    """Per-vehicle values for cars still in the network at sim end (or [])."""
    p = out_dir / "unfinished_vehicles.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [float(v.get(key, 0.0)) for v in data.get("vehicles", [])]
    except Exception:
        return []


def compute_travel_time(sumo_output_dir: Union[str, Path],
                        include_unfinished: bool = True) -> dict:
    """
    Returns
    -------
    {
        "mean_travel_time_s": float,   # mean over completed + still-running cars
        "min_s": float,
        "max_s": float,
        "total_trips": int,            # completed + still-running
        "completed_trips": int,
        "unfinished_trips": int,
        "raw": list[float],            # per-vehicle durations
    }
    """
    out = Path(sumo_output_dir)
    tree = ET.parse(out / "tripinfo.xml")
    completed = [
        float(el.get("duration", 0))
        for el in tree.getroot().iter("tripinfo")
    ]
    unfinished = _load_unfinished(out, "time_in_network_s") if include_unfinished else []
    durations = completed + unfinished
    if not durations:
        return {"mean_travel_time_s": 0.0, "min_s": 0.0, "max_s": 0.0,
                "total_trips": 0, "completed_trips": 0, "unfinished_trips": 0,
                "raw": []}
    return {
        "mean_travel_time_s": sum(durations) / len(durations),
        "min_s": min(durations),
        "max_s": max(durations),
        "total_trips": len(durations),
        "completed_trips": len(completed),
        "unfinished_trips": len(unfinished),
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
