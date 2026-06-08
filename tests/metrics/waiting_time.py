"""
Metric: Average waiting time at junctions
Source: tripinfo.xml — <tripinfo waitingTime="..."/> (completed trips)
        + unfinished_vehicles.json — accumulated waiting of cars still stuck in
          the network at sim end (so gridlocked cars are not silently excluded).
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union


def _load_unfinished(out_dir: Path, key: str) -> list:
    p = out_dir / "unfinished_vehicles.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [float(v.get(key, 0.0)) for v in data.get("vehicles", [])]
    except Exception:
        return []


def compute_waiting_time(sumo_output_dir: Union[str, Path],
                         include_unfinished: bool = True) -> dict:
    """
    Returns
    -------
    {
        "mean_waiting_time_s": float,   # over completed + still-running cars
        "total_waiting_time_s": float,
        "total_trips": int,
        "completed_trips": int,
        "unfinished_trips": int,
        "raw": list[float],
    }
    """
    out = Path(sumo_output_dir)
    tree = ET.parse(out / "tripinfo.xml")
    completed = [
        float(el.get("waitingTime", 0))
        for el in tree.getroot().iter("tripinfo")
    ]
    unfinished = _load_unfinished(out, "waiting_time_s") if include_unfinished else []
    waits = completed + unfinished
    if not waits:
        return {"mean_waiting_time_s": 0.0, "total_waiting_time_s": 0.0,
                "total_trips": 0, "completed_trips": 0, "unfinished_trips": 0,
                "raw": []}
    return {
        "mean_waiting_time_s": sum(waits) / len(waits),
        "total_waiting_time_s": sum(waits),
        "total_trips": len(waits),
        "completed_trips": len(completed),
        "unfinished_trips": len(unfinished),
        "raw": waits,
    }


if __name__ == "__main__":
    from pathlib import Path
    fixture = Path(__file__).parents[1] / "utils" / "fixtures"
    r = compute_waiting_time(fixture)
    assert abs(r["mean_waiting_time_s"] - (15 + 20 + 5) / 3) < 0.01
    print("waiting_time: OK")
