"""
Metric: Average number of stops per vehicle
Source: inline_metrics.json — stops section (tracked via TraCI per step)
A stop is defined as a speed transition from ≥0.1 m/s to <0.1 m/s.
"""

import json
from pathlib import Path
from typing import Union


def compute_stops_per_vehicle(sumo_output_dir: Union[str, Path]) -> dict:
    """
    Returns
    -------
    {
        "avg_stops_per_vehicle": float,
        "total_stops": int,
        "total_vehicles": int,
    }
    """
    inline = Path(sumo_output_dir) / "inline_metrics.json"
    data = json.loads(inline.read_text(encoding="utf-8"))
    stops = data.get("stops", {})
    return {
        "avg_stops_per_vehicle": stops.get("avg_stops_per_vehicle", 0.0),
        "total_stops":           stops.get("total_stops", 0),
        "total_vehicles":        stops.get("total_vehicles", 0),
    }


if __name__ == "__main__":
    from pathlib import Path
    fixture = Path(__file__).parents[1] / "utils" / "fixtures"
    r = compute_stops_per_vehicle(fixture)
    assert abs(r["avg_stops_per_vehicle"] - 8 / 3) < 0.01
    print("stops_per_vehicle: OK —", round(r["avg_stops_per_vehicle"], 4))
