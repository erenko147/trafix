"""
Metric: Time loss — difference between actual and ideal (free-flow) travel time
Source: tripinfo.xml — <tripinfo timeLoss="..."/> (completed trips)
        + unfinished_vehicles.json — accumulated time loss of cars still stuck in
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


def compute_time_loss(sumo_output_dir: Union[str, Path],
                      include_unfinished: bool = True) -> dict:
    """
    Returns
    -------
    {
        "mean_time_loss_s": float,   # over completed + still-running cars
        "total_time_loss_s": float,
        "total_trips": int,
        "completed_trips": int,
        "unfinished_trips": int,
        "raw": list[float],
    }
    """
    out = Path(sumo_output_dir)
    tree = ET.parse(out / "tripinfo.xml")
    completed = [
        float(el.get("timeLoss", 0))
        for el in tree.getroot().iter("tripinfo")
    ]
    unfinished = _load_unfinished(out, "time_loss_s") if include_unfinished else []
    losses = completed + unfinished
    if not losses:
        return {"mean_time_loss_s": 0.0, "total_time_loss_s": 0.0,
                "total_trips": 0, "completed_trips": 0, "unfinished_trips": 0,
                "raw": []}
    return {
        "mean_time_loss_s": sum(losses) / len(losses),
        "total_time_loss_s": sum(losses),
        "total_trips": len(losses),
        "completed_trips": len(completed),
        "unfinished_trips": len(unfinished),
        "raw": losses,
    }


if __name__ == "__main__":
    from pathlib import Path
    fixture = Path(__file__).parents[1] / "utils" / "fixtures"
    r = compute_time_loss(fixture)
    assert abs(r["mean_time_loss_s"] - (10 + 14 + 4) / 3) < 0.01
    print("time_loss: OK —", round(r["mean_time_loss_s"], 2), "s")
