"""
Metric: Time loss — difference between actual and ideal (free-flow) travel time
Source: tripinfo.xml — <tripinfo timeLoss="..."/>
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union


def compute_time_loss(sumo_output_dir: Union[str, Path]) -> dict:
    """
    Returns
    -------
    {
        "mean_time_loss_s": float,
        "total_time_loss_s": float,
        "total_trips": int,
        "raw": list[float],
    }
    """
    tripinfo = Path(sumo_output_dir) / "tripinfo.xml"
    tree = ET.parse(tripinfo)
    losses = [
        float(el.get("timeLoss", 0))
        for el in tree.getroot().iter("tripinfo")
    ]
    if not losses:
        return {"mean_time_loss_s": 0.0, "total_time_loss_s": 0.0,
                "total_trips": 0, "raw": []}
    return {
        "mean_time_loss_s": sum(losses) / len(losses),
        "total_time_loss_s": sum(losses),
        "total_trips": len(losses),
        "raw": losses,
    }


if __name__ == "__main__":
    from pathlib import Path
    fixture = Path(__file__).parents[1] / "utils" / "fixtures"
    r = compute_time_loss(fixture)
    assert abs(r["mean_time_loss_s"] - (10 + 14 + 4) / 3) < 0.01
    print("time_loss: OK —", round(r["mean_time_loss_s"], 2), "s")
