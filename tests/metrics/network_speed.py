"""
Metric: Average network speed
Source: summary.xml — <step meanSpeed="..."/> averaged over all timesteps
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union


def compute_network_speed(sumo_output_dir: Union[str, Path]) -> dict:
    """
    Returns
    -------
    {
        "mean_speed_ms": float,    # time-averaged mean speed across vehicles (m/s)
        "min_step_speed_ms": float,
        "max_step_speed_ms": float,
        "steps_recorded": int,
    }
    """
    summary = Path(sumo_output_dir) / "summary.xml"
    tree = ET.parse(summary)
    speeds = []
    for step in tree.getroot().findall("step"):
        running = int(step.get("running", 0))
        if running > 0:
            speeds.append(float(step.get("meanSpeed", 0)))
    if not speeds:
        return {"mean_speed_ms": 0.0, "min_step_speed_ms": 0.0,
                "max_step_speed_ms": 0.0, "steps_recorded": 0}
    return {
        "mean_speed_ms": sum(speeds) / len(speeds),
        "min_step_speed_ms": min(speeds),
        "max_step_speed_ms": max(speeds),
        "steps_recorded": len(speeds),
    }


if __name__ == "__main__":
    from pathlib import Path
    fixture = Path(__file__).parents[1] / "utils" / "fixtures"
    r = compute_network_speed(fixture)
    assert r["steps_recorded"] > 0
    print("network_speed: OK —", round(r["mean_speed_ms"], 3), "m/s")
