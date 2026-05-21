"""
Metric: Number of teleports (gridlock recovery events)
Source: summary.xml — sum of <step teleports="..."/>
Lower is better.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union


def compute_teleports(sumo_output_dir: Union[str, Path]) -> dict:
    """
    Returns
    -------
    {
        "total_teleports": int,
        "teleports_per_1000veh": float,   # normalised rate
        "arrived_vehicles": int,
    }
    """
    out = Path(sumo_output_dir)
    summary = out / "summary.xml"
    tree = ET.parse(summary)
    total_teleports = sum(
        int(step.get("teleports", 0))
        for step in tree.getroot().findall("step")
    )
    # arrived from last step
    steps = tree.getroot().findall("step")
    arrived = int(steps[-1].get("arrived", 0)) if steps else 0

    # Also check statistics.xml for validated total
    stats_xml = out / "statistics.xml"
    if stats_xml.exists():
        try:
            st = ET.parse(stats_xml)
            tp_el = st.getroot().find("teleports")
            if tp_el is not None:
                total_teleports = int(tp_el.get("total", total_teleports))
        except Exception:
            pass

    rate = (total_teleports / arrived * 1000) if arrived > 0 else 0.0
    return {
        "total_teleports": total_teleports,
        "teleports_per_1000veh": rate,
        "arrived_vehicles": arrived,
    }


if __name__ == "__main__":
    from pathlib import Path
    fixture = Path(__file__).parents[1] / "utils" / "fixtures"
    r = compute_teleports(fixture)
    assert r["total_teleports"] == 1    # one teleport at timestep 1.00 in fixture
    print("teleports: OK —", r)
