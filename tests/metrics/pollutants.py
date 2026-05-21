"""
Metric: Pollutant emissions breakdown — NOx, PMx, HC, CO
Source: tripinfo.xml — <emissions NOx_abs="..." PMx_abs="..." HC_abs="..." CO_abs="..."/>
All values in mg (HBEFA3 model).
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

_POLLUTANTS = ("NOx_abs", "PMx_abs", "HC_abs", "CO_abs")


def compute_pollutants(sumo_output_dir: Union[str, Path]) -> dict:
    """
    Returns
    -------
    {
        "totals_mg":   {"NOx": float, "PMx": float, "HC": float, "CO": float},
        "per_veh_mg":  {"NOx": float, "PMx": float, "HC": float, "CO": float},
        "total_trips": int,
    }
    """
    tripinfo = Path(sumo_output_dir) / "tripinfo.xml"
    tree = ET.parse(tripinfo)
    accum = {p: [] for p in _POLLUTANTS}
    for ti in tree.getroot().iter("tripinfo"):
        em = ti.find("emissions")
        if em is not None:
            for p in _POLLUTANTS:
                accum[p].append(float(em.get(p, 0)))

    n = len(accum["NOx_abs"])
    if n == 0:
        zero = {"NOx": 0.0, "PMx": 0.0, "HC": 0.0, "CO": 0.0}
        return {"totals_mg": zero, "per_veh_mg": zero, "total_trips": 0}

    totals  = {p.replace("_abs", ""): sum(v) for p, v in accum.items()}
    per_veh = {k: v / n for k, v in totals.items()}
    return {"totals_mg": totals, "per_veh_mg": per_veh, "total_trips": n}


if __name__ == "__main__":
    from pathlib import Path
    fixture = Path(__file__).parents[1] / "utils" / "fixtures"
    r = compute_pollutants(fixture)
    assert abs(r["totals_mg"]["NOx"] - (40 + 52 + 22)) < 0.01
    print("pollutants: OK —", r["totals_mg"])
