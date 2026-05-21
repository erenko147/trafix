"""
Metric: CO₂ emissions
Source: tripinfo.xml — <emissions CO2_abs="..."/>
Units: CO2_abs is in mg when volumetric-fuel is not set; with --device.emissions.probability 1.0
       SUMO writes total CO2 in mg per vehicle for HBEFA model.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union


def compute_emissions(sumo_output_dir: Union[str, Path]) -> dict:
    """
    Returns
    -------
    {
        "total_CO2_mg": float,
        "mean_CO2_per_vehicle_mg": float,
        "total_trips": int,
        "raw_CO2": list[float],
    }
    """
    tripinfo = Path(sumo_output_dir) / "tripinfo.xml"
    tree = ET.parse(tripinfo)
    co2_values = []
    for ti in tree.getroot().iter("tripinfo"):
        em = ti.find("emissions")
        if em is not None:
            co2_values.append(float(em.get("CO2_abs", 0)))
    if not co2_values:
        return {"total_CO2_mg": 0.0, "mean_CO2_per_vehicle_mg": 0.0,
                "total_trips": 0, "raw_CO2": []}
    return {
        "total_CO2_mg": sum(co2_values),
        "mean_CO2_per_vehicle_mg": sum(co2_values) / len(co2_values),
        "total_trips": len(co2_values),
        "raw_CO2": co2_values,
    }


if __name__ == "__main__":
    from pathlib import Path
    fixture = Path(__file__).parents[1] / "utils" / "fixtures"
    r = compute_emissions(fixture)
    assert r["total_CO2_mg"] == 2000 + 2500 + 1200
    print("emissions CO2: OK")
