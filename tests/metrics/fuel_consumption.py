"""
Metric: Fuel consumption
Source: tripinfo.xml — <emissions fuel_abs="..."/>
With --emissions.volumetric-fuel true, fuel_abs is in mL; divide by 1000 for litres.
Without the flag, fuel_abs is in mg.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union


def compute_fuel_consumption(sumo_output_dir: Union[str, Path],
                              volumetric: bool = True) -> dict:
    """
    Parameters
    ----------
    volumetric : if True, fuel_abs is in mL → convert to L (divide by 1000).
                 if False, fuel_abs is in mg (raw HBEFA output).

    Returns
    -------
    {
        "total_fuel_L": float,
        "mean_fuel_per_vehicle_L": float,
        "total_trips": int,
        "raw_fuel_mL_or_mg": list[float],
    }
    """
    tripinfo = Path(sumo_output_dir) / "tripinfo.xml"
    tree = ET.parse(tripinfo)
    fuel_raw = []
    for ti in tree.getroot().iter("tripinfo"):
        em = ti.find("emissions")
        if em is not None:
            fuel_raw.append(float(em.get("fuel_abs", 0)))
    if not fuel_raw:
        return {"total_fuel_L": 0.0, "mean_fuel_per_vehicle_L": 0.0,
                "total_trips": 0, "raw_fuel_mL_or_mg": []}
    divisor = 1000.0 if volumetric else 1.0
    total_L = sum(fuel_raw) / divisor
    return {
        "total_fuel_L": total_L,
        "mean_fuel_per_vehicle_L": total_L / len(fuel_raw),
        "total_trips": len(fuel_raw),
        "raw_fuel_mL_or_mg": fuel_raw,
    }


if __name__ == "__main__":
    from pathlib import Path
    fixture = Path(__file__).parents[1] / "utils" / "fixtures"
    r = compute_fuel_consumption(fixture, volumetric=True)
    # fixture fuel_abs: 1.800 + 2.300 + 1.100 = 5.2 mL → 0.0052 L
    assert abs(r["total_fuel_L"] - (1.8 + 2.3 + 1.1) / 1000) < 1e-6
    print("fuel_consumption: OK —", round(r["total_fuel_L"], 6), "L")
