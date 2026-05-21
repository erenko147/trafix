"""
Metric: Junction-level fairness
Source: inline_metrics.json — junction_waiting section

For each signalized junction, computes variance (and std dev) of mean waiting
times across different approach lanes.  A high variance means the AI is
starving some directions to optimise others.

Reported per junction and as a network-wide average.
"""

import json
import math
from pathlib import Path
from typing import Union


def compute_junction_fairness(sumo_output_dir: Union[str, Path]) -> dict:
    """
    Returns
    -------
    {
        "network_wide_variance": float,   # mean of per-junction variances
        "network_wide_std_dev": float,
        "per_junction": {
            tls_id: {
                "approach_mean_wait_s": {edge_id: float},
                "variance": float,
                "std_dev": float,
            }
        }
    }
    """
    inline = Path(sumo_output_dir) / "inline_metrics.json"
    data   = json.loads(inline.read_text(encoding="utf-8"))

    jw = data.get("junction_waiting", {})
    per_junction = {}
    variances = []

    for tls_id, info in jw.items():
        var = info.get("variance", 0.0)
        per_junction[tls_id] = {
            "approach_mean_wait_s": info.get("approach_mean_wait_s", {}),
            "variance": var,
            "std_dev": info.get("std_dev", math.sqrt(var)),
        }
        variances.append(var)

    nw_var = sum(variances) / len(variances) if variances else 0.0
    return {
        "network_wide_variance": nw_var,
        "network_wide_std_dev": math.sqrt(nw_var),
        "per_junction": per_junction,
    }


if __name__ == "__main__":
    from pathlib import Path
    fixture = Path(__file__).parents[1] / "utils" / "fixtures"
    r = compute_junction_fairness(fixture)
    assert "J0" in r["per_junction"]
    print("junction_fairness: OK — nw_variance =", round(r["network_wide_variance"], 4))
