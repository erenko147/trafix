"""
Metric: Average queue length per junction
Source: inline_metrics.json — junction_queue section
        (halting vehicle counts tracked via TraCI per step)
Also parses queue.xml for lane-level queue lengths as supplementary data.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union


def compute_queue_length(sumo_output_dir: Union[str, Path]) -> dict:
    """
    Returns
    -------
    {
        "mean_halting_per_junction": float,   # network-wide mean across all junctions
        "per_junction": {tls_id: mean_halting_total},
        "lane_queue_from_xml": {lane_id: mean_queueing_length},  # from queue.xml
    }
    """
    out = Path(sumo_output_dir)

    # Primary: inline metrics
    inline_path = out / "inline_metrics.json"
    per_junction = {}        # halting (< 0.1 m/s) — SUMO's strict "stopped"
    per_junction_slow = {}   # crawl-aware (< slow threshold) — fair vs roundabouts
    if inline_path.exists():
        data = json.loads(inline_path.read_text(encoding="utf-8"))
        for tls_id, q in data.get("junction_queue", {}).items():
            per_junction[tls_id] = q.get("mean_halting_total", 0.0)
            # fall back to halting if a run predates the slow-queue tracking
            per_junction_slow[tls_id] = q.get("mean_slow_total",
                                              q.get("mean_halting_total", 0.0))

    mean_total = (
        sum(per_junction.values()) / len(per_junction)
        if per_junction else 0.0
    )
    mean_slow = (
        sum(per_junction_slow.values()) / len(per_junction_slow)
        if per_junction_slow else 0.0
    )

    # Supplementary: queue.xml for lane-level data
    lane_queue = {}
    queue_xml = out / "queue.xml"
    if queue_xml.exists():
        try:
            tree = ET.parse(queue_xml)
            lane_accum: dict = {}
            for ts in tree.getroot().findall("data"):
                lanes_el = ts.find("lanes")
                if lanes_el is None:
                    continue
                for lane in lanes_el.findall("lane"):
                    lid = lane.get("id", "")
                    q   = float(lane.get("queueing_length", 0))
                    lane_accum.setdefault(lid, []).append(q)
            lane_queue = {
                lid: sum(vals) / len(vals) if vals else 0.0
                for lid, vals in lane_accum.items()
            }
        except Exception:
            pass

    return {
        "mean_halting_per_junction": mean_total,        # < 0.1 m/s (strict)
        "mean_slow_per_junction": mean_slow,            # < slow threshold (crawl-aware)
        "per_junction": per_junction,
        "per_junction_slow": per_junction_slow,
        "lane_queue_from_xml": lane_queue,
    }


if __name__ == "__main__":
    from pathlib import Path
    fixture = Path(__file__).parents[1] / "utils" / "fixtures"
    r = compute_queue_length(fixture)
    assert "J0" in r["per_junction"]
    print("queue_length: OK —", r["mean_halting_per_junction"])
