"""
Live data simulator for development / dashboard testing.

Sends randomly generated telemetry to the backend every 2 seconds so the
dashboard can be tested without a running SUMO instance.

Usage:
  python scripts/simulate.py           # port 8000
  python scripts/simulate.py 8001
"""

import json
import random
import sys
import time

import requests

_port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
_BATCH_URL = f"http://127.0.0.1:{_port}/telemetry_batch"


def _random_telemetry(intersection_id: int, current_phase: int) -> dict:
    n = random.randint(0, 1)
    s = random.randint(0, 1)
    e = random.randint(0, 1)
    w = random.randint(0, 1)
    total = n + s + e + w
    queue = min(50.0, total * random.uniform(0.5, 1.5))

    if random.random() < 0.2:
        current_phase  = random.randint(0, 3)
        phase_duration = random.uniform(1, 5)
    else:
        phase_duration = random.uniform(10, 45)

    return {
        "intersection_id": intersection_id,
        "north_count":     n,
        "south_count":     s,
        "east_count":      e,
        "west_count":      w,
        "queue_length":    round(queue, 1),
        "current_phase":   current_phase,
        "phase_duration":  round(phase_duration, 1),
    }


def main():
    print(f"TraFix Live Data Simulator  →  {_BATCH_URL}")
    print("Press CTRL+C to stop.\n")

    phases = [0, 1, 0, 2, 3]
    step   = 1

    try:
        while True:
            print(f"--- Step {step} ---")
            payload = {
                "step": step,
                "intersections": [
                    _random_telemetry(i, phases[i]) for i in range(5)
                ],
            }

            try:
                response = requests.post(_BATCH_URL, json=payload, timeout=2)
                if response.status_code == 200:
                    for decision in response.json().get("decisions", []):
                        idx        = decision["intersection_id"]
                        next_phase = decision["next_phase"]
                        phases[idx] = next_phase
                        print(f"  Junction {idx}: phase → {next_phase}")
                else:
                    print(f"  ERROR: status {response.status_code}")
            except Exception:
                print("  CONNECTION ERROR (is the backend running?)")

            print("Waiting 2s...\n")
            time.sleep(2)
            step += 1

    except KeyboardInterrupt:
        print("\nSimulator stopped.")


if __name__ == "__main__":
    main()
