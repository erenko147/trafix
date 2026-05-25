# Test Type 2 — Cross-Checkpoint Comparison (ep1000 vs ep2000)
Simulation: 3600 s  |  Both on standard cross-intersection network

Δ% = improvement of ep2000 over ep1000 (positive = ep2000 better).


---
## Low traffic

| Metric | ep1000 | ep2000 | Δ% (ep2000 vs ep1000) |
|--------|--------|--------|----------------------|
| NOx Total (mg) | 1.921e+04 | 1.791e+04 | ✓ +6.7% |
| Cars Completed Trip | 200 | 202 | ✓ +1.0% |
| CO₂ per Vehicle (mg) | 2.639e+05 | 2.454e+05 | ✓ +7.0% |
| Fairness Variance | 2213 | 11.57 | ✓ +99.5% |
| Fuel per Vehicle (L) | 0.1153 | 0.1072 | ✓ +7.0% |
| Network Speed (m/s) | 2.3 | 4.523 | ✓ +96.7% |
| Queue Length (veh/junction) | 0.7819 | 0.6491 | ✓ +17.0% |
| Stops per Vehicle | 1.61 | 1.551 | ✓ +3.7% |
| Teleports | 0 | 0 | ✓ +0.0% |
| Throughput (veh/hr) | 200 | 202 | ✓ +1.0% |
| Time Loss (s) | 88.12 | 75.92 | ✓ +13.8% |
| Travel Time (s) | 132.4 | 120 | ✓ +9.4% |
| Waiting Time (s) | 72.16 | 60.46 | ✓ +16.2% |

---
## Medium traffic

| Metric | ep1000 | ep2000 | Δ% (ep2000 vs ep1000) |
|--------|--------|--------|----------------------|
| NOx Total (mg) | 5.044e+04 | 4.712e+04 | ✓ +6.6% |
| Cars Completed Trip | 528 | 528 | ✓ -0.0% |
| CO₂ per Vehicle (mg) | 2.631e+05 | 2.471e+05 | ✓ +6.1% |
| Fairness Variance | 821.7 | 425.2 | ✓ +48.3% |
| Fuel per Vehicle (L) | 0.1149 | 0.108 | ✓ +6.1% |
| Network Speed (m/s) | 2.992 | 3.66 | ✓ +22.3% |
| Queue Length (veh/junction) | 2.016 | 1.706 | ✓ +15.4% |
| Stops per Vehicle | 1.786 | 1.813 | ✗ -1.5% |
| Teleports | 0 | 0 | ✓ +0.0% |
| Throughput (veh/hr) | 528 | 528 | ✓ -0.0% |
| Time Loss (s) | 86.01 | 75.31 | ✓ +12.4% |
| Travel Time (s) | 130.4 | 119.7 | ✓ +8.2% |
| Waiting Time (s) | 70.09 | 59.76 | ✓ +14.7% |

---
## High traffic

| Metric | ep1000 | ep2000 | Δ% (ep2000 vs ep1000) |
|--------|--------|--------|----------------------|
| NOx Total (mg) | 9.308e+04 | 8.656e+04 | ✓ +7.0% |
| Cars Completed Trip | 938 | 943 | ✓ +0.5% |
| CO₂ per Vehicle (mg) | 2.741e+05 | 2.555e+05 | ✓ +6.8% |
| Fairness Variance | 264.6 | 289.7 | ✗ -9.5% |
| Fuel per Vehicle (L) | 0.1198 | 0.1116 | ✓ +6.8% |
| Network Speed (m/s) | 4.768 | 5.29 | ✓ +11.0% |
| Queue Length (veh/junction) | 3.646 | 3.012 | ✓ +17.4% |
| Stops per Vehicle | 1.986 | 1.925 | ✓ +3.1% |
| Teleports | 0 | 0 | ✓ +0.0% |
| Throughput (veh/hr) | 938 | 943 | ✓ +0.5% |
| Time Loss (s) | 86.29 | 74.2 | ✓ +14.0% |
| Travel Time (s) | 134.9 | 122.6 | ✓ +9.1% |
| Waiting Time (s) | 69.25 | 57.29 | ✓ +17.3% |