# Test Type 2 — Fixed-Timing vs AI Improvement
Simulation: 3600 s

Δ% = % by which the controller beats the **roundabout_fixed** baseline.
Positive = better than roundabout.


---
## Low traffic

| Metric | Roundabout (ref) | standard_fixed | Δ% vs roundabout | standard_ep1000 | Δ% vs roundabout | standard_ep2000 | Δ% vs roundabout |
|--------|-----------------| --- | --- | --- | --- | --- | --- |
| NOx Total (mg) | 1.879e+04 | 2.154e+04 | ✗ -14.7% | 1.921e+04 | ✗ -2.2% | 1.791e+04 | ✓ +4.6% |
| Cars Completed Trip | 201 | 203 | ✓ +1.0% | 200 | ✗ -0.5% | 202 | ✓ +0.5% |
| CO₂ per Vehicle (mg) | 2.635e+05 | 2.887e+05 | ✗ -9.6% | 2.639e+05 | ✗ -0.1% | 2.454e+05 | ✓ +6.8% |
| Fairness Variance | 8.065 | 21.56 | ✗ -167.3% | 2213 | ✗ -27335.9% | 11.57 | ✗ -43.4% |
| Fuel per Vehicle (L) | 0.1151 | 0.1261 | ✗ -9.6% | 0.1153 | ✗ -0.1% | 0.1072 | ✓ +6.8% |
| Network Speed (m/s) | 5.575 | 3.948 | ✗ -29.2% | 2.3 | ✗ -58.7% | 4.523 | ✗ -18.9% |
| Queue Length (veh/junction) | 0.5438 | 0.9819 | ✗ -80.6% | 0.7819 | ✗ -43.8% | 0.6491 | ✗ -19.4% |
| Stops per Vehicle | 2.058 | 1.602 | ✓ +22.1% | 1.61 | ✓ +21.8% | 1.551 | ✓ +24.6% |
| Teleports | 0 | 0 | ✓ +0.0% | 0 | ✓ +0.0% | 0 | ✓ +0.0% |
| Throughput (veh/hr) | 201 | 203 | ✓ +1.0% | 200 | ✗ -0.5% | 202 | ✓ +0.5% |
| Time Loss (s) | 74.3 | 104.9 | ✗ -41.1% | 88.12 | ✗ -18.6% | 75.92 | ✗ -2.2% |
| Travel Time (s) | 127.3 | 148.9 | ✗ -16.9% | 132.4 | ✗ -4.0% | 120 | ✓ +5.8% |
| Waiting Time (s) | 52.81 | 89.06 | ✗ -68.6% | 72.16 | ✗ -36.6% | 60.46 | ✗ -14.5% |

---
## Medium traffic

| Metric | Roundabout (ref) | standard_fixed | Δ% vs roundabout | standard_ep1000 | Δ% vs roundabout | standard_ep2000 | Δ% vs roundabout |
|--------|-----------------| --- | --- | --- | --- | --- | --- |
| NOx Total (mg) | 4.723e+04 | 5.711e+04 | ✗ -20.9% | 5.044e+04 | ✗ -6.8% | 4.712e+04 | ✓ +0.2% |
| Cars Completed Trip | 528 | 524 | ✗ -0.8% | 528 | ✓ -0.0% | 528 | ✓ -0.0% |
| CO₂ per Vehicle (mg) | 2.531e+05 | 2.964e+05 | ✗ -17.1% | 2.631e+05 | ✗ -3.9% | 2.471e+05 | ✓ +2.4% |
| Fairness Variance | 18.32 | 163.4 | ✗ -791.8% | 821.7 | ✗ -4385.6% | 425.2 | ✗ -2221.1% |
| Fuel per Vehicle (L) | 0.1106 | 0.1295 | ✗ -17.1% | 0.1149 | ✗ -3.9% | 0.108 | ✓ +2.4% |
| Network Speed (m/s) | 5.999 | 3.613 | ✗ -39.8% | 2.992 | ✗ -50.1% | 3.66 | ✗ -39.0% |
| Queue Length (veh/junction) | 1.28 | 2.696 | ✗ -110.6% | 2.016 | ✗ -57.5% | 1.706 | ✗ -33.3% |
| Stops per Vehicle | 2.489 | 1.905 | ✓ +23.5% | 1.786 | ✓ +28.3% | 1.813 | ✓ +27.2% |
| Teleports | 0 | 0 | ✓ +0.0% | 0 | ✓ +0.0% | 0 | ✓ +0.0% |
| Throughput (veh/hr) | 528 | 524 | ✗ -0.8% | 528 | ✓ -0.0% | 528 | ✓ -0.0% |
| Time Loss (s) | 66.74 | 108.3 | ✗ -62.3% | 86.01 | ✗ -28.9% | 75.31 | ✗ -12.8% |
| Travel Time (s) | 120.1 | 152.6 | ✗ -27.1% | 130.4 | ✗ -8.6% | 119.7 | ✓ +0.3% |
| Waiting Time (s) | 46.39 | 92.25 | ✗ -98.9% | 70.09 | ✗ -51.1% | 59.76 | ✗ -28.8% |

---
## High traffic

| Metric | Roundabout (ref) | standard_fixed | Δ% vs roundabout | standard_ep1000 | Δ% vs roundabout | standard_ep2000 | Δ% vs roundabout |
|--------|-----------------| --- | --- | --- | --- | --- | --- |
| NOx Total (mg) | 5.043e+04 | 1.12e+05 | ✗ -122.1% | 9.308e+04 | ✗ -84.6% | 8.656e+04 | ✗ -71.7% |
| Cars Completed Trip | 598 | 928 | ✓ +55.2% | 938 | ✓ +56.9% | 943 | ✓ +57.7% |
| CO₂ per Vehicle (mg) | 2.394e+05 | 3.28e+05 | ✗ -37.0% | 2.741e+05 | ✗ -14.5% | 2.555e+05 | ✗ -6.7% |
| Fairness Variance | 1.667e+07 | 3017 | ✓ +100.0% | 264.6 | ✓ +100.0% | 289.7 | ✓ +100.0% |
| Fuel per Vehicle (L) | 0.1046 | 0.1433 | ✗ -37.0% | 0.1198 | ✗ -14.5% | 0.1116 | ✗ -6.7% |
| Network Speed (m/s) | 2.804 | 3.697 | ✓ +31.8% | 4.768 | ✓ +70.0% | 5.29 | ✓ +88.6% |
| Queue Length (veh/junction) | 19.79 | 5.494 | ✓ +72.2% | 3.646 | ✓ +81.6% | 3.012 | ✓ +84.8% |
| Stops per Vehicle | 2.121 | 2.175 | ✗ -2.5% | 1.986 | ✓ +6.3% | 1.925 | ✓ +9.3% |
| Teleports | 0 | 0 | ✓ +0.0% | 0 | ✓ +0.0% | 0 | ✓ +0.0% |
| Throughput (veh/hr) | 598 | 928 | ✓ +55.2% | 938 | ✓ +56.9% | 943 | ✓ +57.7% |
| Time Loss (s) | 56.28 | 121.8 | ✗ -116.4% | 86.29 | ✗ -53.3% | 74.2 | ✗ -31.8% |
| Travel Time (s) | 112 | 170.3 | ✗ -52.0% | 134.9 | ✗ -20.5% | 122.6 | ✗ -9.5% |
| Waiting Time (s) | 38.1 | 103.2 | ✗ -170.9% | 69.25 | ✗ -81.7% | 57.29 | ✗ -50.4% |