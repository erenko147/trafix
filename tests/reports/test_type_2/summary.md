# Test Type 2 — Junction & Controller Comparison
Simulation: 3600 s  |  Route files: shared (type1_low/medium/high.rou.xml)

| Key | Controller |
|-----|-----------|
| roundabout_fixed | Turkish-style roundabout × 5 — Webster 86 s fixed-timing TLS |
| standard_fixed   | Standard cross-intersection × 5 — SUMO built-in fixed timing |
| standard_ep1000  | Standard cross-intersection × 5 — AI TraFix v6 @ ep1000 |
| standard_ep2000  | Standard cross-intersection × 5 — AI TraFix v6 @ ep2000 (final) |


---
## Traffic Level: Low

| Metric | roundabout_fixed | standard_fixed | standard_ep1000 | standard_ep2000 | Winner |
|--------| -------- | -------- | -------- | -------- | ------ |
| NOx Total (mg) | 1.879e+04 | 2.154e+04 | 1.921e+04 | 1.791e+04 ★ | standard_ep2000 |
| Cars Completed Trip | 201 | 203 ★ | 200 | 202 | standard_fixed |
| CO₂ per Vehicle (mg) | 2.635e+05 | 2.887e+05 | 2.639e+05 | 2.454e+05 ★ | standard_ep2000 |
| Fairness Variance | 8.065 ★ | 21.56 | 2213 | 11.57 | roundabout_fixed |
| Fuel per Vehicle (L) | 0.1151 | 0.1261 | 0.1153 | 0.1072 ★ | standard_ep2000 |
| Network Speed (m/s) | 5.575 ★ | 3.948 | 2.3 | 4.523 | roundabout_fixed |
| Queue Length (veh/junction) | 0.5438 ★ | 0.9819 | 0.7819 | 0.6491 | roundabout_fixed |
| Stops per Vehicle | 2.058 | 1.602 | 1.61 | 1.551 ★ | standard_ep2000 |
| Teleports | 0 ★ | 0 | 0 | 0 | roundabout_fixed |
| Throughput (veh/hr) | 201 | 203 ★ | 200 | 202 | standard_fixed |
| Time Loss (s) | 74.3 ★ | 104.9 | 88.12 | 75.92 | roundabout_fixed |
| Travel Time (s) | 127.3 | 148.9 | 132.4 | 120 ★ | standard_ep2000 |
| Waiting Time (s) | 52.81 ★ | 89.06 | 72.16 | 60.46 | roundabout_fixed |

---
## Traffic Level: Medium

| Metric | roundabout_fixed | standard_fixed | standard_ep1000 | standard_ep2000 | Winner |
|--------| -------- | -------- | -------- | -------- | ------ |
| NOx Total (mg) | 4.723e+04 | 5.711e+04 | 5.044e+04 | 4.712e+04 ★ | standard_ep2000 |
| Cars Completed Trip | 528 ★ | 524 | 528 | 528 | roundabout_fixed |
| CO₂ per Vehicle (mg) | 2.531e+05 | 2.964e+05 | 2.631e+05 | 2.471e+05 ★ | standard_ep2000 |
| Fairness Variance | 18.32 ★ | 163.4 | 821.7 | 425.2 | roundabout_fixed |
| Fuel per Vehicle (L) | 0.1106 | 0.1295 | 0.1149 | 0.108 ★ | standard_ep2000 |
| Network Speed (m/s) | 5.999 ★ | 3.613 | 2.992 | 3.66 | roundabout_fixed |
| Queue Length (veh/junction) | 1.28 ★ | 2.696 | 2.016 | 1.706 | roundabout_fixed |
| Stops per Vehicle | 2.489 | 1.905 | 1.786 ★ | 1.813 | standard_ep1000 |
| Teleports | 0 ★ | 0 | 0 | 0 | roundabout_fixed |
| Throughput (veh/hr) | 528 ★ | 524 | 528 | 528 | roundabout_fixed |
| Time Loss (s) | 66.74 ★ | 108.3 | 86.01 | 75.31 | roundabout_fixed |
| Travel Time (s) | 120.1 | 152.6 | 130.4 | 119.7 ★ | standard_ep2000 |
| Waiting Time (s) | 46.39 ★ | 92.25 | 70.09 | 59.76 | roundabout_fixed |

---
## Traffic Level: High

| Metric | roundabout_fixed | standard_fixed | standard_ep1000 | standard_ep2000 | Winner |
|--------| -------- | -------- | -------- | -------- | ------ |
| NOx Total (mg) | 5.043e+04 ★ | 1.12e+05 | 9.308e+04 | 8.656e+04 | roundabout_fixed |
| Cars Completed Trip | 598 | 928 | 938 | 943 ★ | standard_ep2000 |
| CO₂ per Vehicle (mg) | 2.394e+05 ★ | 3.28e+05 | 2.741e+05 | 2.555e+05 | roundabout_fixed |
| Fairness Variance | 1.667e+07 | 3017 | 264.6 ★ | 289.7 | standard_ep1000 |
| Fuel per Vehicle (L) | 0.1046 ★ | 0.1433 | 0.1198 | 0.1116 | roundabout_fixed |
| Network Speed (m/s) | 2.804 | 3.697 | 4.768 | 5.29 ★ | standard_ep2000 |
| Queue Length (veh/junction) | 19.79 | 5.494 | 3.646 | 3.012 ★ | standard_ep2000 |
| Stops per Vehicle | 2.121 | 2.175 | 1.986 | 1.925 ★ | standard_ep2000 |
| Teleports | 0 ★ | 0 | 0 | 0 | roundabout_fixed |
| Throughput (veh/hr) | 598 | 928 | 938 | 943 ★ | standard_ep2000 |
| Time Loss (s) | 56.28 ★ | 121.8 | 86.29 | 74.2 | roundabout_fixed |
| Travel Time (s) | 112 ★ | 170.3 | 134.9 | 122.6 | roundabout_fixed |
| Waiting Time (s) | 38.1 ★ | 103.2 | 69.25 | 57.29 | roundabout_fixed |

---
## Wins Tally (across all traffic levels)

| Controller | Metrics Won |
|-----------|------------|
| roundabout_fixed | 21 |
| standard_fixed | 2 |
| standard_ep1000 | 2 |
| standard_ep2000 | 14 |