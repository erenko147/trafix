# Test Type 2 — Junction & Controller Comparison
Simulation: 3600 s  |  Route files: shared (type1_low/medium/high.rou.xml)

> Metrics now include cars still stuck in the network at sim end (no more
> survivorship bias), queue counts crawling cars (< 5 km/h), and AI uses
> greedy argmax (matches production). Teleporting disabled in every run.

| Key | Controller |
|-----|-----------|
| roundabout_fixed | Turkish-style roundabout × 5 — Webster 86 s fixed TLS |
| standard_fixed   | Standard cross-intersection × 5 — SUMO fixed timing |
| standard_ep1000  | Standard cross-intersection × 5 — AI TraFix v6 @ ep1000 |
| standard_ep2000  | Standard cross-intersection × 5 — AI TraFix v6 @ ep2000 (final) |

---
## Traffic Level: Low

| Metric | roundabout_fixed | standard_fixed | standard_ep1000 | standard_ep2000 | Winner |
|--------|--------|--------|--------|--------|------|
| Trip Completion (%) | 75.76 | 76.89 | 77.27 ★ | 76.89 | standard_ep1000 |
| Cars NOT Completed | 64 | 61 | 60 ★ | 61 | standard_ep1000 |
| Cars Completed Trip | 200.0 | 203.0 | 204.0 ★ | 203.0 | standard_ep1000 |
| Cars Stuck in Network | 64 | 61 | 60 ★ | 61 | standard_ep1000 |
| Cars Never Inserted | 0 ★ | 0 | 0 | 0 | roundabout_fixed |
| Throughput (veh/hr) | 200.0 | 203.0 | 204.0 ★ | 203.0 | standard_ep1000 |
| Waiting Time (s) | 32.11 ★ | 70.67 | 39.22 | 41.15 | roundabout_fixed |
| Travel Time (s) | 96.4 | 121.7 | 90.14 ★ | 91.48 | standard_ep1000 |
| Time Loss (s) | 47.77 ★ | 84.15 | 52.01 | 53.45 | roundabout_fixed |
| Queue (slow veh/junction) | 0.4543 ★ | 1.017 | 0.5434 | 0.5681 | roundabout_fixed |
| Network Speed (m/s) | 5.554 | 4.005 | 5.348 | 5.781 ★ | standard_ep2000 |
| Stops per Vehicle | 1.569 | 1.602 | 1.358 | 1.346 ★ | standard_ep2000 |
| CO₂ per Vehicle (mg) | 2.465e+05 | 2.875e+05 | 2.227e+05 ★ | 2.258e+05 | standard_ep1000 |
| Fuel per Vehicle (L) | 0.1077 | 0.1256 | 0.09731 ★ | 0.09867 | standard_ep1000 |
| NOx Total (mg) | 1.708e+04 | 2.146e+04 | 1.631e+04 ★ | 1.651e+04 | standard_ep1000 |
| Fairness Variance | 0.2836 ★ | 22.47 | 11.77 | 12.47 | roundabout_fixed |

---
## Traffic Level: Medium

| Metric | roundabout_fixed | standard_fixed | standard_ep1000 | standard_ep2000 | Winner |
|--------|--------|--------|--------|--------|------|
| Trip Completion (%) | 100.0 ★ | 99.43 | 100.0 | 100.0 | roundabout_fixed |
| Cars NOT Completed | 0 ★ | 3 | 0 | 0 | roundabout_fixed |
| Cars Completed Trip | 528.0 ★ | 525.0 | 528.0 | 528.0 | roundabout_fixed |
| Cars Stuck in Network | 0 ★ | 3 | 0 | 0 | roundabout_fixed |
| Cars Never Inserted | 0 ★ | 0 | 0 | 0 | roundabout_fixed |
| Throughput (veh/hr) | 528.0 ★ | 525.0 | 528.0 | 528.0 | roundabout_fixed |
| Waiting Time (s) | 37.06 ★ | 92.87 | 39.46 | 42.06 | roundabout_fixed |
| Travel Time (s) | 115.5 | 153.3 | 97.88 ★ | 100.5 | standard_ep1000 |
| Time Loss (s) | 55.9 | 109.0 | 53.46 ★ | 56.06 | standard_ep1000 |
| Queue (slow veh/junction) | 1.152 ★ | 2.796 | 1.185 | 1.263 | roundabout_fixed |
| Network Speed (m/s) | 5.999 | 3.602 | 5.314 | 6.169 ★ | standard_ep2000 |
| Stops per Vehicle | 1.918 | 1.919 | 1.475 ★ | 1.479 | standard_ep1000 |
| CO₂ per Vehicle (mg) | 2.46e+05 | 2.961e+05 | 2.106e+05 ★ | 2.147e+05 | standard_ep1000 |
| Fuel per Vehicle (L) | 0.1075 | 0.1293 | 0.09201 ★ | 0.09381 | standard_ep1000 |
| NOx Total (mg) | 4.498e+04 | 5.716e+04 | 3.966e+04 ★ | 4.052e+04 | standard_ep1000 |
| Fairness Variance | 0.7791 ★ | 168.9 | 17.54 | 67.36 | roundabout_fixed |

---
## Traffic Level: High

| Metric | roundabout_fixed | standard_fixed | standard_ep1000 | standard_ep2000 | Winner |
|--------|--------|--------|--------|--------|------|
| Trip Completion (%) | 97.31 | 95.86 | 97.83 ★ | 97.62 | standard_ep1000 |
| Cars NOT Completed | 26 | 40 | 21 ★ | 23 | standard_ep1000 |
| Cars Completed Trip | 940.0 | 926.0 | 945.0 ★ | 943.0 | standard_ep1000 |
| Cars Stuck in Network | 26 | 40 | 21 ★ | 23 | standard_ep1000 |
| Cars Never Inserted | 0 ★ | 0 | 0 | 0 | roundabout_fixed |
| Throughput (veh/hr) | 940.0 | 926.0 | 945.0 ★ | 943.0 | standard_ep1000 |
| Waiting Time (s) | 41.3 | 103.3 | 42.8 | 40.56 ★ | standard_ep2000 |
| Travel Time (s) | 125.2 | 169.0 | 105.9 | 103.1 ★ | standard_ep2000 |
| Time Loss (s) | 60 | 121.6 | 58.01 | 55.24 ★ | standard_ep2000 |
| Queue (slow veh/junction) | 2.368 | 5.784 | 2.421 | 2.291 ★ | standard_ep2000 |
| Network Speed (m/s) | 5.725 | 3.674 | 6.045 | 6.359 ★ | standard_ep2000 |
| Stops per Vehicle | 1.982 | 2.175 | 1.668 | 1.581 ★ | standard_ep2000 |
| CO₂ per Vehicle (mg) | 2.668e+05 | 3.296e+05 | 2.291e+05 | 2.241e+05 ★ | standard_ep2000 |
| Fuel per Vehicle (L) | 0.1166 | 0.144 | 0.1001 | 0.09792 ★ | standard_ep2000 |
| NOx Total (mg) | 8.71e+04 | 1.123e+05 | 7.726e+04 | 7.534e+04 ★ | standard_ep2000 |
| Fairness Variance | 12.43 ★ | 3176.4 | 114.3 | 65.2 | roundabout_fixed |

---
## Wins Tally (across all traffic levels)

| Controller | Metrics Won |
|-----------|------------|
| roundabout_fixed | 16 |
| standard_fixed | 0 |
| standard_ep1000 | 20 |
| standard_ep2000 | 12 |