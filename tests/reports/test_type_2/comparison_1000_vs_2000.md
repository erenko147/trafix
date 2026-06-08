# Test Type 2 — AI ep1000 vs ep2000 (final)
Simulation: 3600 s

Δ% = % by which ep2000 (final) beats ep1000. Positive = final model improved.

---
## Low traffic

| Metric | standard_ep1000 (ref) | standard_ep2000 | Δ% |
|--------|----------| --- | --- |
| Trip Completion (%) | 77.27 | 76.89 | ✗ -0.5% |
| Cars NOT Completed | 60 | 61 | ✗ -1.7% |
| Cars Completed Trip | 204.0 | 203.0 | ✗ -0.5% |
| Cars Stuck in Network | 60 | 61 | ✗ -1.7% |
| Cars Never Inserted | 0 | 0 | ✓ +0.0% |
| Throughput (veh/hr) | 204.0 | 203.0 | ✗ -0.5% |
| Waiting Time (s) | 39.22 | 41.15 | ✗ -4.9% |
| Travel Time (s) | 90.14 | 91.48 | ✗ -1.5% |
| Time Loss (s) | 52.01 | 53.45 | ✗ -2.8% |
| Queue (slow veh/junction) | 0.5434 | 0.5681 | ✗ -4.5% |
| Network Speed (m/s) | 5.348 | 5.781 | ✓ +8.1% |
| Stops per Vehicle | 1.358 | 1.346 | ✓ +0.9% |
| CO₂ per Vehicle (mg) | 2.227e+05 | 2.258e+05 | ✗ -1.4% |
| Fuel per Vehicle (L) | 0.09731 | 0.09867 | ✗ -1.4% |
| NOx Total (mg) | 1.631e+04 | 1.651e+04 | ✗ -1.2% |
| Fairness Variance | 11.77 | 12.47 | ✗ -5.9% |

---
## Medium traffic

| Metric | standard_ep1000 (ref) | standard_ep2000 | Δ% |
|--------|----------| --- | --- |
| Trip Completion (%) | 100.0 | 100.0 | ✓ -0.0% |
| Cars NOT Completed | 0 | 0 | ✓ +0.0% |
| Cars Completed Trip | 528.0 | 528.0 | ✓ -0.0% |
| Cars Stuck in Network | 0 | 0 | ✓ +0.0% |
| Cars Never Inserted | 0 | 0 | ✓ +0.0% |
| Throughput (veh/hr) | 528.0 | 528.0 | ✓ -0.0% |
| Waiting Time (s) | 39.46 | 42.06 | ✗ -6.6% |
| Travel Time (s) | 97.88 | 100.5 | ✗ -2.7% |
| Time Loss (s) | 53.46 | 56.06 | ✗ -4.9% |
| Queue (slow veh/junction) | 1.185 | 1.263 | ✗ -6.6% |
| Network Speed (m/s) | 5.314 | 6.169 | ✓ +16.1% |
| Stops per Vehicle | 1.475 | 1.479 | ✗ -0.3% |
| CO₂ per Vehicle (mg) | 2.106e+05 | 2.147e+05 | ✗ -2.0% |
| Fuel per Vehicle (L) | 0.09201 | 0.09381 | ✗ -2.0% |
| NOx Total (mg) | 3.966e+04 | 4.052e+04 | ✗ -2.2% |
| Fairness Variance | 17.54 | 67.36 | ✗ -284.0% |

---
## High traffic

| Metric | standard_ep1000 (ref) | standard_ep2000 | Δ% |
|--------|----------| --- | --- |
| Trip Completion (%) | 97.83 | 97.62 | ✗ -0.2% |
| Cars NOT Completed | 21 | 23 | ✗ -9.5% |
| Cars Completed Trip | 945.0 | 943.0 | ✗ -0.2% |
| Cars Stuck in Network | 21 | 23 | ✗ -9.5% |
| Cars Never Inserted | 0 | 0 | ✓ +0.0% |
| Throughput (veh/hr) | 945.0 | 943.0 | ✗ -0.2% |
| Waiting Time (s) | 42.8 | 40.56 | ✓ +5.2% |
| Travel Time (s) | 105.9 | 103.1 | ✓ +2.7% |
| Time Loss (s) | 58.01 | 55.24 | ✓ +4.8% |
| Queue (slow veh/junction) | 2.421 | 2.291 | ✓ +5.4% |
| Network Speed (m/s) | 6.045 | 6.359 | ✓ +5.2% |
| Stops per Vehicle | 1.668 | 1.581 | ✓ +5.3% |
| CO₂ per Vehicle (mg) | 2.291e+05 | 2.241e+05 | ✓ +2.2% |
| Fuel per Vehicle (L) | 0.1001 | 0.09792 | ✓ +2.2% |
| NOx Total (mg) | 7.726e+04 | 7.534e+04 | ✓ +2.5% |
| Fairness Variance | 114.3 | 65.2 | ✓ +43.0% |
