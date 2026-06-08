# TraFix v6 — Test Suite Results


## type1 | high

| Metric | Baseline | AI (v6) | Δ% |
|--------|----------|---------|-----|
| CO_total_mg | 8.32e+05 | 8.346e+05 | ✗ -0.3% |
| HC_total_mg | 5535 | 5556 | ✗ -0.4% |
| NOx_total_mg | 1.123e+05 | 7.534e+04 | ✓ +32.9% |
| PMx_total_mg | 1.32e+04 | 1.345e+04 | ✗ -1.9% |
| cars_not_completed | 40 | 23 | ✓ +42.5% |
| co2_mg_per_vehicle | 3.296e+05 | 2.241e+05 | ✓ +32.0% |
| co2_total_mg | 3.053e+08 | 2.113e+08 | ✓ +30.8% |
| completion_rate_pct | 95.86 | 97.62 | ✓ +1.8% |
| fairness_variance | 3176 | 65.2 | ✓ +97.9% |
| fuel_per_vehicle_L | 0.144 | 0.09792 | ✓ +32.0% |
| network_speed_ms | 3.674 | 6.359 | ✓ +73.1% |
| queue_length_vehicles | 5.784 | 2.291 | ✓ +60.4% |
| stops_per_vehicle | 2.175 | 1.581 | ✓ +27.3% |
| teleports | 0 | 0 | ✓ +0.0% |
| throughput_veh_hr | 926 | 943 | ✓ +1.8% |
| time_loss_s | 121.6 | 55.24 | ✓ +54.6% |
| travel_time_s | 169 | 103.1 | ✓ +39.0% |
| vehicles_still_running | 40 | 23 | ✓ +42.5% |
| waiting_time_s | 103.3 | 40.56 | ✓ +60.7% |

## type1 | low

| Metric | Baseline | AI (v6) | Δ% |
|--------|----------|---------|-----|
| CO_total_mg | 1.652e+05 | 1.635e+05 | ✓ +1.0% |
| HC_total_mg | 1099 | 1088 | ✓ +1.0% |
| NOx_total_mg | 2.146e+04 | 1.651e+04 | ✓ +23.1% |
| PMx_total_mg | 2637 | 2628 | ✓ +0.4% |
| cars_not_completed | 61 | 61 | ✓ +0.0% |
| co2_mg_per_vehicle | 2.875e+05 | 2.258e+05 | ✓ +21.4% |
| co2_total_mg | 5.836e+07 | 4.585e+07 | ✓ +21.4% |
| completion_rate_pct | 76.89 | 76.89 | ✓ -0.0% |
| fairness_variance | 22.47 | 12.47 | ✓ +44.5% |
| fuel_per_vehicle_L | 0.1256 | 0.09867 | ✓ +21.4% |
| network_speed_ms | 4.005 | 5.781 | ✓ +44.3% |
| queue_length_vehicles | 1.017 | 0.5681 | ✓ +44.1% |
| stops_per_vehicle | 1.602 | 1.346 | ✓ +16.0% |
| teleports | 0 | 0 | ✓ +0.0% |
| throughput_veh_hr | 203 | 203 | ✓ -0.0% |
| time_loss_s | 84.15 | 53.45 | ✓ +36.5% |
| travel_time_s | 121.7 | 91.48 | ✓ +24.8% |
| vehicles_still_running | 61 | 61 | ✓ +0.0% |
| waiting_time_s | 70.67 | 41.15 | ✓ +41.8% |

## type1 | medium

| Metric | Baseline | AI (v6) | Δ% |
|--------|----------|---------|-----|
| CO_total_mg | 4.369e+05 | 4.36e+05 | ✓ +0.2% |
| HC_total_mg | 2907 | 2902 | ✓ +0.2% |
| NOx_total_mg | 5.716e+04 | 4.052e+04 | ✓ +29.1% |
| PMx_total_mg | 6891 | 6939 | ✗ -0.7% |
| cars_not_completed | 3 | 0 | ✓ +100.0% |
| co2_mg_per_vehicle | 2.961e+05 | 2.147e+05 | ✓ +27.5% |
| co2_total_mg | 1.554e+08 | 1.134e+08 | ✓ +27.1% |
| completion_rate_pct | 99.43 | 100 | ✓ +0.6% |
| fairness_variance | 168.9 | 67.36 | ✓ +60.1% |
| fuel_per_vehicle_L | 0.1293 | 0.09381 | ✓ +27.5% |
| network_speed_ms | 3.602 | 6.169 | ✓ +71.2% |
| queue_length_vehicles | 2.796 | 1.263 | ✓ +54.8% |
| stops_per_vehicle | 1.919 | 1.479 | ✓ +22.9% |
| teleports | 0 | 0 | ✓ +0.0% |
| throughput_veh_hr | 525 | 528 | ✓ +0.6% |
| time_loss_s | 109 | 56.06 | ✓ +48.6% |
| travel_time_s | 153.3 | 100.5 | ✓ +34.5% |
| vehicles_still_running | 3 | 0 | ✓ +100.0% |
| waiting_time_s | 92.87 | 42.06 | ✓ +54.7% |

## Verdict

**Metrics where AI improved over baseline:**
- NOx_total_mg (type1/high)
- cars_not_completed (type1/high)
- co2_mg_per_vehicle (type1/high)
- co2_total_mg (type1/high)
- completion_rate_pct (type1/high)
- fairness_variance (type1/high)
- fuel_per_vehicle_L (type1/high)
- network_speed_ms (type1/high)
- queue_length_vehicles (type1/high)
- stops_per_vehicle (type1/high)
- throughput_veh_hr (type1/high)
- time_loss_s (type1/high)
- travel_time_s (type1/high)
- vehicles_still_running (type1/high)
- waiting_time_s (type1/high)
- CO_total_mg (type1/low)
- HC_total_mg (type1/low)
- NOx_total_mg (type1/low)
- PMx_total_mg (type1/low)
- co2_mg_per_vehicle (type1/low)

**Metrics where AI regressed vs baseline:**
- CO_total_mg (type1/high)
- HC_total_mg (type1/high)
- PMx_total_mg (type1/high)
- PMx_total_mg (type1/medium)

**Overall:** AI improved 46 metric-scenarios, regressed 4.  Net score: +42.  Recommend deploying AI controller.