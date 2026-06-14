# TraFix v6 — Test Suite

End-to-end evaluation framework that compares the **TraFix v6 AI controller**
against SUMO's built-in **fixed-timing traffic lights** across multiple traffic
conditions and scenario types.

---

## Directory Structure

```
tests/
├── config/
│   ├── seeds.yaml              # Fixed RNG seeds (all = 42)
│   └── traffic_levels.yaml     # Traffic load definitions
├── scenarios/
│   ├── generate_scenarios.py   # Run once to create type1/type2 route files
│   ├── type1_low.rou.xml       # ~30% capacity, uniform OFFPEAK
│   ├── type1_medium.rou.xml    # ~60% capacity, uniform OFFPEAK
│   ├── type1_high.rou.xml      # ~90% capacity, directional
│   ├── type2_morning_peak.rou.xml
│   ├── type2_evening_peak.rou.xml
│   ├── type2_incident.rou.xml
│   └── type2_pulse.rou.xml
├── runners/
│   ├── run_all.py              # Main orchestrator (Type 1 + Type 2 scenario)
│   └── run_checkpoint_compare.py  # 3-controller × 13 scenario comparison
├── metrics/                    # One module per metric
│   ├── travel_time.py
│   ├── emissions.py
│   ├── waiting_time.py
│   ├── queue_length.py
│   ├── throughput.py
│   ├── network_speed.py
│   ├── teleports.py
│   ├── time_loss.py
│   ├── fuel_consumption.py
│   ├── pollutants.py
│   ├── stops_per_vehicle.py
│   └── junction_fairness.py
├── analysis/
│   ├── compare.py              # CSV, markdown report, bar charts (Type 1)
│   └── checkpoint_compare.py   # 3-controller checkpoint comparison
├── outputs/                    # Created at runtime — raw SUMO XML per run
├── reports/                    # Created at runtime — results.csv, summary.md
│   └── charts/                 # PNG bar charts per metric
└── utils/
    ├── seeds.py                # Seed management
    ├── sumo_runner.py          # Core simulation runner (TraCI)
    └── fixtures/               # Sample SUMO output files for unit tests
```

---

## Quick Start

### 1. Prerequisites

- SUMO 1.26+ installed; `SUMO_HOME` environment variable set
- Project virtual environment activated
- TraFix v6 final checkpoint at `trafix_v6/checkpoints/trafix_v6_final.pt`

### 2. Generate route files (one time only)

```bash
python tests/scenarios/generate_scenarios.py
```

Route files are deterministic (no RNG). Re-running produces identical XML.

### 3. Run the full suite

```bash
python tests/runners/run_all.py
```

This takes ~30–90 minutes depending on hardware (14 simulation runs × up to
3 600 s each, plus 2 reproducibility-check runs).

### 4. Shorter runs

```bash
# Only Test Type 1 (traffic load)
python tests/runners/run_all.py --type1-only

# Only Test Type 2 (scenario robustness)
python tests/runners/run_all.py --type2-only

# Shorter simulation for quick validation
python tests/runners/run_all.py --sim-duration 600

# Reproducibility check only
python tests/runners/run_all.py --repro-check

# SUMO GUI (slow — for visual debugging of a single run)
python tests/runners/run_all.py --type1-only --gui
```

### 5. View results

| File | Contents |
|------|----------|
| `tests/reports/results.csv` | All metrics, all runs (long format) |
| `tests/reports/summary.md` | Comparison tables + percentage improvement |
| `tests/reports/charts/*.png` | Bar charts (baseline vs AI) per metric |

---

## Test Architecture

### Test Type 1 — Variable Traffic Load (same map, 3 load levels)

| Level | Total flow | Style |
|-------|-----------|-------|
| Low   | ~200 veh/hr | Uniform OFFPEAK |
| Medium | ~500 veh/hr | Uniform OFFPEAK |
| High  | ~950 veh/hr | Directional morning-peak |

For each level: **baseline** (SUMO fixed timing) vs **AI v6** (TraFix v6
GRU+GATConv actor-critic, 6 phases, RuleGovernor).

### Test Type 2 (scenarios) — Scenario Robustness (medium base load)

| Scenario | Description |
|----------|-------------|
| `morning_peak` | Heavy inbound commute (J0/J1/J2 → J3/J4), 700 veh/hr main flow |
| `evening_peak` | Heavy outbound commute (J3/J4 → J0/J1/J2), 700 veh/hr main flow |
| `incident` | J2 fringe blocked 300–600 s (road closure simulation) |
| `pulse` | Quiet → sudden 600 veh/hr inbound burst → quiet |

---

## Metrics

| # | Metric | Source |
|---|--------|--------|
| 1 | Average travel time | `tripinfo.xml` → `duration` |
| 2 | CO₂ emissions (total + per vehicle) | `tripinfo.xml` → `emissions.CO2_abs` |
| 3 | Average waiting time | `tripinfo.xml` → `waitingTime` |
| 4 | Average queue length / junction | `inline_metrics.json` (TraCI) |
| 5 | Throughput (veh/hr) | `tripinfo.xml` count / sim hours |
| 6 | Mean network speed | `summary.xml` → `meanSpeed` |
| 7 | Teleport count | `summary.xml` + `statistics.xml` |
| 8 | Time loss | `tripinfo.xml` → `timeLoss` |
| 9 | Fuel consumption | `tripinfo.xml` → `emissions.fuel_abs` (L) |
| 10 | Pollutants (NOx, PMx, HC, CO) | `tripinfo.xml` → `emissions.*_abs` |
| 11 | Stops per vehicle | `inline_metrics.json` (TraCI) |
| 12 | Junction fairness (waiting variance) | `inline_metrics.json` (TraCI) |

---

## Determinism / Reproducibility

All sources of randomness are eliminated:

| Layer | Mechanism |
|-------|-----------|
| SUMO | `--seed 42` passed to every `sumo` invocation |
| Python | `random.seed(42)` + `PYTHONHASHSEED=0` |
| NumPy | `numpy.random.seed(42)` |
| PyTorch | `torch.manual_seed(42)`, `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True` |
| Route files | Pre-generated, committed — never regenerated at test time |
| AI inference | `model.eval()`, no dropout, greedy governor mask |

The `--repro-check` flag runs the medium-baseline scenario twice and asserts
`SHA-256(tripinfo.xml + inline_metrics.json)` is identical.

---

## Unit-testing Metric Modules

Each metric module is independently testable with the sample fixtures:

```bash
# From project root:
python tests/metrics/travel_time.py
python tests/metrics/emissions.py
python tests/metrics/waiting_time.py
python tests/metrics/throughput.py
python tests/metrics/network_speed.py
python tests/metrics/teleports.py
python tests/metrics/time_loss.py
python tests/metrics/fuel_consumption.py
python tests/metrics/pollutants.py
python tests/metrics/stops_per_vehicle.py
python tests/metrics/junction_fairness.py
python tests/metrics/queue_length.py
```

All should print `<metric>: OK`.

---

## Extending the Suite

- **New metric**: add `tests/metrics/<name>.py` with `compute_<name>(out_dir)`.
  Register it in `tests/analysis/compare.py → _extract_all_metrics()`.
- **New scenario**: add a generator function in `tests/scenarios/generate_scenarios.py`,
  add the route file name to `TYPE2_SCENARIOS` in `tests/runners/run_all.py`.
- **Different AI checkpoint**: pass `--checkpoint <path>` (or edit
  `_CKPT` in `run_all.py`).
