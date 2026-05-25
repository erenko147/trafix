# TraFix v6 — Mandatory Test List

Derived from system functional/non-functional requirements, product backlog, and project report.
Existing simulation-performance tests are already in `tests/` — this list covers everything else.

---

## 1. Unit Tests

### AI Core (GRU + GATConv + PPO)
- [ ] GRU temporal encoder: correct hidden-state shape for single and batched observations
- [ ] GATConv graph encoder: correct output shape; edges respected
- [ ] PPO actor output: valid probability distribution over phases (sums to 1, all ≥ 0)
- [ ] PPO critic output: scalar value per junction
- [ ] Reward function: correct sign, magnitude for congestion / throughput / fairness inputs
- [ ] Rule governor: legal-phase mask eliminates all forbidden phase transitions
- [ ] Observation parser: correct normalisation of queue, speed, wait-time signals

### SUMO / TraCI Interface
- [ ] Observation collector: correct lane-by-lane queue and speed extraction
- [ ] Phase setter: TraCI `setPhase()` call maps agent action to correct SUMO TLS index
- [ ] Inline metrics logger: `inline_metrics.json` written with all required keys

### Emergency Preemption
- [ ] Preemption controller overrides AI phase selection when flag is active
- [ ] Preemption releases control after configurable hold time
- [ ] Normal AI resumes from a valid phase after preemption ends

### Redis Streams Module
- [ ] Telemetry message serialised to correct schema before publish
- [ ] Consumer reads back identical payload (round-trip)
- [ ] Backpressure / stream trim does not drop entries within `maxlen`

### PostgreSQL Logging Module
- [ ] Episode row inserted with all required columns on episode start
- [ ] Step row inserted per simulation step with correct junction_id
- [ ] Indexes present; duplicate-key constraint rejected on replay

### CV Pipeline (CARLA + YOLOv8)
- [ ] YOLOv8 model loads without error on CPU (no GPU assumed in CI)
- [ ] Bounding-box output schema: class, confidence, xyxy present
- [ ] CARLA→SUMO coordinate translation: known pixel maps to expected road segment
- [ ] Vehicle count per frame matches ground-truth fixture for sample image

### Data Preprocessing
- [ ] Observation normalisation: all features clipped to [0, 1]
- [ ] Missing sensor value filled with configured default, not NaN

---

## 2. Integration Tests

### SUMO ↔ TraCI Loop
- [ ] Simulation starts, advances steps, closes cleanly via `traci.close()`
- [ ] Agent receives observation of correct shape every step
- [ ] Agent action applied; SUMO phase advances accordingly
- [ ] `tripinfo.xml` and `summary.xml` produced at end of run

### AI Inference Pipeline (end-to-end)
- [ ] Observation → GRU → GATConv → PPO → masked action → TraCI phase set (one full cycle)
- [ ] No NaN / Inf in any tensor during inference on standard inputs

### Redis Streams Ingestion
- [ ] Simulation step publishes telemetry; consumer receives it within 500 ms
- [ ] 1000 consecutive messages received with zero loss (in-process test)

### PostgreSQL Write / Read
- [ ] Simulate one episode; all step rows retrievable with correct episode_id FK

### Emergency Preemption Override Flow
- [ ] Preemption signal arrives mid-episode → AI paused → special phase set → AI resumed
- [ ] Metrics recorded during preemption window are flagged correctly

### Fallback Mechanism (AI timeout → fixed timing)
- [ ] If inference call blocks > 500 ms, system switches to fixed-timing fallback (NFR-02)
- [ ] Fixed-timing fallback produces valid SUMO phase output without crash
- [ ] AI resumes automatically once latency drops below threshold

### Full Pipeline Smoke Test
- [ ] SUMO + TraCI + AI + Redis + PostgreSQL run together for 60 simulated seconds without error
- [ ] Dashboard WebSocket receives at least one update per 5 s during run

---

## 3. Performance / NFR Tests

| ID | Requirement | Test |
|----|-------------|------|
| NFR-01 | AI inference latency ≤ 500 ms per step | Time 1000 `agent.act()` calls; assert p99 ≤ 500 ms |
| NFR-02 | Automatic fallback on latency breach | Inject artificial 600 ms delay; assert fallback activates |
| NFR-03 | Dashboard WebSocket update ≤ 1 s | Assert last frame age < 1 s while simulation runs |
| NFR-04 | Zero data loss in Redis under load | Publish 10 000 messages at sim speed; assert consumer count matches |
| NFR-05 | Cross-platform: runs on Linux and Windows | CI matrix: ubuntu-latest + windows-latest |

---

## 4. Functional Requirement Tests (FR-01 – FR-06)

| ID | Requirement | Test |
|----|-------------|------|
| FR-01 | Telemetry ingestion from TraCI | Assert telemetry dict contains queue, speed, wait-time for every junction each step |
| FR-02 | AI phase selection per junction | Assert phase index is in [0, NUM_PHASES) and changes are law-compliant |
| FR-03 | Emergency vehicle preemption | Inject emergency flag; assert dedicated phase set within 1 simulation step |
| FR-04 | Data logging to PostgreSQL | After 300 s run, assert row count matches step count |
| FR-05 | Dashboard visualises live data | Assert WebSocket payload includes junction_id, phase, queue for all 5 junctions |
| FR-06 | Yellow phase enforced between green phases | Assert no direct green→green transition in phase log |

---

## 5. System / End-to-End Tests

- [ ] **Full episode comparison**: AI vs fixed-timing over 3 600 s; assert AI travel time ≤ baseline (existing Type 1 suite)
- [ ] **OOD generalisation**: run on a map not seen during training; assert no crash and throughput > 0
- [ ] **Gridlock recovery**: inject total blockage at step 600; assert system detects gridlock within 300 s
- [ ] **Reproducibility**: two runs with identical seeds produce bit-identical `tripinfo.xml` and `inline_metrics.json` (existing `--repro-check`)
- [ ] **Checkpoint regression**: ep2000 mean travel time ≤ ep1000 mean travel time across all three load levels

---

## 6. Existing Simulation Performance Metrics (already in `tests/metrics/`)

All modules below have standalone self-tests (`python tests/metrics/<name>.py`):

| Module | Metric |
|--------|--------|
| `travel_time.py` | Mean trip duration (s) |
| `waiting_time.py` | Mean junction wait time (s) |
| `time_loss.py` | Mean time loss vs free-flow (s) |
| `queue_length.py` | Mean halting vehicles per junction |
| `throughput.py` | Vehicles completed per hour |
| `network_speed.py` | Mean network speed (m/s) |
| `teleports.py` | Total teleport count |
| `emissions.py` | Mean CO₂ per vehicle (mg) |
| `fuel_consumption.py` | Mean fuel per vehicle (L) |
| `pollutants.py` | Total NOx / PMx / HC / CO (mg) |
| `stops_per_vehicle.py` | Mean stops per vehicle |
| `junction_fairness.py` | Network-wide waiting-time variance |
