# system_explained.md — How Every Part of TraFix Works and Connects

> **Audience:** code review / oral defense. This is the *system* companion to
> [`master_ai.md`](master_ai.md). Where `master_ai.md` answers **"what is the model and
> why,"** this file answers **"what is doing what, where, and how does the data actually
> flow"** — backend, the AI inference bridge, how live values are pulled out of a running
> SUMO, how the AI's answers are actuated back into SUMO and reflected on the dashboard,
> how the database persists everything, and how the test suite drives all of it.
> Everything below was read directly from the source on `main`.
>
> **One-paragraph summary.** Two processes are launched by `baslat.py`: a **FastAPI
> backend** (`uvicorn main:app`, port 8000) that owns the AI model, and a **live SUMO
> runner** (`sumo/run_sumo_live.py`) that owns the simulation. Every simulation step the
> runner reads per-lane vehicle counts out of SUMO via TraCI; every 10 steps it POSTs a
> telemetry batch to the backend; the backend runs the **TraFix v6** model + **RuleGovernor**
> and returns one phase per junction; the runner actuates those phases back into SUMO
> through yellow transitions, with four heuristic starvation/fallback safety overrides and
> an independent emergency-vehicle preemption layer on top. The backend also stamps the raw
> telemetry into a global `state_dict`, which the **browser dashboard** polls once a second
> via `GET /state`. Completed AI decisions and emergency events are persisted to
> **PostgreSQL** on background threads. The **test suite** replaces the live runner with a
> deterministic, seeded TraCI runner that drives the same model and measures 12 metrics
> against a SUMO fixed-timing baseline.

---

## Table of Contents

1. [The big picture — two processes, one loop](#1-the-big-picture--two-processes-one-loop)
2. [Process startup: what `baslat.py` actually does](#2-process-startup-what-baslatpy-actually-does)
3. [Reading values out of a running SUMO (the live runner)](#3-reading-values-out-of-a-running-sumo-the-live-runner)
4. [The telemetry contract (runner → backend)](#4-the-telemetry-contract-runner--backend)
5. [The backend: telemetry → AI decision](#5-the-backend-telemetry--ai-decision)
6. [Reflecting the AI's answer back into SUMO](#6-reflecting-the-ais-answer-back-into-sumo)
7. [The four safety overrides and the fallback](#7-the-four-safety-overrides-and-the-fallback)
8. [Emergency-vehicle preemption (the decoupled layer)](#8-emergency-vehicle-preemption-the-decoupled-layer)
9. [How the frontend is connected](#9-how-the-frontend-is-connected)
10. [The database layer (persistence)](#10-the-database-layer-persistence)
11. [How the tests are built and how they work](#11-how-the-tests-are-built-and-how-they-work)
12. [End-to-end trace of one decision](#12-end-to-end-trace-of-one-decision)
13. [Who-does-what file map](#13-who-does-what-file-map)
14. [Defense crib: "where does X happen?"](#14-defense-crib-where-does-x-happen)

---

## 1. The big picture — two processes, one loop

There are **two independent OS processes** that talk over HTTP on `127.0.0.1:8000`:

```
                          ┌─────────────────────────────────────────────┐
   PROCESS A              │  FastAPI backend  (uvicorn main:app :8000)  │
   the "brain"            │   • owns TraFixV6 model + RuleGovernor       │
                          │   • POST /telemetry_batch  → decisions       │
                          │   • GET  /state            → raw telemetry   │
                          │   • POST /emergency_event  → metrics         │
                          │   • writes to PostgreSQL (background thread) │
                          └───────────────▲───────────────┬─────────────┘
                                          │ HTTP          │ HTTP
                       telemetry batch    │               │   decisions
                       (every 10 steps)   │               ▼
                          ┌───────────────┴─────────────────────────────┐
   PROCESS B              │  sumo/run_sumo_live.py  (drives SUMO/TraCI)  │
   the "world + hands"    │   • reads lane counts from SUMO each step    │
                          │   • applies returned phases (yellow→green)   │
                          │   • starvation/fallback overrides            │
                          │   • emergency preemption (separate module)   │
                          └───────────────┬─────────────────────────────┘
                                          │ TraCI
                                   ┌──────▼──────┐
                                   │    SUMO     │  (sumo-gui or headless)
                                   │  5 junctions│
                                   └─────────────┘

   PROCESS C (any browser)  ──GET /state every 1s──►  backend  (read-only dashboard)
```

**Key separation of concerns:**

- The **backend never touches SUMO.** It only knows about 20-float observation vectors and
  returns phase numbers. This is what makes it swappable and testable without a simulator.
- The **runner never runs the model.** It only samples SUMO, calls the API, and actuates
  lights. This is what lets the same model serve the live demo and the test harness
  unchanged.
- The **dashboard never makes decisions.** It is a pure read-only view of `state_dict`.

---

## 2. Process startup: what `baslat.py` actually does

`baslat.py` is the one-click launcher. Steps (`baslat.py:66`–`87`):

1. Parse `--model {v2,v5,v6}` (default **v6**), `--port` (default 8000), `--no-gui`, and an
   optional `.sumocfg` path.
2. Start **Process A** in a daemon thread (`run_fastapi`): runs
   `python -m uvicorn main:app --host 127.0.0.1 --port 8000`, with the env var
   **`TRAFIX_MODEL_VERSION`** set from `--model`. That env var is the single switch the
   backend reads to decide which model class/weights to load.
3. `time.sleep(3)` — give the backend time to bind the port and load weights **before** SUMO
   starts POSTing (otherwise the first batches would fail and trip the fallback counter).
4. Start **Process B** in the foreground (`run_sumo`): runs `sumo/run_sumo_live.py`, passing
   `TRAFIX_API_PORT` so the runner knows where to POST, plus the optional sumocfg and
   `--no-gui`. Because SUMO is foreground, closing it tears down everything.

`main.py` (root) is a thin **bridge**: it imports `app` from `backend.main` and adds three
HTML routes — `GET /` → `dashboard.html`, `GET /architecture` → `index.html`,
`GET /emergency` → `emergency.html`. All the API endpoints live in `backend/main.py`.

---

## 3. Reading values out of a running SUMO (the live runner)

File: `sumo/run_sumo_live.py`. This is where **raw values are pulled out of the live
simulation**. The simulation loop (`run_sumo_live.py:224`) runs once per simulated second:

```
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()          # advance SUMO 1 step (= 1 sim-second)
    step += 1
    advance any pending yellow transitions
    run emergency preemption for every junction
    if step % DECISION_INTERVAL (=10) != 0: continue   # only act every 10 steps
    ... build telemetry, POST, apply decisions ...
```

### How each junction's 20 raw inputs are measured

`collect_lane_obs(tls_id, jx, jy)` (`run_sumo_live.py:96`) builds the 12 per-lane counts:

1. `traci.trafficlight.getControlledLinks(tls_id)` returns every signal-controlled
   connection at the junction.
2. For each link it takes the **incoming lane** (`link[0][0]`), dedupes (a lane can appear in
   several links), and splits the lane id into `edge_id` + `lane_idx`.
3. **Movement type** comes from the lane index via `LANE_TYPE = {0:right, 1:through, 2:left}`
   — i.e. the rightmost lane is the right-turn lane, etc. (this is how the 3-lane map was
   built).
4. **Compass direction** comes from geometry: `_classify_edge_direction` compares the lane's
   start point to the junction position `(jx, jy)` — bigger horizontal delta ⇒ east/west,
   bigger vertical ⇒ north/south.
5. `direction + "_" + movement` (e.g. `north_through`) is the dict key; the count is
   `traci.lane.getLastStepVehicleNumber(lane)` — the number of vehicles on that lane in the
   last step.

Then per junction the runner derives the other fields (`run_sumo_live.py:269`–`296`):

- `queue_length = min(total_queue * 1.5, 200.0)` — total of all 12 lane counts, scaled and
  saturated at 200 (same formula the env and tests use, so the model sees a consistent
  distribution).
- `current_phase` = `SUMO_TO_MODEL.get(traci.trafficlight.getPhase(tls_id))` — SUMO's 12-phase
  index folded down to the model's 6-phase space.
- `phase_duration = step - last_phase_change_step[tls_id]` — **manually tracked**, not read
  from SUMO. This is critical: calling `setPhase()` every decision resets SUMO's own phase
  timer, so SUMO would always report ~10 s. `last_phase_change_step` is updated only on a
  *real* switch, so this gives the true hold time the governor and reward depend on.

> **Defense point.** "Where do the AI's inputs come from?" → `getLastStepVehicleNumber` per
> controlled lane, classified by lane index (movement) and geometry (direction), plus a
> manually-tracked hold time. Nothing is faked or precomputed.

---

## 4. The telemetry contract (runner → backend)

Every 10th step the runner assembles one payload (`run_sumo_live.py:267`):

```json
{
  "step": 1234,
  "intersections": [
    {"intersection_id": 0,
     "north_left": 2, "north_through": 7, "north_right": 1,
     "south_left": 1, "south_through": 6, "south_right": 0,
     "east_left": 0,  "east_through": 3,  "east_right": 1,
     "west_left": 1,  "west_through": 4,  "west_right": 0,
     "queue_length": 48.0, "current_phase": 0, "phase_duration": 30.0},
    ... 4 more junctions ...
  ]
}
```

It POSTs this to `http://127.0.0.1:8000/telemetry_batch` with a **0.5 s timeout**
(`run_sumo_live.py:315`). The tight timeout matters: if the backend stalls, the runner must
fail fast and fall back rather than freeze the simulation. The response is
`{"decisions": [...]}`, one decision per junction.

The Pydantic schema on the backend side is `Telemetry` / `TelemetryBatch`
(`backend/main.py:73`, `:278`). It also accepts **legacy `*_count` fields** for v2 backward
compatibility — if only `north_count` etc. are present they're routed into the `*_through`
lanes (`backend/main.py:302`).

---

## 5. The backend: telemetry → AI decision

File: `backend/main.py`, endpoint `POST /telemetry_batch` (`:285`). Full sequence:

1. **Model selection at import time** (`:24`–`53`). The module reads `TRAFIX_MODEL_VERSION`
   once. For `v6` it imports `TraFixV6` + `RuleGovernor` and sets the weight path to
   `trafix_v6/checkpoints/trafix_v6_final.pt`. `load_model()` (`:130`) instantiates the model,
   loads the checkpoint's `model_state_dict`, calls `.eval()`, and builds the production
   `RuleGovernor` (min-green 10 s, max-green 90 s, flicker_penalty 3.0, pressure_thresh 0.12).
   If weights are missing or mismatched, `ai_agent` stays `None` and the **heuristic fallback**
   takes over.
2. **Restart detection** (`:290`). If `batch.step < _last_batch_step`, a new SUMO run has
   restarted the step counter, so the rolling window is cleared and the governor `reset()` —
   otherwise stale temporal history would poison the first decisions of the new run.
3. **Stash raw telemetry** into the global `state_dict` keyed by junction id (`:299`–`307`).
   This is exactly what the dashboard later reads via `GET /state`.
4. **Build the observation** (`:313`). `_build_obs_list()` returns a 5-element list (zero-filled
   for any missing junction), then `parse_sumo_observations(obs_list)` converts it to the
   `[5, 20]` float tensor (the shared parser documented in `master_ai.md §2`).
5. **Rolling temporal window** (`:321`). A `deque(maxlen=30)`. On the very first call it is
   **pre-filled with 30 copies** of the current frame so the GRU always sees a full `T=30`
   window; afterwards one frame is appended per call. `window_tensor` is `[1, 30, 5, 20]`.
6. **Forward pass** (`:328`). `logits_list, _ = ai_agent(window_tensor)` → 5 raw logit tensors.
7. **Govern** (`:332`). `_v6_governor.apply(logits_list, obs_last)` applies min/max-green
   masking, pressure boost, and stateful anti-flicker, using the **last** frame of the window
   to read each junction's current phase/duration.
8. **Softmax → argmax** (`:334`, `:365`). Per junction, `softmax(governed_logits)`; the chosen
   phase is the `argmax`; **confidence is the max softmax probability** (rounded to 3 dp).
   Deployment is deterministic — it does not sample.
9. **Update flicker state** (`:387`) with the chosen phases, but only if all 5 junctions are
   present (guards the stateful anti-flicker rule).
10. **Persist** (`:393`). If a DB session exists, `db.log_step` is queued on a
    **background task** so the HTTP response is not blocked by the database.
11. **Return** the decisions list: `intersection_id, next_phase, confidence, total_vehicles,
    queue_length`.

### Heuristic fallback (no model loaded)

If `ai_agent is None` (`:400`): per junction, **hold** if `phase_duration < 10` or there is
no demand; otherwise pick phase `0` (NS) vs `3` (EW) by whichever through-direction is busier.
This guarantees the lights never freeze even with no weights — and it is itself unit-tested
(see NFR-02).

---

## 6. Reflecting the AI's answer back into SUMO

Back in the runner, the `decisions` list is applied per junction (`run_sumo_live.py:324`–`491`):

1. If emergency preemption is active for this junction, **skip the AI** (`:331`).
2. `model_phase = decision["next_phase"] % 6`.
3. Run the **four override checks** (§7) which may replace `model_phase`.
4. Map to SUMO with `MODEL_TO_SUMO_GREEN[model_phase]` (green phases are even: 0,2,4,6,8,10).
5. **If the target equals the current green**, just re-assert it and continue (no transition).
6. **Otherwise enforce min-green**: through phases (0/3) need 10 s held, left phases need 8 s;
   if not yet elapsed, skip the switch this cycle.
7. **Insert a yellow**: set SUMO to `current_green + 1` (the yellow), record
   `pending_targets[tls_id] = target_green` and `yellow_remaining[tls_id] = YELLOW_STEPS (3)`,
   and stamp `last_phase_change_step[tls_id] = step`.
8. On subsequent steps, the top of the loop (`:229`) counts the yellow down; when it hits 0 it
   calls `setPhase(tls_id, pending_target)` — the actual green switch. **Yellows are never
   interrupted.**

So the AI's answer is not applied instantly as a hard jump — it is always mediated by the
min-green floor and a mandatory 3-step yellow, exactly mirroring how the training environment
and the test runner actuate phases. This is why the live behavior matches training.

### How the answer becomes *visible*

The actuated phase changes the SUMO state. On the **next** decision cycle, `collect_lane_obs`
+ `getPhase` read the new reality, the runner POSTs it, the backend stores it in `state_dict`,
and the dashboard's next 1 s poll shows the new `current_phase`/arrow and updated counts. So
the dashboard reflects the AI's decision **one telemetry cycle after** it is actuated — it
shows the *consequence* of decisions (the live world state), not the decision message itself.

---

## 7. The four safety overrides and the fallback

A learned policy can starve a movement, so the runner layers heuristics **on top of** the
governed AI output (`run_sumo_live.py`). All counters are in "decisions" (× 10 sim-seconds):

| Override | Constant | Where | Trigger & action |
|----------|----------|-------|------------------|
| Through starvation | `STARVE_LIMIT = 8` | `:338` | 8 consecutive non-through (left-only) decisions ⇒ force the busier through phase (0 or 3) |
| Direction starvation | `DIRECTION_STARVE_LIMIT = 10` | `:363` | one through direction monopolizes 10 decisions while the other has waiting cars ⇒ force the starved through |
| Left-turn starvation | `LEFT_STARVE_LIMIT = 15` | `:405` | a left phase (1/2/4/5) with demand unserved for 15 decisions ⇒ force the most-starved one with demand |
| Fixed-time fallback | `FALLBACK_THRESHOLD = 3`, `FALLBACK_CYCLE = 40` | `:499` | backend unreachable 3 cycles in a row ⇒ cycle NS/EW on a 40-step timer so lights never freeze |

The fallback is the API-down safety net: `api_consecutive_failures` increments whenever the
POST fails or times out; at 3 it switches to deterministic NS↔EW cycling until the backend
returns. Loosening these constants lets the model's raw behavior show through; tightening them
lets the heuristics dominate.

> **Defense point.** "What if the AI starves a left turn / the API dies?" → Three
> independent layers: (1) the **reward function** includes a per-movement anti-starvation
> term (−0.20, share-based, active at any demand level) so the policy is *trained* to
> avoid starvation; (2) the **RuleGovernor** enforces max-green hard switches in the
> backend; (3) explicit named guards in the **runner** (STARVE/DIRECTION/LEFT overrides)
> catch anything that slips through. API-down is handled separately by the fixed-time
> fallback after 3 consecutive failures.

---

## 8. Emergency-vehicle preemption (the decoupled layer)

File: `sumo/emergency_preemption.py`, class `EmergencyPreemptionController`. It is deliberately
**separate from the sim loop** (the comments note this is so the detection/actuation layer can
later be swapped for CARLA+YOLO without touching the loop). The loop just calls
`preempt.update(tls_id, step, ...)` for every junction every step (`run_sumo_live.py:245`).

It is a **per-junction state machine**: `None → yellow → allred → green → return_yellow → None`
(`emergency_preemption.py:351`). The three layers it isolates:

- **① Detection** (`_closest_emergency`): scans approach lanes for vehicles whose
  `getTypeID == "emergency"`, picks the one closest to the junction. (CARLA equivalent:
  camera → YOLO "ambulance" box.)
- **② Decision** (the state machine): switches the approach edge to all-green, holds all other
  approaches red, returns control when the vehicle clears or after `MAX_EMERGENCY_GREEN = 60`
  steps. This core is portable and unchanged across backends.
- **③ Actuation** (`_set_state` / `_restore_program`): drives
  `setRedYellowGreenState` / `setProgram`. **Critical detail** (`_restore_program`): because
  `setRedYellowGreenState` puts the light into an online single-phase program, it must restore
  program `"0"` afterwards or the AI's later `setPhase` calls would be ignored and the junction
  would freeze.

While a junction is preempted, `is_active()` returns `True`, which is exactly the signal the
runner checks to **skip both the AI decision and the fallback** for that junction.

It also **collects two metrics** per event: `transit_steps` (≈ seconds for the emergency
vehicle to clear) and `vehicles_waited` / `total_wait_steps` (how many other vehicles waited,
and for how long). Completed events are popped via `pop_completed_sessions()` and POSTed by the
runner to `POST /emergency_event` (`run_sumo_live.py:255`), where the backend buffers them in
`emergency_events` (a `deque(maxlen=200)`) and persists each via `db.log_emergency`.

---

## 9. How the frontend is connected

Three static HTML pages, served by the root `main.py` bridge. They are **read-only**; they
never POST and never call the model.

### `dashboard.html` — the live operations view

- **Polls `GET /state` every 1000 ms** (`dashboard.html:748`–`764`) using `fetch`, aborting any
  in-flight request before the next. `/state` returns the backend's raw `state_dict` (the
  telemetry the runner most recently POSTed).
- Each poll runs three render passes: `updateSummary` (totals, busiest junction, average queue,
  and a **YZ Karar Sayısı / "AI decision count"** that increments whenever a junction's
  `current_phase` changes between polls), `updateMap` (the 5-node SVG colored by queue level),
  and `renderPanel` (per-junction cards or a focused single-junction cross diagram).
- It computes per-direction totals with `dirCount` (`:405`), which sums the per-lane
  `*_left/_through/_right` fields and falls back to legacy `*_count` — this is the frontend
  mirror of the backend's backward-compat handling.
- **How the AI's answer shows up here:** the green-highlighted directions come from
  `GREEN_DIRS[d.current_phase]` (`:364`). Since `current_phase` is whatever the runner last
  actuated and reported, the dashboard's green arrows are the AI's decisions as realized in the
  world, one cycle delayed. The "AI decision count" is a proxy for how often phases changed.
- The node coordinates are hard-coded to match `sumo/map.net.xml` (a 2×2 grid plus K5 on the
  right); the chain `0–1–2–3–4` connectivity in `CONNECTIONS` matches the model's graph
  topology.

### `emergency.html` — emergency metrics view

Polls `GET /emergency_metrics`, which returns the buffered events plus a summary (count, average
transit steps, total vehicles waited). This is the human-readable view of the preemption metrics
described in §8.

### `index.html` — architecture/explainer page

Static documentation of the system (served at `/architecture`).

---

## 10. The database layer (persistence)

File: `backend/database.py`. PostgreSQL via `psycopg2` with a small threaded connection pool.

- **Connection is optional and non-fatal.** On startup (`backend/main.py:238`) `init_db` tries
  to connect using `DATABASE_URL` (default `dbname=trafix`). If PostgreSQL is unreachable it
  logs a warning, sets `_pool = None`, and the whole system keeps running **without
  persistence**. Every write helper early-returns when `_pool is None` — this is the FR-04
  "fails gracefully without a live DB" guarantee, and it is unit-tested.
- **Schema** (`_DDL`, `:33`): three tables —
  - `sessions` (one row per backend run, with `started_at`/`ended_at`),
  - `step_log` (one row **per junction per decision**: sim_step, queue, current_phase,
    next_phase, confidence, total_vehicles),
  - `emergency_log` (one row per completed preemption event with the transit/wait metrics).
- **Writes happen off the request thread.** `/telemetry_batch` queues `db.log_step` as a
  FastAPI `BackgroundTask`, so DB latency never slows the decision response. `log_step`
  merges the decisions list with the telemetry list (keyed by junction id) and `executemany`s
  the rows.
- **Reads** are for verification/dashboards: `query_session_summary` (`:256`) returns row
  counts and average queue for a session, exposed at `GET /db_summary`.

---

## 11. How the tests are built and how they work

There are **two distinct test layers**.

### 11a. Mandatory unit/FR/NFR tests — fast, no SUMO

`tests/mandatory/` — **121 pytest tests**, run with `python -m pytest tests/mandatory/ -v`.
They never launch SUMO; they import the real modules and assert on them. Seven files:

| File | What it pins down |
|------|-------------------|
| `test_unit_model.py` (22) | `TraFixV6` shapes: GRU `[B,5,128]`, GAT `[5,128]`, 5 logit tensors of `[B,6]`, value `[B,5]`, no NaN, softmax sums to 1, argmax ∈ [0,5], checkpoint exists and loads |
| `test_unit_rule_governor.py` (12) | min-green locks other phases before 10 s (through) / 8 s (left); max-green frees switch; anti-flicker penalizes A→B→A; pressure boosts the congested through; `apply()` returns 5 × `[1,6]` |
| `test_unit_observation.py` (15) | `parse_sumo_observations` → `[5,20]` float32; lane features are junction-relative shares (`count / max(total_12, 1)`); queue ÷200; duration ÷120 capped at 3.0; 6-bit phase one-hot; sorted by junction id; no NaN on zero/huge inputs |
| `test_unit_preemption.py` (13) | state machine starts inactive, activates within 1 step, cancels pending yellow, yellow→allred→green timing, approach edge gets `G` others `r`, metrics recorded |
| `test_nfr.py` (18) | **NFR-01** inference + governor < 1000 ms (single + p99 over 50 calls); **NFR-02** heuristic fallback validity; **NFR-05** required libs import and model runs on CPU |
| `test_fr.py` (22) | **FR-01** telemetry fields; **FR-02** valid phase per junction + deterministic in eval; **FR-03** preemption overrides AI in 1 step; **FR-04** DB graceful without a live DB; **FR-06** yellow inserted between distinct greens, 3-step hold |
| `test_metrics_modules.py` (19) | each metric module computes sane values from the committed fixture XML in `tests/utils/fixtures/` |

These are the tests to run for a quick "is the system structurally sound" check; they are
deterministic and need no simulator.

### 11b. Evaluation suite — full SUMO runs, AI vs baseline

`tests/runners/` + `tests/utils/sumo_runner.py` + `tests/metrics/` + `tests/analysis/`.
This is the **performance** harness, documented in `tests/README.md`.

The heart is `tests/utils/sumo_runner.py::run_simulation`, which runs one SUMO simulation in
one of two modes:

- **`baseline`** — SUMO's built-in fixed-timing TLS programs; **no TraCI control** of the
  lights. This is the comparison floor.
- **`ai`** — loads `trafix_v6_final.pt` and drives every TLS, using a **near-identical copy of
  the live pipeline**: the same `_get_observations` lane-classification, the same 30-step
  rolling window, the same `governor.apply_stateless` + `sample_governed`, the same
  `MODEL_TO_SUMO_GREEN` mapping and 3-step yellow transitions (`sumo_runner.py:400`–`445`).

Both modes write SUMO's output XML (`tripinfo`, `summary`, `queue`, `statistics`) plus an
`inline_metrics.json` of things only TraCI can see (stops-per-vehicle, per-approach waiting and
halting, junction fairness variance). A **gridlock detector** flags a run if mean network speed
stays below 0.5 m/s for 300 consecutive seconds.

**Determinism is enforced everywhere** (`tests/README.md`): `--seed 42` to SUMO,
`PYTHONHASHSEED=0`, seeded `random`/`numpy`/`torch`, deterministic torch algorithms,
pre-generated committed route files, and `model.eval()` greedy inference. The `--repro-check`
flag runs the medium baseline twice and asserts the output hashes match.

The runners then orchestrate scenarios and the analysis modules turn the raw outputs into
`results.csv`, `summary.md`, and per-metric bar charts:

- `run_all.py` — **Type 1** (low/medium/high traffic load) + **Type 2** (morning/evening peak,
  incident, pulse) — baseline vs AI on the standard map.
- `run_checkpoint_compare.py` — 3 controllers × 13 scenarios (e.g. fixed vs ep1000 vs ep2000).
- `run_test_type_2.py` — roundabout (Webster 86 s fixed) vs standard intersection.

The 12 metrics (travel time, CO₂, waiting time, queue, throughput, network speed, teleports,
time loss, fuel, pollutants, stops/vehicle, fairness) each live in their own
`tests/metrics/<name>.py` and are independently runnable.

> **Defense point.** "Are the tests honest?" → the `ai`-mode test runner is a faithful copy of
> the live actuation path (same observation builder, window, governor, mapping, yellows), and
> the baseline uses SUMO's *own* fixed timing with no TraCI help, on identical seeded demand.
> So any improvement is attributable to the controller, not to a measurement difference.

---

## 12. End-to-end trace of one decision

Putting it all together — one decision interval, step 1230 → 1240:

1. **SUMO** advances 10 steps. Cars move; queues change.
2. **Runner** (`run_sumo_live.py`) at step 1240 calls `collect_lane_obs` for each of the 5
   junctions → 12 counts each; computes `queue_length`, folds `getPhase` to a model phase,
   computes `phase_duration` from `last_phase_change_step`.
3. **Runner** POSTs the `{step:1240, intersections:[…5…]}` batch to `/telemetry_batch`.
4. **Backend** stores it in `state_dict`, parses → `[5,20]`, appends to the 30-frame window,
   runs `TraFixV6` → 5 logit tensors, applies the `RuleGovernor`, softmax+argmax → e.g.
   `[0, 3, 0, 3, 0]` with confidences, queues a `log_step` DB write, returns the decisions.
5. **Runner** receives the decisions. For each junction: applies starvation/direction/left
   overrides; if a phase change is warranted and min-green elapsed, sets the SUMO yellow,
   queues the target green, and starts a 3-step yellow countdown.
6. **Over the next 3 steps**, the loop counts the yellow down and calls `setPhase(target)` —
   the green actually changes.
7. **Meanwhile**, every junction's `preempt.update` ran each step; if an ambulance had
   appeared, that junction's AI decision in step 5 would have been skipped.
8. **Browser** polls `GET /state` ~1 s later, sees the new `current_phase`/counts, and repaints
   the cards, map colors, green arrows, and the AI-decision counter.
9. **PostgreSQL** (if up) now has 5 new `step_log` rows for sim_step 1240.

---

## 13. Who-does-what file map

```
LAUNCH
  baslat.py                 starts backend (Process A) + live runner (Process B)
  main.py                   bridge: mounts backend.app + serves the 3 HTML pages
  run.py                    bare uvicorn launcher (alt entry)

BACKEND  (the brain — never touches SUMO)
  backend/main.py           model selection (TRAFIX_MODEL_VERSION); load_model();
                            POST /telemetry_batch (window→model→governor→argmax);
                            GET /state, /last_decisions, /db_summary;
                            POST /emergency_event, GET /emergency_metrics; heuristic fallback
  backend/database.py       PostgreSQL pool; sessions/step_log/emergency_log; graceful no-DB
  backend/ai/trafix_v2.py   parse_sumo_observations (shared 20-dim parser) + reward/GAE (training)

MODEL  (see master_ai.md for internals)
  trafix_v6/trafix_v6.py        TraFixV6 (GRU→GAT→trunk→5 actors + hybrid critic)
  trafix_v6/rule_governor.py    RuleGovernor (min/max-green, pressure, anti-flicker)
  trafix_v6/checkpoints/        trafix_v6_final.pt (production weights)

LIVE SIMULATION  (the world + hands — never runs the model)
  sumo/run_sumo_live.py     sim loop: read lanes → POST telemetry → actuate phases (yellows);
                            STARVE/DIRECTION/LEFT overrides; fixed-time fallback
  sumo/emergency_preemption.py  EmergencyPreemptionController (decoupled state machine + metrics)
  sumo/map.net.xml          5-junction, 3-lane network; 5 tlLogic × 12 phases
  sumo/demo.sumocfg         live demo config

FRONTEND  (read-only views)
  frontend/dashboard.html   polls GET /state every 1s; cards + SVG map + AI-decision counter
  frontend/emergency.html   polls GET /emergency_metrics
  frontend/index.html       static architecture page (/architecture)

TESTS
  tests/mandatory/          121 fast pytest tests (no SUMO): unit + FR + NFR + metrics
  tests/utils/sumo_runner.py   the seeded SUMO runner (baseline vs ai modes)
  tests/runners/            run_all, run_checkpoint_compare, run_test_type_2
  tests/metrics/            12 metric modules
  tests/analysis/           CSV / markdown / chart generation
  tests/scenarios/          committed deterministic route files
```

---

## 14. Defense crib: "where does X happen?"

| Question | Answer (file:where) |
|----------|---------------------|
| Where are the AI's inputs measured? | `run_sumo_live.py:collect_lane_obs` — `getLastStepVehicleNumber` per controlled lane, classified by lane index (movement) and geometry (direction) |
| Why is phase_duration tracked manually? | `setPhase()` resets SUMO's timer every decision; `last_phase_change_step` (`run_sumo_live.py:286`) gives the true hold time |
| Where does the model actually run? | `backend/main.py:328` (`ai_agent(window_tensor)`), inside `POST /telemetry_batch` only |
| Where is the temporal window built? | `backend/main.py:321` — `deque(maxlen=30)`, pre-filled with 30 copies on first call |
| Where are traffic-law constraints applied? | `_v6_governor.apply` (`backend/main.py:332`) — the RuleGovernor masks logits |
| How is the chosen phase decided? | softmax → argmax (`backend/main.py:365`); confidence = max softmax prob |
| How is the AI's answer put back into SUMO? | `run_sumo_live.py:454`–`491` — map to even green, enforce min-green, insert 3-step yellow, then `setPhase(target)` |
| What stops the model from starving a movement? | Three layers: (1) **reward starvation term** trains the policy to rotate movements (−0.20, share-based, active at any demand); (2) **governor max-green** forces a switch in the backend; (3) **STARVE/DIRECTION/LEFT overrides** in the runner as a final safety net |
| What happens if the backend is down? | runner fixed-time fallback after 3 failures (`run_sumo_live.py:499`) |
| How do emergency vehicles override the AI? | `EmergencyPreemptionController.is_active()` gates the AI/fallback per junction; state machine drives the lights |
| How does the dashboard get its data? | `GET /state` polled every 1 s; returns the raw `state_dict` the runner last POSTed |
| Are the AI's decisions reflected on the dashboard directly? | No — the dashboard shows the *world state* (`current_phase`) that results from actuated decisions, one telemetry cycle later |
| Where is everything persisted? | `backend/database.py` — `step_log` (per junction per decision) and `emergency_log`, on background threads, graceful if no DB |
| How are tests kept fair/deterministic? | seeded SUMO + libs, committed route files, `model.eval()` greedy, baseline uses SUMO fixed timing with no TraCI (`tests/README.md`, `tests/utils/sumo_runner.py`) |

---

*Companion to `master_ai.md`. All paths and line references read from source on branch `main`.
For model internals (architecture, reward, PPO, training stages) see `master_ai.md`; this file
covers the surrounding system: processes, data flow, SUMO I/O, frontend, database, and tests.*
