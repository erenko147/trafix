# code_reference.md — Functions & Structures Index

> **Audience:** code review / oral defense. This is the **symbol-level** companion to
> [`master_ai.md`](master_ai.md) (model & training) and [`system_explained.md`](system_explained.md)
> (data flow & integration). It is a flat index of **every class, function, dataclass, and
> module-level structure** that matters, with its **file and line number** and a one-line
> description of what it does. Use it to answer *"where is X and what does it do?"* All line
> numbers read directly from source on `main`; if you edit a file the numbers drift, so treat
> them as "near line N."
>
> Legend: `🏛 class` · `ƒ function` · `🅼 method` · `📦 dataclass/Enum` · `🔢 constant/structure`

---

## Table of Contents

1. [Model — `trafix_v6/trafix_v6.py`](#1-model--trafix_v6trafix_v6py)
2. [Governor — `trafix_v6/rule_governor.py`](#2-governor--trafix_v6rule_governorpy)
3. [Scenario generator — `trafix_v6/scenario_generator.py`](#3-scenario-generator--trafix_v6scenario_generatorpy)
4. [Shared obs / reward — `backend/ai/trafix_v2.py`](#4-shared-obs--reward--backendaitrafix_v2py)
5. [Training env & utils — `backend/ai/train_v2.py`](#5-training-env--utils--backendaitrain_v2py)
6. [Training stage scripts — `trafix_v6/stage*.py`, `finetune`, `eval`, `train_v6.py`](#6-training-stage-scripts)
7. [Backend API — `backend/main.py`](#7-backend-api--backendmainpy)
8. [Database — `backend/database.py`](#8-database--backenddatabasepy)
9. [Bridge — `main.py`](#9-bridge--mainpy)
10. [Live runner — `sumo/run_sumo_live.py`](#10-live-runner--sumorun_sumo_livepy)
11. [Emergency preemption — `sumo/emergency_preemption.py`](#11-emergency-preemption--sumoemergency_preemptionpy)
12. [Test runner — `tests/utils/sumo_runner.py`](#12-test-runner--testsutilssumo_runnerpy)
13. [Test orchestration, metrics, analysis](#13-test-orchestration-metrics-analysis)
14. [Module-level constants/structures cross-reference](#14-module-level-constantsstructures-cross-reference)

---

## 1. Model — `trafix_v6/trafix_v6.py`

| Sym | Name | Line | What it does |
|-----|------|-----:|--------------|
| 🔢 | `OBS_DIM=20`, `NUM_PHASES=6`, `NUM_JUNCTIONS=5` | 31–33 | Core dimensions; imported across backend/tests as the I/O contract |
| ƒ | `_make_chain_edge_index(n)` | 36 | Builds the bidirectional chain `0–1–2–3–4` edge index (registered as a buffer) |
| 🏛 | `_TemporalEncoder(nn.Module)` | 47 | GRU(20→128); the time encoder |
| 🅼 | `_TemporalEncoder.forward(obs)` | 56 | Reshapes `[B,T,J,20]`→`[B*J,T,20]`, runs GRU, returns final hidden `[B,J,128]` |
| 🏛 | `_GraphEncoder(nn.Module)` | 64 | GATConv(128→32, heads=4, concat ⇒128); the space encoder |
| 🅼 | `_GraphEncoder.forward(x, edge_index)` | 73 | Attends each junction over its chain neighbours |
| 🏛 | `_SharedTrunk(nn.Module)` | 77 | MLP 128→128→64 shared by actor & critic |
| 🅼 | `_SharedTrunk.forward(x)` | 85 | Two Linear+ReLU layers |
| 🏛 | `TraFixV6(nn.Module)` | 93 | **The production model.** Composes the 3 encoders + 5 actor heads + hybrid critic |
| 🔢 | `TraFixV6._GAT_IN=128` | 107 | Must equal GRU hidden dim |
| 🅼 | `TraFixV6.__init__(...)` | 109 | Wires temporal→graph→trunk→`actor_heads`(5×Linear64→6)+`local_critics`(5)+`global_critic` |
| 🅼 | `TraFixV6._encode(obs, edge_index)` | 144 | Runs temporal→graph→trunk, returns the 64-dim per-junction features |
| 🅼 | `TraFixV6._batch_edge_index(edge_index, B)` | 161 | Offsets the chain edges per graph for batches >1 |
| 🅼 | `TraFixV6.forward(obs, edge_index=None)` | 167 | → `(logits_list[5×[B,6]], value[B,J])`; the inference entry point |
| 🅼 | `TraFixV6.get_action(obs)` | 180 | Samples `Categorical(logits)`; used in training rollouts |
| 🅼 | `TraFixV6.evaluate_actions(obs, actions)` | 191 | → `(log_probs, entropy, value)` for the PPO update (no governor) |
| 🅼 | `TraFixV6.__repr__()` | 202 | Param summary string |

---

## 2. Governor — `trafix_v6/rule_governor.py`

| Sym | Name | Line | What it does |
|-----|------|-----:|--------------|
| 🔢 | `_IDX_*` (N_LEFT..DURATION) | 29–43 | Named indices into the 20-dim obs (`_IDX_PHASE=slice(13,19)`, `_IDX_DURATION=19`) |
| 🔢 | `_NORM_LEFT_RIGHT=15`, `_NORM_THROUGH=30`, `_NORM_DURATION=120`, `_NEG_INF=-1e9` | 45–49 | De-normalisers + the masking sentinel |
| 🔢 | `MIN_GREEN_THROUGH=10`, `MIN_GREEN_LEFT=8`, `MAX_GREEN_THROUGH=90`, `MAX_GREEN_LEFT=45` | 52–55 | Phase-type green bounds (override constructor args inside `_hard_mask`) |
| ƒ | `_decode_obs(obs_j)` | 58 | Reads `(phase, duration_seconds)` from one junction's obs (`argmax` one-hot, `dur×120`) |
| 🏛 | `RuleGovernor` | 67 | The constraint layer between logits and the phase choice |
| 🅼 | `RuleGovernor.__init__(...)` | 88 | Stores min/max-green, flicker window/penalty, pressure boost/thresh; inits flicker history |
| 🅼 | `RuleGovernor.reset()` | 114 | Clears flicker history (call per episode / SUMO restart) |
| 🅼 | `RuleGovernor.update_state(actions_1d)` | 118 | Pushes the chosen phases into the flicker history |
| 🅼 | `RuleGovernor._hard_mask(phase, duration)` | 125 | Min-green: forbid all-but-current; max-green: forbid current |
| 🅼 | `RuleGovernor._pressure_bonus(obs_j)` | 142 | Adds a bonus to the busiest movement's phase if it exceeds `pressure_thresh` |
| 🅼 | `RuleGovernor._flicker_penalty(j)` | 172 | Subtracts `flicker_penalty` from a back-to-A reversal |
| 🅼 | `RuleGovernor.apply(logits_list, obs_last)` | 184 | **Live/rollout path**: hard mask + pressure + stateful flicker |
| 🅼 | `RuleGovernor.apply_stateless(...)` | 202 | Hard mask + pressure only (no flicker); used by test runner |
| 🅼 | `RuleGovernor.apply_stateless_batch(...)` | 218 | Vectorised over a minibatch; used inside the PPO update |
| ƒ | `sample_governed(masked_logits)` | 242 | → `(actions, log_probs)` from governed logits (training/test sampling) |
| ƒ | `evaluate_governed(...)` | 257 | Re-evaluates governed log-probs/entropy/value for the PPO ratio |

---

## 3. Scenario generator — `trafix_v6/scenario_generator.py`

| Sym | Name | Line | What it does |
|-----|------|-----:|--------------|
| 🔢 | `_DEFAULT_NET_FILE`, `_DEFAULT_OUTPUT_DIR` | 45–46 | Net path + where `.rou.xml` are written |
| 📦 | `ScenarioType(Enum)` | 53 | OFFPEAK, MORNING_PEAK, EVENING_PEAK, INCIDENT, PULSE |
| 🔢 | `_SCENARIO_ORDER` | 112 | Curriculum weight ordering |
| 🏛 | `ScenarioGenerator` | 121 | Builds a fresh per-episode demand file so the agent never overfits |
| 🅼 | `.generate(scenario_type, episode)` | 144 | Writes one `.rou.xml` for the given type/episode, returns its path |
| 🅼 | `.sample(episode)` | 166 | Picks a scenario by the curriculum and generates it |
| 🅼 | `.curriculum_schedule(episode)` | 171 | Returns the scenario type for an episode (threshold-based ramp) |
| 🅼 | `.summary(...)` | 182 | Human-readable description string |
| 🅼 | `._gen_offpeak/_morning_peak/_evening_peak/_incident/_pulse(...)` | 202–251 | The 5 demand builders (flow ranges per `master_ai.md §10`) |
| 🅼 | `._write_rou_xml(out_path, flows)` | 276 | Serialises flows to SUMO route XML |
| 🅼 | `._check_edge / _parse_net_edges / _make_rng` | 303–316 | Validates every edge against the net file; seeded RNG per episode |
| 🏛 | `ScenarioEnvironment` | 322 | Thin wrapper that delays the `traci` import to `start()` and proxies to `SumoEnvironment` |
| 🅼 | `.set_route_file / .start / .close / .__getattr__` | 330–391 | Route injection, episode start, teardown, attribute proxy |
| ƒ | `_import_sumo_env()` | 398 | Lazy import of the training env (keeps SUMO out of module load) |

---

## 4. Shared obs / reward — `backend/ai/trafix_v2.py`

> **Load-bearing for all versions.** The observation parser, reward, and GAE live here and are
> imported by the v6 backend and every trainer. The v2 *model* classes are legacy/unused.

| Sym | Name | Line | What it does |
|-----|------|-----:|--------------|
| 🔢 | `_LANE_KEYS`, `_NS_KEYS`, `_EW_KEYS` | 35–45 | Canonical lane-name lists |
| 🔢 | `_NORM` | 48 | Per-feature normalisers (÷15/÷30/÷200/…) |
| 🔢 | `NUM_NODE_FEATURES=20` | 60 | The obs width; imported everywhere as `OBS_DIM` |
| ƒ | **`parse_sumo_observations(obs_list, device=None)`** | 63 | **Single source of truth** for telemetry→`[5,20]` float tensor |
| 🏛 | `SpatioTemporalGNN(nn.Module)` | 107 | Legacy v2 GCN (no GRU despite the name) |
| 🏛 | `IntersectionCoordinator(nn.Module)` | 126 | Legacy v2 attention block |
| 🏛 | `CoordinatedPPOAgent(nn.Module)` | 155 | Legacy v2 agent (`forward`@189, `select_actions`@197, `compute_ppo_loss`@204) |
| 📦 | `RewardWeights` (`@dataclass`) | 230 | The 8 reward weights (pressure −0.30 … starvation −0.15; fairness 0.0) |
| ƒ | `_intersection_total(o)` | 242 | Sums the 12 lane counts for one junction |
| ƒ | `_compute_green_wave(cur, prev)` | 247 | Global green-wave bonus along directed edges with platoons |
| ƒ | **`compute_reward(...)`** | 290 | Per-junction `(N,)` reward tensor (used by Stage-3 PPO) |
| ƒ | **`compute_gae(...)`** | 384 | Generalised Advantage Estimation (γ=0.99, λ=0.95), standardised |
| ƒ | `train_step(...)` | 427 | Legacy v2 single-sample PPO step |

---

## 5. Training env & utils — `backend/ai/train_v2.py`

| Sym | Name | Line | What it does |
|-----|------|-----:|--------------|
| 📦 | `TrainConfig` (`@dataclass`) | 110 | Centralises all env/v2 hyperparameters |
| 🔢 | `MODEL_TO_SUMO_GREEN`, `SUMO_TO_MODEL_PHASE`, `LANE_TYPE` | 168–170 | The model↔SUMO phase maps + lane-index→movement |
| 🏛 | **`SumoEnvironment`** | 173 | The TraCI wrapper used by **all** trainers (`YELLOW_STEPS=3`@179) |
| 🅼 | `.start(episode)` | 193 | Launches SUMO with `--time-to-teleport -1`, seed=base+episode, warm-up |
| 🅼 | `.is_running()` | 260 | Whether SUMO is still alive |
| 🅼 | **`.get_observations()`** | 269 | Reads lane counts, classifies movement+direction, builds the obs dicts |
| 🅼 | `._classify_edge_direction(edge, jx, jy)` | 328 | Geometry → compass direction |
| 🅼 | `.apply_actions(actions)` | 345 | Model phase → SUMO green via yellow transitions |
| 🅼 | `._advance_transitions()` | 375 | Counts down yellows, applies queued greens |
| 🅼 | `.step(actions)` | 391 | Apply + advance `decision_interval` steps; returns `(obs, done)` |
| 🅼 | `.get_metrics()` | 418 | Queue/wait/throughput metrics for logging |
| ƒ | `build_edge_index(num_nodes, net_file)` | 449 | Reads graph from net via sumolib, else the hard-coded chain |
| 🏛 | `RolloutBuffer` | 517 | Stores transitions; `.to_dict(edge_index, next_value)`@547 emits a training batch |
| 🏛 | `TrainingLogger` | 563 | Per-episode logging + `save_history`@613 |
| 🏛 | `CosineWarmupScheduler` | 622 | Cosine LR decay with warm-up (`.step(episode)`@638) |
| ƒ | `train(cfg)` | 660 | Legacy v2 training entry |
| ƒ | `_save_checkpoint(...)` | 968 | Writes model+optimizer+episode+best_reward |
| ƒ | `parse_args()` | 997 | CLI → `TrainConfig` |

---

## 6. Training stage scripts

> Orchestrated by `train_v6.py`. Each produces a checkpoint consumed by the next.

### `train_v6.py` (orchestrator)
| ƒ | Name | Line | What it does |
|---|------|-----:|--------------|
| ƒ | `_header(title)` / `_run(cmd,label)` | 38 / 45 | Pretty banner; subprocess wrapper |
| ƒ | `stage1(args)` / `stage2(args)` / `stage3(args)` | 60 / 76 / 99 | Invoke each stage script |
| ƒ | `main()` | 134 | Run all 3 stages (or `--stage N`) |

### `trafix_v6/stage1_pretrain_gru.py` — GRU pretrain (next-step prediction)
`make_env_config`@95 · `train(args)`@107 (trains GRU + temp `Linear(128→12)` head, random actions) · `parse_args`@215 → `checkpoints/stage1_gru.pt`

### `trafix_v6/stage2_pretrain_gatconv.py` — GAT+trunk pretrain (neighbour-queue prediction)
`_nmse(pred,target)`@99 (variance-normalised MSE) · `make_env_config`@105 · `train(args)`@117 (loads & freezes GRU; offpeak curriculum first) · `parse_args`@266 → `stage2_gatconv.pt`, `stage2_trunk.pt`

### `trafix_v6/stage3_train_ppo.py` — full PPO (**produces the production model**)
| Sym | Name | Line | What it does |
|-----|------|-----:|--------------|
| 🏛 | `RolloutBuffer` | 113 | Collects windows/actions/log-probs/rewards/values (`.add`@124, `.clear`@117) |
| ƒ | **`ppo_update(...)`** | 139 | Clipped PPO + clipped value loss + KL early-stop; governs via `apply_stateless_batch` |
| ƒ | `save_checkpoint(...)` | 231 | Every 100 eps → `stage3_ep{N}.pt`; end → `trafix_v6_final.pt` |
| ƒ | `train(args)` | 248 | Differential LRs, freeze-then-unfreeze, cosine decay, rollout=64, 2000 eps |
| ƒ | `parse_args()` | 550 | CLI |

### `trafix_v6/finetune_morning_peak.py` (optional)
`RolloutBuffer`@104 · `ppo_update`@130 · `quick_eval`@229 (anti-forgetting eval) · `finetune(args)`@301 (LR 1e-5, encoders frozen, 70/30 curriculum, gated save) · `parse_args`@615

### `trafix_v6/eval_stage3.py`
`run_episode(model, env, ...)`@81 (one scenario rollout) · `evaluate(args)`@137 (`_mean`@205 helper) · `parse_args`@218

---

## 7. Backend API — `backend/main.py`

| Sym | Name | Line | What it does |
|-----|------|-----:|--------------|
| 🔢 | `_MODEL_VERSION` | 24 | Reads `TRAFIX_MODEL_VERSION` (default v6) — the model switch |
| 🔢 | `_USE_V6/_USE_V5/_USE_GRAPH`, `_WEIGHT_FILENAME` | 26–53 | Version flags + weight path set at import |
| 📦 | `Telemetry(BaseModel)` | 73 | One junction's input schema (12 lanes + queue + phase + duration + legacy `*_count`) |
| 🔢 | `NUM_FEATURES=20`, `NUM_ACTIONS=6`, `NUM_NODES=5`, `edge_index` | 101–111 | Inference dims + chain graph |
| 🔢 | `state_dict` | 69 | Global latest telemetry per junction — **what `/state` returns** |
| 🔢 | `_V6_T_WINDOW=30`, `_v6_window` (deque), `_v6_governor`, `_last_batch_step` | 124–127 | Rolling window + governor + restart tracker |
| ƒ | **`load_model()`** | 130 | Instantiates `TraFixV6`, loads checkpoint, builds production `RuleGovernor`; sets fallback if missing |
| ƒ | `startup_event()` | 239 | `@on_event("startup")`: load model + open DB session |
| ƒ | `shutdown_event()` | 252 | Closes the DB session |
| ƒ | `_build_obs_list()` | 260 | 5-element obs list (zero-fills missing junctions) |
| 📦 | `TelemetryBatch(BaseModel)` | 278 | `{step, intersections:[Telemetry]}` |
| ƒ | **`receive_telemetry_batch(batch, …)`** | 286 | `POST /telemetry_batch`: restart-detect → stash → parse → window → model → govern → argmax → log → return |
| 📦 | `EmergencyEventBatch(BaseModel)` | 442 | `{events:[…]}` |
| ƒ | `receive_emergency_event(...)` | 447 | `POST /emergency_event`: buffer + persist |
| ƒ | `get_emergency_metrics()` | 456 | `GET /emergency_metrics`: events + summary |
| ƒ | `get_db_summary()` | 477 | `GET /db_summary` |
| ƒ | `get_last_decisions()` | 485 | `GET /last_decisions` (cached) |
| ƒ | `get_state()` | 490 | `GET /state` — the dashboard's data source |

---

## 8. Database — `backend/database.py`

| Sym | Name | Line | What it does |
|-----|------|-----:|--------------|
| 🔢 | `_DDL` | 33 | Schema: `sessions`, `step_log`, `emergency_log` (+ index) |
| ƒ | `init_db(url)` | 76 | Opens the threaded pool, runs DDL; returns False (no crash) if PG down |
| ƒ | `_exec_ddl / _conn / _put` | 95–111 | Pool helpers |
| ƒ | `create_session(scenario)` | 118 | Inserts a session row, returns id |
| ƒ | `close_session(id)` | 138 | Stamps `ended_at` |
| ƒ | **`log_step(session_id, sim_step, decisions, telemetry)`** | 158 | One row per junction per decision (off the request thread) |
| ƒ | `log_emergency(session_id, event)` | 216 | One row per completed preemption event |
| ƒ | `query_session_summary(id)` | 256 | Row counts + avg queue for the dashboard/tests |

---

## 9. Bridge — `main.py`

| ƒ | Name | Line | What it does |
|---|------|-----:|--------------|
| 🔢 | imports `app` from `backend.main` | 5 | All API routes come from the backend |
| ƒ | `dashboard()` | 13 | `GET /` → `frontend/dashboard.html` |
| ƒ | `architecture()` | 18 | `GET /architecture` → `frontend/index.html` |
| ƒ | `emergency()` | 23 | `GET /emergency` → `frontend/emergency.html` |

---

## 10. Live runner — `sumo/run_sumo_live.py`

| Sym | Name | Line | What it does |
|-----|------|-----:|--------------|
| 🔢 | `API_URL` | 31 | `http://127.0.0.1:{port}/telemetry` (rewritten to `/telemetry_batch`, `/emergency_event`) |
| 🔢 | `MODEL_TO_SUMO_GREEN`, `SUMO_TO_MODEL`, `PHASE_NAMES` | 34–44 | Phase maps + human names |
| 🔢 | `MIN_GREEN_THROUGH=10`, `MIN_GREEN_LEFT=8`, `YELLOW_STEPS=3`, `ALL_RED_STEPS=2`, `DECISION_INTERVAL=10` | 46–50 | Actuation timing |
| 🔢 | `MAX_EMERGENCY_GREEN/STUCK_SPEED/PUSH_AFTER`, `LANE_TYPE`, `_LOG_FILE` | 51–60 | Emergency tuning, lane map, log path |
| ƒ | `_classify_edge_direction(edge, jx, jy)` | 73 | Geometry → compass direction |
| ƒ | `build_intersection_map()` | 88 | `{tls_id:{jx,jy}}` for all lights |
| ƒ | **`collect_lane_obs(tls_id, jx, jy)`** | 96 | Reads the 12 per-lane counts via `getLastStepVehicleNumber` |
| ƒ | `_log_summary(...)` | 126 | Prints the phase-distribution table |
| ƒ | **`main()`** | 145 | The sim loop (see locals below) |
| 🔢 | `STARVE_LIMIT=8` (+`decisions_since_through`) | 179 | Through-starvation override |
| 🔢 | `DIRECTION_STARVE_LIMIT=10` (+`decisions_since_ns/ew`) | 186 | Direction-starvation override |
| 🔢 | `LEFT_STARVE_LIMIT=15`, `_LEFT_DEMAND_KEYS`, `decisions_since_left` | 194–202 | Left-turn starvation override |
| 🔢 | `FALLBACK_THRESHOLD=3`, `FALLBACK_CYCLE=40` | 207–208 | Fixed-time fallback when API is down |

> The loop body (≈224–532): step SUMO → advance yellows → `preempt.update` per junction →
> every 10 steps build+POST telemetry → apply decisions with the 4 overrides + yellow
> transitions → fixed-time fallback on failures.

---

## 11. Emergency preemption — `sumo/emergency_preemption.py`

| Sym | Name | Line | What it does |
|-----|------|-----:|--------------|
| 🏛 | **`EmergencyPreemptionController`** | 31 | Per-junction state machine + metric collector, decoupled from the sim loop |
| 🅼 | `.__init__(traci, tls_ids, log, …)` | 40 | Stores timing; inits `stage/edge/countdown/_session/_completed` |
| 🅼 | `.is_active(tls_id)` | 74 | True if this junction is under preemption (gates AI/fallback) |
| 🅼 | `.pop_completed_sessions()` | 78 | Drains finished event records (runner POSTs them) |
| 🅼 | `._closest_emergency(tls_id)` | 86 | **① Detection**: nearest `type=="emergency"` vehicle on an approach |
| 🅼 | `._emergency_on_edge / _emergency_speed / _held_vehicle_ids` | 141–189 | Detection helpers (still present? speed? who's waiting?) |
| 🅼 | `._build_state(tls_id, edge, on_char)` | 221 | **③ Actuation**: green the approach, red everything else |
| 🅼 | `._set_state / _set_all_red` | 239–246 | `setRedYellowGreenState` wrappers |
| 🅼 | `._restore_program(tls_id)` | 253 | **Critical**: restore program "0" so AI's `setPhase` works again |
| 🅼 | `._junction_name / _approach_dir` | 266–273 | Human labels (K1..K5, compass) for the dashboard |
| 🅼 | `._open_session / _accumulate_wait / _close_session` | 297–320 | Metric lifecycle (transit_steps, vehicles_waited, total_wait_steps) |
| 🅼 | **`.update(tls_id, step, *, yellow_remaining, pending_targets, last_phase_change_step)`** | 351 | **② Decision**: advances `None→yellow→allred→green→return_yellow→None` |

---

## 12. Test runner — `tests/utils/sumo_runner.py`

| Sym | Name | Line | What it does |
|-----|------|-----:|--------------|
| 🔢 | `_T_WINDOW=30`, `_MODEL_TO_SUMO_GREEN`, `_SUMO_TO_MODEL_PHASE`, `_LANE_TYPE` | 57–64 | Mirror of the live constants (kept identical on purpose) |
| 🔢 | `_GRIDLOCK_SPEED_THRESHOLD=0.5`, `_GRIDLOCK_DURATION_S=300` | 67–68 | Gridlock detector thresholds |
| ƒ | `_get_observations(tls_ids, phase_held_since, step)` | 73 | Test-side copy of the live lane-classification obs builder |
| ƒ | `_classify_direction(edge, jx, jy)` | 116 | Geometry → direction |
| 🏛 | `_InlineTracker(tls_ids, tls_incoming)` | 132 | TraCI-only metrics: stops/vehicle, per-approach wait & halt, fairness variance |
| 🅼 | `.update()` / `.results()` | 148 / 173 | Accumulate per step / reduce to a metrics dict |
| ƒ | **`run_simulation(run_id, route_file, mode, …)`** | 228 | Runs one SUMO sim in `baseline` or `ai` mode; writes all output XML + `inline_metrics.json` + `gridlock.json` |

---

## 13. Test orchestration, metrics, analysis

### Mandatory pytest suite (`tests/mandatory/`, 121 tests)
Test **classes** by file (each holds the individual `test_*` methods listed in
`tests/mandatory/test_list.md`):

- `test_unit_model.py` — `TestTemporalEncoder`, `TestGraphEncoder`, `TestTraFixV6Forward`,
  `TestTraFixV6GetAction`, `TestTraFixV6EvaluateActions`, `TestTraFixV6Checkpoint`
- `test_unit_rule_governor.py` — `TestHardMaskMinGreen`, `TestHardMaskMaxGreen`,
  `TestAntiFlicker`, `TestPressureBonus`, `TestApplyOutputShape`
- `test_unit_observation.py` — `TestOutputShape`, `TestNormalisation`, `TestPhaseOneHot`,
  `TestOrdering`, `TestEdgeCases`
- `test_unit_preemption.py` — `TestInitialState`, `TestPreemptionActivation`,
  `TestStateTransitions`, `TestMetricsCollection`, `TestBuildState`
- `test_nfr.py` — `TestNFR01InferenceLatency`, `TestNFR02Fallback`, `TestNFR05CrossPlatform`
- `test_fr.py` — `TestFR01TelemetryIngestion`, `TestFR02AIPhaseSelection`,
  `TestFR03EmergencyPreemption`, `TestFR04DatabaseLogging`, `TestFR06YellowPhase`
- `test_metrics_modules.py` — `TestTravelTime`, `TestWaitingTime`, … `TestJunctionFairness`,
  `TestAllMetricsReturnDicts`

### Runners (`tests/runners/`)
- `run_all.py` — `_route`@57, `_hash_tripinfo`@69, `_hash_run`@78, `_run_pair`@92,
  `_repro_check`@123, `main`@164 (Type 1 + Type 2)
- `run_checkpoint_compare.py` — `_route`@59, `main`@66 (3 controllers × 13 scenarios)
- `run_test_type_2.py` — `_route_file`@69, `_demand_hash`@79, `_find_standard_output`@83,
  `_check_prerequisites`@101, `main`@121 (roundabout vs standard)

### Metric modules (`tests/metrics/`)
One `compute_<name>(sumo_output_dir)` per file, each returning a dict:
`compute_travel_time`, `compute_waiting_time`, `compute_time_loss`, `compute_queue_length`,
`compute_throughput`, `compute_network_speed`, `compute_teleports`, `compute_emissions`,
`compute_fuel_consumption`, `compute_pollutants`, `compute_stops_per_vehicle`,
`compute_junction_fairness` (all near line 11–18 of their file).

### Analysis (`tests/analysis/compare.py`)
`_extract_all_metrics`@38, `_pct_change`@96, `write_results_csv`@106, `write_summary_report`@122,
`write_charts`@181, `run_analysis`@231 — turn raw outputs into CSV, markdown, and PNG charts.

---

## 14. Module-level constants/structures cross-reference

The **same constants are intentionally duplicated** across the live and training/test paths so
each can run standalone. If you change one, change all — they form an implicit contract.

| Structure | Live runner | Backend | Training env | Test runner | Model/governor |
|-----------|-------------|---------|--------------|-------------|----------------|
| `MODEL_TO_SUMO_GREEN` (model→even green) | `run_sumo_live.py:34` | — | `train_v2.py:168` | `sumo_runner.py:62` | — |
| `SUMO_TO_MODEL(_PHASE)` (12→6) | `:35` | — | `:169` | `:63` | — |
| `LANE_TYPE` (idx→movement) | `:55` | — | `:170` | `:61` | — |
| `YELLOW_STEPS=3` | `:48` | — | `:179` | `:60` | — |
| `DECISION_INTERVAL=10` | `:50` | — | (cfg) | (arg) | — |
| `MIN/MAX_GREEN` | `:46–47` | governor ctor `main.py:154` | — | governor ctor `:286` | `rule_governor.py:52–55` |
| `T window=30` | — | `main.py:124` | — | `:57` | (model input) |
| obs `_IDX_*` / `_NORM` | — | — | (`get_observations`) | — | `rule_governor.py:29–47`, `trafix_v2.py:48` |
| chain `edge_index` | — | `main.py:108` | `build_edge_index` `:449` | — | `_make_chain_edge_index` `:36` |
| `OBS_DIM=20` / `NUM_PHASES=6` / `NUM_JUNCTIONS=5` | — | `main.py:101–104` | — | `:58–59` | `trafix_v6.py:31–33`, `trafix_v2.py:60` |

> **Defense point.** "Why are these maps copy-pasted in 4 places instead of imported?" → so each
> process has zero cross-dependencies at runtime: the backend never imports SUMO code, and the
> runner/test harness never import the FastAPI app. The duplication is the price of that clean
> process boundary; the cross-reference table above is how you verify they stay in sync.

---

*Symbol index companion to `master_ai.md` (model/training) and `system_explained.md` (data flow).
All file:line references read from source on branch `main`; line numbers are approximate after edits.*
