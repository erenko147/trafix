# TraFix v6 — Mandatory Test List

**121 tests across 7 files — all pass.**
Run: `python -m pytest tests/mandatory/ -v`

---

## 1. Unit Tests — AI Model (`test_unit_model.py`) — 22 tests

### `TestTemporalEncoder`
- `test_output_shape` — GRU output is `[batch=1, J=5, hidden=128]`
- `test_batch_shape` — GRU output is `[batch=4, J=5, hidden=128]` under batched input
- `test_no_nan` — no NaN in GRU output on random input
- `test_different_inputs_different_outputs` — distinct observations produce distinct GRU encodings

### `TestGraphEncoder`
- `test_output_shape` — GATConv output is `[J=5, 128]` (4 heads × 32)
- `test_no_nan` — no NaN in GATConv output on random input

### `TestTraFixV6Forward`
- `test_logits_list_length` — `forward()` returns exactly 5 logit tensors (one per junction)
- `test_logits_shape` — each logit tensor is `[batch, 6]`
- `test_value_shape` — critic value is `[batch, J=5]`
- `test_no_nan_in_logits` — no NaN in actor logits
- `test_no_nan_in_value` — no NaN in critic value
- `test_actor_probabilities_sum_to_one` — `softmax(logits).sum() ≈ 1.0` per junction
- `test_actor_probabilities_non_negative` — all softmax outputs ≥ 0
- `test_chosen_phase_in_valid_range` — `argmax(logits) ∈ [0, 5]` for all junctions
- `test_batch_inference` — forward pass works correctly with batch size > 1

### `TestTraFixV6GetAction`
- `test_action_shape` — `get_action()` returns actions of shape `[batch, J=5]`
- `test_actions_in_valid_range` — sampled actions ∈ `[0, 5]`
- `test_log_probs_finite` — log-probabilities are finite (no `-inf`, no NaN)

### `TestTraFixV6EvaluateActions`
- `test_evaluate_actions_shapes` — `evaluate_actions()` returns `log_probs`, `entropy`, `value` of shape `[B, J]`
- `test_entropy_non_negative` — entropy ≥ 0 for all junctions

### `TestTraFixV6Checkpoint`
- `test_checkpoint_exists` — `trafix_v6/checkpoints/trafix_v6_final.pt` is present on disk
- `test_checkpoint_loads` — checkpoint loads into `TraFixV6` without shape mismatch

---

## 2. Unit Tests — Rule Governor (`test_unit_rule_governor.py`) — 12 tests

### `TestHardMaskMinGreen`
- `test_phase_locked_before_min_green_through` — through phase (0 or 3) held for < 10 s: all other phase logits set to −∞
- `test_phase_locked_before_min_green_left` — left-turn phase (1, 2, 4, 5) held for < 8 s: all other phase logits set to −∞
- `test_phase_free_after_min_green` — once min-green elapsed, no logit masking applied

### `TestHardMaskMaxGreen`
- `test_current_phase_blocked_after_max_green` — current phase logit set to −∞ after 90 s (through) or 45 s (left-turn)
- `test_other_phases_available_after_max_green` — non-current phase logits remain unmasked

### `TestAntiFlicker`
- `test_reversal_penalised` — switching A→B→A applies −10.0 penalty to phase A's logit
- `test_no_penalty_without_reversal` — A→B→C sequence incurs no anti-flicker penalty
- `test_reset_clears_history` — `reset()` clears phase history so no false penalties after reset

### `TestPressureBonus`
- `test_congested_ns_gets_boost` — heavy NS through-lane counts boost phase 0 logit
- `test_congested_ew_gets_boost` — heavy EW through-lane counts boost phase 3 logit

### `TestApplyOutputShape`
- `test_output_list_length` — `apply()` returns a list of exactly 5 logit tensors
- `test_output_logit_shape` — each returned logit tensor has shape `[1, 6]`

---

## 3. Unit Tests — Observation Parser (`test_unit_observation.py`) — 15 tests

### `TestOutputShape`
- `test_shape_five_junctions` — `parse_sumo_observations()` returns tensor of shape `[5, 20]`
- `test_dtype_float32` — output dtype is `float32`
- `test_no_nan` — no NaN values in output for standard telemetry input

### `TestNormalisation`
- `test_queue_normalised` — queue_length normalised by ÷ 200 (value in `[0, 1]` for queue ≤ 200)
- `test_zero_queue` — zero queue_length → feature index 12 = 0.0 (no division-by-zero)
- `test_duration_normalised` — phase_duration normalised by ÷ 120
- `test_duration_capped_at_3` — phase_duration > 360 s produces feature = 3.0 (cap enforced)
- `test_lane_counts_non_negative` — all 12 lane-count features ≥ 0 after normalisation

### `TestPhaseOneHot`
- `test_phase_zero_one_hot` — phase 0 → one-hot `[1,0,0,0,0,0]` at indices 13–18
- `test_phase_three_one_hot` — phase 3 → one-hot `[0,0,0,1,0,0]` at indices 13–18
- `test_each_phase_produces_unique_encoding` — all 6 phases produce distinct one-hot vectors
- `test_phase_one_hot_sums_to_one` — one-hot block (indices 13–18) sums to exactly 1.0

### `TestOrdering`
- `test_sorted_by_junction_id` — junctions are ordered by `intersection_id` regardless of input list order

### `TestEdgeCases`
- `test_zero_vehicles` — all vehicle counts = 0: output is valid, no NaN
- `test_high_counts_do_not_produce_nan` — vehicle counts of 999: output is finite

---

## 4. Unit Tests — Emergency Preemption (`test_unit_preemption.py`) — 13 tests

### `TestInitialState`
- `test_initially_inactive` — all 5 junctions report `is_active() = False` before any update
- `test_pop_completed_sessions_empty_at_start` — no completed sessions at initialisation
- `test_update_no_emergency_stays_inactive` — calling `update()` with no emergency vehicle present leaves junction inactive

### `TestPreemptionActivation`
- `test_activates_on_emergency` — `is_active()` returns `True` within 1 step of detecting emergency vehicle
- `test_stage_is_yellow_after_activation` — internal stage = `"yellow"` immediately after activation
- `test_cancels_pending_yellow` — active preemption clears any queued yellow-phase countdown
- `test_other_junctions_not_affected` — activating preemption on J0 leaves J1–J4 inactive

### `TestStateTransitions`
- `test_yellow_to_allred` — after `YELLOW_STEPS` (3) steps, stage transitions `yellow → allred`
- `test_allred_to_green` — after `ALL_RED_STEPS` (2) steps in allred, stage transitions `allred → green`

### `TestMetricsCollection`
- `test_session_opened_on_activation` — a metrics session dict is created when preemption activates
- `test_completed_event_recorded` — completed event contains `transit_steps`, `vehicles_waited`, `result`

### `TestBuildState`
- `test_approach_edge_gets_green` — the approach lane toward the junction receives `'G'` in the TLS state string
- `test_non_approach_edges_get_red` — all other lanes receive `'r'` in the TLS state string

---

## 5. NFR Tests (`test_nfr.py`) — 18 tests

### `TestNFR01InferenceLatency` — NFR-01: AI inference ≤ 1 000 ms
- `test_single_call_under_500ms` — a single inference + RuleGovernor call completes in < 1 000 ms on CPU
- `test_p99_under_500ms` — p99 over 50 warm calls < 1 000 ms; prints mean / p50 / p99 / limit
- `test_all_phases_returned` — inference returns exactly 5 phase decisions (one per junction)

### `TestNFR02Fallback` — NFR-02: Heuristic fallback produces valid output without the AI model
- `test_fallback_returns_valid_phase` — heuristic always returns a phase in `{0, 3, current_phase}`
- `test_fallback_works_for_all_junctions` — fallback produces a valid decision for all 5 junctions
- `test_fallback_holds_phase_if_short_duration` — phase_duration < 10 s → hold current phase
- `test_fallback_chooses_busier_direction` — NS-heavier demand → phase 0; EW-heavier → phase 3
- `test_fallback_no_crash_on_empty_input` — all-zero telemetry does not raise an exception

### `TestNFR05CrossPlatform` — NFR-05: Core pipeline runs on Linux and Windows Python environments
- `test_torch_available` — `torch.Tensor` accessible
- `test_torch_geometric_available` — `torch_geometric.__version__` accessible
- `test_fastapi_available` — `fastapi.FastAPI` accessible
- `test_pydantic_available` — `pydantic.BaseModel` accessible
- `test_psycopg2_available` — `psycopg2.connect` accessible
- `test_python_version` — Python ≥ 3.10
- `test_platform_reported` — `platform.system()` ∈ `{"Linux", "Windows", "Darwin"}`
- `test_model_instantiates_on_cpu` — `TraFixV6` instantiates and runs a forward pass on CPU
- `test_trafix_v6_module_imports` — `trafix_v6.trafix_v6` and `trafix_v6.rule_governor` importable
- `test_backend_module_imports` — `backend.main` and `backend.database` importable

---

## 6. Functional Requirement Tests (`test_fr.py`) — 22 tests

### `TestFR01TelemetryIngestion` — FR-01: Telemetry contains all required fields
- `test_all_required_keys_present` — all 15 required keys present in each junction's telemetry dict
- `test_vehicle_counts_non_negative` — all 12 directional lane counts ≥ 0
- `test_queue_length_non_negative` — queue_length ≥ 0.0
- `test_phase_in_valid_range` — current_phase ∈ `[0, 5]`
- `test_parse_accepts_telemetry_format` — `parse_sumo_observations()` accepts standard dict format and returns `[5, 20]`

### `TestFR02AIPhaseSelection` — FR-02: AI outputs a valid phase for every junction
- `test_returns_logits_for_all_junctions` — forward pass returns exactly 5 logit tensors
- `test_phase_in_valid_range` — `argmax(logits) ∈ [0, 5]` for all junctions
- `test_probabilities_valid` — softmax probabilities sum to 1.0 and are all ≥ 0
- `test_value_is_finite` — critic value tensor contains only finite numbers
- `test_model_is_deterministic_in_eval` — identical input in `model.eval()` produces identical logits on two runs

### `TestFR03EmergencyPreemption` — FR-03: Emergency preemption overrides AI within 1 simulation step
- `test_preemption_activates_in_one_step` — `is_active()` is `True` after a single `update()` call with emergency vehicle
- `test_preemption_sets_tls_state` — controller calls `traci.trafficlight.setRedYellowGreenState` or `setPhase`
- `test_ai_overridden_during_preemption` — `is_active()` signals the simulation loop to skip AI phase decision
- `test_no_preemption_without_emergency` — no emergency vehicle present → `is_active()` remains `False`

### `TestFR04DatabaseLogging` — FR-04: Database module is present and fails gracefully without a live DB
- `test_db_module_importable` — `backend.database` exposes `init_db`, `create_session`, `log_step`, `close_session`
- `test_log_step_graceful_when_no_db` — `log_step()` with `_pool=None` returns silently (no exception)
- `test_log_emergency_graceful_when_no_db` — `log_emergency()` with `_pool=None` returns silently

### `TestFR06YellowPhase` — FR-06: A yellow phase is inserted between any two distinct green phases
- `test_yellow_inserted_on_phase_change` — switching phase A→B sets an odd SUMO phase (yellow) and populates `yellow_remaining`
- `test_no_yellow_when_same_phase` — no phase change → even SUMO phase (green), `yellow_remaining` untouched
- `test_target_queued_correctly` — target SUMO phase for model phase 3 is stored as `6` in `pending_targets`
- `test_yellow_decrements_each_step` — `yellow_remaining` counts down 3→2→1→0 then target phase is applied
- `test_yellow_duration_is_three_steps` — yellow hold is exactly 3 simulation steps (≈ 3 s)

---

## 7. Metric Module Tests (`test_metrics_modules.py`) — 19 tests

All tests use fixture data from `tests/utils/fixtures/` (pre-recorded SUMO output files).

| Test class | Tests | What is verified |
|------------|-------|-----------------|
| `TestTravelTime` | 2 | `mean_travel_time_s` ≥ 0; min ≤ mean ≤ max |
| `TestWaitingTime` | 2 | `mean_waiting_time_s` ≥ 0; waiting ≤ travel time |
| `TestTimeLoss` | 1 | `mean_time_loss_s` ≥ 0 |
| `TestQueueLength` | 1 | `mean_halting_per_junction` ≥ 0 |
| `TestThroughput` | 2 | `throughput_veh_per_hr` ≥ 0; arrived ≤ total trips + still running |
| `TestNetworkSpeed` | 1 | `mean_speed_ms` ≥ 0 |
| `TestTeleports` | 1 | `total_teleports` ≥ 0 and is numeric |
| `TestEmissions` | 1 | `mean_CO2_per_vehicle_mg` ≥ 0 |
| `TestFuelConsumption` | 2 | `mean_fuel_per_vehicle_L` ≥ 0; fuel > 0 when vehicles completed trips |
| `TestPollutants` | 2 | `totals_mg` dict present; all 4 pollutants (NOx, PMx, HC, CO) ≥ 0 |
| `TestStopsPerVehicle` | 1 | `avg_stops_per_vehicle` ≥ 0 |
| `TestJunctionFairness` | 2 | `network_wide_variance` ≥ 0 and is finite (not NaN) |
| `TestAllMetricsReturnDicts` | 1 | Smoke test: every compute_* function returns a non-empty dict |

---

## Summary

| File | Category | Tests | Status |
|------|----------|-------|--------|
| `test_unit_model.py` | Unit | 22 | ✅ All pass |
| `test_unit_rule_governor.py` | Unit | 12 | ✅ All pass |
| `test_unit_observation.py` | Unit | 15 | ✅ All pass |
| `test_unit_preemption.py` | Unit | 13 | ✅ All pass |
| `test_nfr.py` | NFR | 18 | ✅ All pass |
| `test_fr.py` | FR | 22 | ✅ All pass |
| `test_metrics_modules.py` | Metrics | 19 | ✅ All pass |
| **Total** | | **121** | **✅ All pass** |
