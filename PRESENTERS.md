# TraFix — Oral Presentation: Presenter Assignment Sheet

Six presenters. **Person 1 owns the entire ML story (model + training) and nothing else.**
The other five are systems/engineering roles that need essentially **no machine-learning
knowledge** — safe to study immediately.

> **Stable-vs-changing rule:** *structure, mechanism and methodology are FINAL now; only
> result-numbers and the deployed weights change after the retrain.*
> The retrain (done on a separate machine) will change: the checkpoint weights,
> `best_reward`, the diagnostic AFTER tables, the Type-1/Type-2 result tables & figures,
> and the "overrides now fire rarely" claim. **Everything else below is already final.**

> ⚠️ **Two corrections everyone must know:**
> 1. The legacy **v2 model was REMOVED** (GCN + multi-head attention `CoordinatedPPOAgent`,
>    no trained weights). Present it as *lineage/why-removed*, not a live alternative.
>    Model lineage: **v2 → v5 (deleted earlier) → v6 (the only running model)**.
> 2. Governor `pressure_thresh` is now **0.12** (was 0.35), and the **reward has a new
>    per-movement anti-starvation term + `clear_bonus`**. Don't quote the old numbers.

---

## Person 1 — TraFix AI model & training  *(ML lead; heaviest role)*

**Scope:** v6 architecture (GRU temporal → GAT spatial → shared trunk → 5 actor heads +
hybrid local+global critic; ~101k params; why each block) · v2→v5→v6 lineage and **why v2
was removed** · 3-stage training (GRU pretrain → GAT pretrain → PPO) + `train_v6.py`,
differential LRs / freeze-unfreeze / cosine decay · PPO objective + GAE · the **reward
function & weights** (incl. new anti-starvation + `clear_bonus`) · scenario generator +
curriculum (5 types) · fine-tunes (`finetune_morning_peak`, new `finetune_argmax` = entropy
annealing + greedy/argmax checkpoint selection) + `eval_stage3` · **the argmax train/deploy
mismatch, the fix, and the diagnostic.**

**Files:** `trafix_v6/{trafix_v6,stage1_pretrain_gru,stage2_pretrain_gatconv,
stage3_train_ppo,finetune_morning_peak,finetune_argmax,eval_stage3,scenario_generator}.py`,
`train_v6.py`, reward/GAE in `backend/ai/trafix_v2.py`, `tests/diagnostics/*`.

**Likely questions:** Why GRU not LSTM/Transformer? Why GAT over GCN, and why a *chain*
graph? Why a hybrid (local+global) critic / CTDE? Why did greedy argmax collapse, and how
do the reward + greedy selection fix it? Why 3-stage pretraining instead of end-to-end PPO?

**Status:** method/architecture **stable now**; after training → new weights, `best_reward`,
diagnostic AFTER tables.

## Person 2 — Inference path & RuleGovernor  *(the backend "brain")*

**Scope:** `/telemetry_batch` flow (rolling 30-frame window, restart detection,
softmax→argmax, confidence, heuristic fallback) · the 20-dim observation contract
(`parse_sumo_observations`) and model↔SUMO **phase mapping** + 3-step yellow · the
**RuleGovernor** (min/max-green, pressure boost @ **0.12**, anti-flicker; three `apply`
variants).

**Files:** `backend/main.py`, `trafix_v6/rule_governor.py`.

**Likely questions:** What stops the AI from doing something illegal/unsafe? Min/max-green
vs the 10 s decision interval? What happens with no model loaded? Why argmax at deploy?

**Status:** **stable now** (mechanism); `pressure_thresh` already 0.12.

## Person 3 — Live runner, actuation & the safety net

**Scope:** `run_sumo_live.py` loop (DECISION_INTERVAL, yellow handling, fixed-time
fallback) · the **starvation overrides** (STARVE/DIRECTION/LEFT) — and that they're
**mirrored** in the test runner · SUMO network & configs (`map.net.xml` 5-junction/12-phase,
`training/demo.sumocfg`, `generate_demand.py`, `rebuild_network.py`).

**Files:** `sumo/run_sumo_live.py`, `sumo/*.sumocfg`, `sumo/*.net.xml`,
`sumo/generate_demand.py`.

**Likely questions:** "Your AI starved a left turn" → answered here. How do lights never
freeze? Why disable teleporting?

**Status:** loop/overrides/network **stable now**; after training → "overrides trigger rarely."

## Person 4 — Test framework  *(mandatory + external + runner core)*

**Scope:** the **121 mandatory tests** (FR, NFR, unit: model/observation/governor/preemption,
database, metrics-modules) · external **Type-1** (low/med/high), **Type-2** (morning/evening/
incident/pulse), **unseen** scenarios · the test runner core (`sumo_runner.py`: gridlock
detection, teleport sanity check, inline metrics, unfinished-vehicle capture).

**Files:** `tests/mandatory/*`, `tests/runners/{run_all,run_test_type_2}.py`,
`tests/utils/sumo_runner.py`, `tests/scenarios/*`.

**Likely questions:** What makes a test "mandatory"? How is gridlock detected? Why read
teleport totals from `statistics.xml`?

**Status:** structure/methodology **stable now**; pass/fail **numbers** land after retrain.

## Person 5 — Metrics & results/figures  *(evaluation + the presentation's data)*

**Scope:** the **12 metric modules** (throughput/completion %, queue [halting + crawl-aware
slow], travel time, teleports, junction fairness, waiting, time-loss, network speed,
stops/veh, emissions, fuel, pollutants) · the **analysis/reporting** layer producing tables
and plots.

**Files:** `tests/metrics/*`, `tests/analysis/{compare,make_figures*,
generate_results_table_type1/2,checkpoint_compare,junction_compare}.py`.

**Likely questions:** How do you avoid survivorship bias (cars still in network)? Why a
crawl-aware queue, not just halting? How is completion % computed?

**Status:** what each metric **computes/why** is **stable now**; the **figures/tables
regenerate** after retrain.

## Person 6 — Frontend, ambulance preemption, persistence & launchers

**Scope:** the three frontends (`dashboard/emergency/index.html`) and how they poll the API ·
**ambulance/emergency preemption** (`emergency_preemption.py` state machine +
`AMBULANS_PREEMPTION.md` + `/emergency_event` + `test_unit_preemption`) · the **database**
(`backend/database.py`: sessions/step_log/emergency_log) · launchers (`baslat.py` one-click,
`run.py`, `main.py` bridge, `simulate_live_data.py`).

**Files:** `frontend/*`, `sumo/emergency_preemption.py`, `AMBULANS_PREEMPTION.md`,
`backend/database.py`, `baslat.py`, `run.py`, `main.py`, `simulate_live_data.py`.

**Likely questions:** How does an ambulance preempt the lights, and why is it decoupled from
the AI loop? What is persisted and where? How do you launch the whole demo?

**Status:** **entirely stable now** — independent of model weights.

---

## Coordination seams (so two presenters don't contradict each other)

- **Governor** — Person 2 owns the mechanism; Person 1 must say "we also train *under* the governor."
- **Starvation overrides** — Person 3 owns them; Person 4 notes the test runner mirrors them.
- **Observation / phase mapping** — Person 2 owns the contract; Person 1 just cites the 20-dim input.
- **The argmax-fix narrative** — Person 1 owns it end-to-end; Persons 3 (overrides become rare)
  and 4/5 (numbers improve) should know the one-line version.

## Required reading (per the three source-of-truth docs)

`master_ai.md` (model/training/governor/reward — **stale on v2/governor/reward until the
post-training doc regen**), `system_explained.md` (data flow), `code_reference.md`
(symbol→file:line index). Until those are regenerated, trust the **code** over the docs for
v2 (removed), `pressure_thresh` (0.12) and the reward weights.
