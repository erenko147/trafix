# master_ai.md — TraFix AI Master Reference

> **Audience:** code review / oral defense. Everything below was read directly from
> the source on `main` (commit `87d872c`) and cross-checked against the checkpoints
> on disk and a live `import` of the model. Where the older
> `backend/ai/MODEL_EXPLANATION.md` disagrees with the actual code, the code wins and
> the discrepancy is flagged explicitly in **§13**.
>
> **One-sentence summary:** the production model is **TraFix v6**, a per-junction
> actor-critic that reads a 30-second rolling window of 20 features per junction,
> runs it through a **GRU (time) → GATConv (space) → shared MLP → 5 actor heads +
> hybrid critic**, has its raw logits constrained by a **RuleGovernor**, and emits
> one of **6 phases** for each of **5 junctions** every **10 simulated seconds**.
> It is trained in **3 stages** (GRU pretrain → GAT pretrain → full PPO).

---

## Table of Contents

1. [What actually runs in production](#1-what-actually-runs-in-production)
2. [The 20-dimensional observation](#2-the-20-dimensional-observation)
3. [Phases and the model↔SUMO mapping](#3-phases-and-the-modelsumo-mapping)
4. [The v6 model architecture, layer by layer](#4-the-v6-model-architecture-layer-by-layer)
5. [The RuleGovernor](#5-the-rulegovernor)
6. [Inference path: telemetry → decision](#6-inference-path-telemetry--decision)
7. [The reward function](#7-the-reward-function)
8. [GAE and the PPO objective](#8-gae-and-the-ppo-objective)
9. [Training pipeline (3 stages + fine-tune)](#9-training-pipeline-3-stages--fine-tune)
10. [The SUMO environment & scenario generator](#10-the-sumo-environment--scenario-generator)
11. [The live runner and its safety overrides](#11-the-live-runner-and-its-safety-overrides)
12. ["What happens if I change X?" — defense crib sheet](#12-what-happens-if-i-change-x--defense-crib-sheet)
13. [Discrepancies vs the old MODEL_EXPLANATION.md](#13-discrepancies-vs-the-old-model_explanationmd)
14. [File / import map](#14-file--import-map)
15. [Every number in one table](#15-every-number-in-one-table)
16. [Design rationale — why each structure (and the alternatives we rejected)](#16-design-rationale--why-each-structure-and-the-alternatives-we-rejected)

---

## 1. What actually runs in production

### Model selection

`backend/main.py` chooses a model from the `TRAFIX_MODEL_VERSION` env var (default `v6`):

| Version | Code path | Weight file | State on disk |
|---------|-----------|-------------|---------------|
| **v6** (default) | `trafix_v6/trafix_v6.py::TraFixV6` | `trafix_v6/checkpoints/trafix_v6_final.pt` | **Present & loaded** |
| v5 (legacy) | `trafix_v5/trafix_v5.py::TraFixV5` | `trafix_v5/checkpoints/trafix_v5_final.pt` | **`trafix_v5/` was deleted** (commit `76133fb`) → selecting v5 raises `ImportError` at startup |
| v2 (legacy) | model classes **removed** from `backend/ai/trafix_v2.py` | `coordinated_agent_weights.pth` | **No weight file, no model code** → backend v2 branch raises `ImportError`; selecting v2 falls back to heuristic |

> **Defense point:** *”Only v6 is runnable.”* `baslat.py` defaults to `--model v6`.
> If asked “what about v2/v5?”: v2's model classes (`SpatioTemporalGNN`,
> `IntersectionCoordinator`, `CoordinatedPPOAgent`, `train_step`) were **removed** from
> `trafix_v2.py` — the file is now observation/reward/GAE-only. v5 was removed from the
> repo. The backend keeps the version branches for forward compatibility only.

### The production checkpoint

`trafix_v6/checkpoints/trafix_v6_final.pt` — verified by loading it:

- Keys: `model_state_dict`, `optimizer_state_dict`, `episode`, `best_reward`
- `episode = 1999` (i.e. the 2000th episode, 0-indexed)
- `best_reward = -0.0292`
- **It is byte-for-byte identical to `stage3_ep2000.pt`** (confirmed by hashing the
  weights). `stage3_ep1000.pt` is kept only for checkpoint-comparison experiments.

### Launch

```bash
python baslat.py                 # v6 (default), SUMO GUI on
python baslat.py --no-gui        # headless
python baslat.py --model v6      # explicit
```

`baslat.py` starts two processes: (1) `uvicorn main:app` on port 8000 (FastAPI +
dashboard, with `TRAFIX_MODEL_VERSION` set from `--model`), and (2)
`sumo/run_sumo_live.py` which drives SUMO and POSTs telemetry to the API.

---

## 2. The 20-dimensional observation

Every junction is described by a **20-float vector** built by
`parse_sumo_observations()` in `backend/ai/trafix_v2.py` (this function is shared
by *all* model versions and by every training script — it is the single source of
truth for observation format).

| Index | Feature | Raw source | Normaliser |
|-------|---------|-----------|-----------|
| 0 | north_left | per-lane vehicle count | ÷ total-12-lane-sum |
| 1 | north_through | " | ÷ total-12-lane-sum |
| 2 | north_right | " | ÷ total-12-lane-sum |
| 3 | south_left | " | ÷ total-12-lane-sum |
| 4 | south_through | " | ÷ total-12-lane-sum |
| 5 | south_right | " | ÷ total-12-lane-sum |
| 6 | east_left | " | ÷ total-12-lane-sum |
| 7 | east_through | " | ÷ total-12-lane-sum |
| 8 | east_right | " | ÷ total-12-lane-sum |
| 9 | west_left | " | ÷ total-12-lane-sum |
| 10 | west_through | " | ÷ total-12-lane-sum |
| 11 | west_right | " | ÷ total-12-lane-sum |
| 12 | queue_length | total queue | ÷ 200 |
| 13–18 | current_phase | one-hot of phase 0–5 (`phase % 6`) | — |
| 19 | phase_duration | seconds held | `min(dur/120, 3.0)` |

**total-12-lane-sum** = `max(Σ(north_left … west_right), 1)` — the sum of all 12 raw lane counts at that junction, floored at 1 to avoid division by zero. Each lane feature is therefore a **junction-relative demand share** in `[0, 1]` that sums to 1.0 across the 12 lanes.

**Why junction-relative shares (not fixed ÷15 / ÷30):**

- **Scale invariance:** dividing by the per-junction total means 1 car out of 5 total and 4
  cars out of 20 total both read as `0.20`. The model receives the same signal regardless of
  whether the junction is quiet or busy. Fixed divisors (÷15, ÷30) would map near-zero
  raw counts to near-zero features, giving the model almost no gradient signal at low demand
  — exactly the root cause of argmax collapse at low/medium traffic.
- **All 12 lanes use the same denominator** so the ratio between e.g. a through lane and a
  left-turn lane reflects their actual relative demand, not their lane-capacity ratio.
- **÷200 for queue:** 200 is the saturation cap (`queue_length = min(total*1.5, 200.0)` in
  the env), mapping the queue to `[0, 1]`.
- **`min(dur/120, 3.0)`:** duration normalised by 120 s but **capped at 3.0 (= 6 min)**
  rather than 1.0 — keeps long holds distinguishable (a 2-min hold reads differently from a
  6-min hold).
- **`current_phase` one-hot is 6 bits** because v6 has 6 phases. `phase % 6` guards
  against any out-of-range phase id.

`NUM_NODE_FEATURES = 20` is defined here and imported everywhere as `OBS_DIM`.

---

## 3. Phases and the model↔SUMO mapping

The model speaks **“model phase” space (0–5)**. SUMO’s `tlLogic` uses **12 phases per
junction** (6 greens interleaved with 6 yellows). The net file
(`sumo/map.net.xml`) has 5 `tlLogic` blocks × 12 phases = **60 `<phase>` entries**.

| Model phase | Movement | SUMO green phase | SUMO yellow |
|-------------|----------|------------------|-------------|
| 0 | NS-through | 0 | 1 |
| 1 | N-left | 2 | 3 |
| 2 | S-left | 4 | 5 |
| 3 | EW-through | 6 | 7 |
| 4 | E-left | 8 | 9 |
| 5 | W-left | 10 | 11 |

Mapping dictionaries (identical in `train_v2.py`, `run_sumo_live.py`):

```python
MODEL_TO_SUMO_GREEN = {0:0, 1:2, 2:4, 3:6, 4:8, 5:10}      # green = model*2
SUMO_TO_MODEL_PHASE = {0:0,1:0, 2:1,3:1, 4:2,5:2, 6:3,7:3, 8:4,9:4, 10:5,11:5}
```

**Rule:** every green SUMO phase is even; its yellow is `green + 1`. Switching always
goes `current green → (green+1) yellow → hold YELLOW_STEPS=3 → target green`. Yellow
is never interrupted once started.

> **Defense point:** “through phases 0 and 3” appears everywhere (reward, governor,
> starvation overrides). That’s because only NS-through (0) and EW-through (3) move the
> bulk of traffic and create green waves; left turns (1,2,4,5) are low-volume.

---

## 4. The v6 model architecture, layer by layer

File: `trafix_v6/trafix_v6.py`. Class `TraFixV6`. Constants `OBS_DIM=20`,
`NUM_PHASES=6`, `NUM_JUNCTIONS=5`.

```
obs  [B, T=30, J=5, 20]
  │
  ▼  _TemporalEncoder  (nn.GRU, 1 layer, batch_first)
     reshape to [B*J, 30, 20]; take final hidden state h_n
  │
  ▼  h  [B, J=5, 128]
  │   reshape to [B*J, 128] and build a batched edge_index
  ▼  _GraphEncoder  (GATConv 128→32, heads=4, concat ⇒ 128)
  │
  ▼  g  [B, J=5, 128]
  ▼  _SharedTrunk   Linear(128→128)→ReLU→Linear(128→64)→ReLU
  │
  ▼  t  [B, J=5, 64]
  ├──► actor_heads[j]   5 × Linear(64→6)   ⇒ logits_list: 5 × [B, 6]
  └──► hybrid critic:
         local_critics[j]  5 × Linear(64→1)        ⇒ V_local [B,J,1]
         global_critic     Linear(64→1) on t.mean(dim=1)  ⇒ V_global [B,1,1]
         value = (V_local + V_global).squeeze(-1)   ⇒ [B, J]
```

### Verified parameter counts (loaded the model and counted)

| Component | Module | Params |
|-----------|--------|-------:|
| Temporal encoder | GRU(20→128) | 57,600 |
| Graph encoder | GATConv(128→32×4) | 16,768 |
| Shared trunk | MLP 128→128→64 | 24,768 |
| Actor heads | 5 × Linear(64→6) | 1,950 |
| Local critics | 5 × Linear(64→1) | 325 |
| Global critic | Linear(64→1) | 65 |
| **Total** | | **101,476** |

### Why each block exists

- **GRU (temporal):** traffic is a time series; the 30-step window lets the model see
  whether a queue is growing or draining, not just its instantaneous size. All 5
  junctions are pushed through the GRU in a single batched call (`B*J` rows) — one
  shared GRU, not 5.
- **GATConv (spatial):** the 5 junctions form a physical **chain `0–1–2–3–4`
  (bidirectional)**. GAT lets each junction attend to its neighbours’ states so the
  network can coordinate (e.g. green waves). `heads=4 × head_dim=32 = 128`, concatenated.
  The chain edge index is a registered buffer (`_make_chain_edge_index`); for batches
  >1 it is offset per graph (`_batch_edge_index`).
- **Shared trunk:** projects the 128-dim graph features to a 64-dim representation that
  both actor and critic reuse.
- **5 independent actor heads:** each junction picks its own phase. Outputs are **raw
  logits** (not softmaxed inside the model) — the governor needs raw logits to add/mask.
- **Hybrid critic:** `V_j = V_local_j + V_global`. The local head gives a
  junction-specific baseline; the global head (on the mean-pooled trunk) injects
  network-wide context. This is the v6 change from v5’s global-only critic.

### Model methods

- `forward(obs)` → `(logits_list, value)`.
- `get_action(obs)` → samples (`Categorical(logits)`) — used in training rollouts.
- `evaluate_actions(obs, actions)` → `(log_probs, entropy, value)` — used in the
  PPO update when **no** governor is passed.
- Deployment uses **argmax** (deterministic), not sampling.

---

## 5. The RuleGovernor

File: `trafix_v6/rule_governor.py`. It sits between the actor logits and the final
phase choice and applies **traffic-law constraints by additive masking** (add 0,
add a bonus, or add `-1e9` to forbid). It never edits weights — purely an inference-
and rollout-time wrapper.

### The four rules

| Rule | Type | Mechanism |
|------|------|-----------|
| **Min-green** | hard, stateless | If current phase held `< min_green`, set **every other** phase’s logit to `-1e9` (forces hold). |
| **Max-green** | hard, stateless | If current phase held `> max_green`, set the **current** phase’s logit to `-1e9` (forces switch). |
| **Pressure boost** | soft, stateless | Add `pressure_boost × fraction` to the most-congested movement’s phase, but only if that movement is `> pressure_thresh` of total demand. |
| **Anti-flicker** | soft, **stateful** | If the last two chosen phases differ (A→B), subtract `flicker_penalty` from the logit of going back to A. Needs `update_state()` each step and `reset()` per episode. |

### Min/max green are phase-type dependent (module constants)

```python
MIN_GREEN_THROUGH = 10.0   # phases 0, 3
MIN_GREEN_LEFT    =  8.0   # phases 1, 2, 4, 5
MAX_GREEN_THROUGH = 90.0
MAX_GREEN_LEFT    = 45.0
```

> Note: the `min_green_s` / `max_green_s` constructor args are effectively
> overridden by these per-type constants inside `_hard_mask`. Through phases use
> 10/90, left phases use 8/45, regardless of what you pass to the constructor.

### Production instantiation (`backend/main.py::load_model`)

```python
RuleGovernor(num_junctions=5, num_phases=6,
             min_green_s=10.0, max_green_s=90.0,
             flicker_window=2, flicker_penalty=3.0,
             pressure_boost=1.0, pressure_thresh=0.12)
```

> `pressure_thresh` was lowered from 0.35 → **0.12** so the pressure boost fires for any
> movement holding more than 12 % of total demand (previously needed 35 %). This makes the
> soft boost effective even at low-demand junctions where no single phase exceeds 35 %.

### Three apply variants (why there are three)

- `apply()` — hard mask + pressure + **flicker** (stateful). Used during **rollout
  collection** and during **live inference**.
- `apply_stateless()` — hard mask + pressure only. Safe when you can’t track flicker.
- `apply_stateless_batch()` — same, but vectorised over a minibatch. Used **inside the
  PPO update** (`ppo_update`) so the policy is evaluated under the *same* constraints it
  acted under, without leaking per-step flicker history into the batch.

The governor decodes phase/duration straight from the 20-dim obs: phase =
`argmax(obs[13:19])`, duration = `obs[19] × 120`.

---

## 6. Inference path: telemetry → decision

File: `backend/main.py`, endpoint `POST /telemetry_batch`.

1. **Receive batch** `{step, intersections:[Telemetry×5]}`. Backward-compat: if only
   old `north_count/...` fields are present they’re routed into the `*_through` lanes.
2. **Restart detection:** if `batch.step < _last_batch_step`, clear the rolling
   window and `reset()` the governor (a new SUMO run restarted the step counter).
3. **Build obs list** (`_build_obs_list`) — missing junctions are zero-filled.
4. **Parse** → `node_features = parse_sumo_observations(obs_list)` → `[5, 20]`.
5. **Rolling window:** a `deque(maxlen=30)`. On the very first call it is
   **pre-filled with 30 copies** of the current frame (so the GRU always sees a full
   `T=30`); afterwards one frame is appended per call.
6. **Forward:** `window_tensor [1,30,5,20]` → `logits_list, _ = ai_agent(...)`.
7. **Govern:** `_v6_governor.apply(logits_list, obs_last)` where `obs_last` is the
   last frame of the window.
8. **Softmax → argmax:** `action_probs[j] = softmax(governed_logits[j])`; chosen phase
   = `argmax`; **confidence = max softmax prob** (rounded to 3 dp, shown on dashboard).
9. **Update governor flicker state** with the chosen phases (only if all 5 present).
10. **Persist:** decisions + telemetry are logged to Postgres on a background thread
    (`db.log_step`) if a DB session exists.
11. **Return** `{decisions:[{intersection_id, next_phase, confidence,
    total_vehicles, queue_length}]}`.

### Heuristic fallback

If no weights loaded (`ai_agent is None`), the endpoint uses a pure rule:
hold if `phase_duration < 10` or no demand, else pick `0` (NS) vs `3` (EW) by which
through-demand is larger. This guarantees the lights never freeze even with no model.

---

## 7. The reward function

File: `backend/ai/trafix_v2.py::compute_reward`. Returns a **per-junction `(N,)`
tensor**. Used by Stage-3 PPO and by v2 training. Weights live in the `RewardWeights`
dataclass:

```python
pressure      = -0.30
queue         = -0.25
throughput    =  0.25
fairness      =  0.00     # computed but zero-weighted
phase_penalty = -0.08
wait_penalty  = -0.05
green_wave    =  0.20     # global, added uniformly to all junctions
starvation    = -0.20     # per-movement anti-starvation
clear_bonus   =  0.06     # small positive for any vehicles actually cleared
```

Per junction, the eight local terms:

| Term | Formula | Meaning |
|------|---------|---------|
| pressure | `Σ(12 lanes) / 60` | total demand; penalised |
| queue | `queue_length / 200` | standing queue; penalised |
| throughput | `(prev_total − cur_total)/max(prev_total,1)`, floored at −1 | vehicles cleared; rewarded |
| fairness | `std(12 lanes)/max(mean,1)` | imbalance across lanes (weight 0 ⇒ inert) |
| phase_penalty | `1.0` if phase changed vs last step else 0 | discourages thrashing |
| wait_penalty | `(dur−60)/60` when `dur>60` else 0 | penalise overlong holds |
| starvation | see below | penalise holding a phase while other movements have demand |
| clear_bonus | `min(cleared, 5) / 5` when `cleared > 0` else 0 | absolute clearing reward |

**Starvation term (per-movement, scale-invariant):**

```python
excess         = min(dur / 45.0, 2.0)
unserved_share = (total_demand − demand[current_phase]) / total_demand
starvation     = unserved_share × excess
```

Key properties vs the old NS/EW-only, >30 s-gated version:

- **Active at any demand level** — uses shares, not absolute counts. 1 car in the cross
  direction out of 3 total already gives `unserved_share = 0.67`, so the penalty fires even
  during off-peak periods where the old >30 s gate was the only signal.
- **Per-movement (all 6 phase groups)** — not just NS vs EW. If N-left has cars and the
  junction is stuck on EW-through, the penalty fires for the N-left starvation too.
- **Zero when there is no cross-direction demand** — `unserved_share = 0` when all demand
  is served by the current phase, so it never penalises a junction that is correctly holding.

**Clear-bonus term (low-demand shaping):** a small positive reward for any absolute vehicle
reduction at this junction (capped at 5 vehicles / step). Unlike `throughput`, which is
relative to `prev_total` and collapses to noise with only 1–3 cars, `clear_bonus` gives a
reliable signal whenever even a single car is cleared.

**Green-wave bonus (global):** `_compute_green_wave` rewards adjacent junctions whose
through phases align along the directed edges `[(0,1),(1,2),(1,3),(3,4)]`, but only
when the upstream junction has a platoon (`through_demand ≥ 5`). It adds a bonus that
grows with platoon size and with the downstream queue *draining*. The scalar bonus is
added uniformly to every junction’s reward (`reward += green_wave × gw`).

> **Defense point:** fairness weight is intentionally `0.00` — the term is kept in the
> code (and computed) so it can be re-enabled by changing one number, but it currently
> contributes nothing. If asked “does fairness matter?” the honest answer is *it’s
> wired but disabled*.

---

## 8. GAE and the PPO objective

### GAE (`compute_gae` in `trafix_v2.py`)

```
δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
A_t = δ_t + γλ·A_{t+1}        (accumulated backwards over the rollout)
returns = A + values
```

`γ = 0.99`, `λ = 0.95`. Both advantages and returns are **standardised**
(mean 0, std 1) before the update — but only if `numel > 1` and `std > 0.01` (guards
against degenerate single-step / flat rollouts). Works for both per-junction `[T,J]`
and scalar values.

### PPO update (Stage 3, `stage3_train_ppo.py::ppo_update`)

For each minibatch:

```
log_ratio = clamp(new_logp − old_logp, −2.0, 2.0)     # max_log_ratio
ratio     = exp(log_ratio)
surr1     = ratio · A
surr2     = clamp(ratio, 1−0.2, 1+0.2) · A            # clip_eps=0.2
policy_loss = −min(surr1, surr2).mean()

# clipped value loss
v_clipped  = old_v + clamp(v_new − old_v, −0.2, 0.2)  # value_clip_eps=0.2
value_loss = 0.5 · max((v_new−ret)², (v_clipped−ret)²).mean()

total = policy_loss + 0.25·value_loss − 0.01·entropy  # value_coef=0.25, ent_coef=0.01
```

Extra stabilisers in the loop:

- **KL early-stop:** `approx_kl = 0.5·log_ratio².mean()`; if it exceeds
  `target_kl = 0.015`, the whole update stops (breaks out of the epoch loop).
- **Non-finite guard:** minibatches producing non-finite loss are skipped.
- **Grad clip:** `clip_grad_norm_(…, 0.5)`.
- When a `governor` is passed (it is, in Stage 3), the policy is re-evaluated through
  `governor.apply_stateless_batch` + `evaluate_governed` so the ratio is consistent
  with the governed action distribution.

> The **v2** `train_step`/`compute_ppo_loss` path is simpler (single-sample clip, plain
> MSE value loss, `value_coef=0.25`, `entropy_coef=0.005`) and only matters if you run
> the legacy v2 trainer.

---

## 9. Training pipeline (3 stages + fine-tune)

Orchestrated by `train_v6.py` (`python train_v6.py` runs all 3; `--stage N` runs one).
Each stage uses SUMO via TraCI and the shared `parse_sumo_observations`.

### Stage 1 — GRU pretraining (`trafix_v6/stage1_pretrain_gru.py`)

- **Goal:** give the GRU a useful temporal prior before RL.
- **Task:** supervised **next-step prediction** — a temporary head `Linear(128→12)`
  predicts the next frame’s 12 per-lane counts from the 30-step window. Loss = MSE.
- **Only** `temporal_enc` + the prediction head are trained; **actions are random**
  (`torch.randint(0,6)`), so the GRU learns traffic dynamics, not a policy.
- Defaults: `--episodes 300`, `--lr 1e-3`, Adam + `ReduceLROnPlateau(factor=0.5,
  patience=15, min_lr=1e-5)`, grad-clip 0.5. Saves **best** (lowest mean loss) →
  `checkpoints/stage1_gru.pt` (GRU state dict only).
- `T_WINDOW = 30`. *(The module docstring says “T=10”; the actual constant used is 30
  — see §13.)*

### Stage 2 — GATConv + trunk pretraining (`stage2_pretrain_gatconv.py`)

- **Prereq:** `stage1_gru.pt` must exist (else `sys.exit`).
- Loads the Stage-1 GRU and **freezes it** (`requires_grad=False`).
- **Task:** auxiliary **neighbour-queue prediction** — per junction, a temporary
  `Linear(trunk_out→#neighbours)+Sigmoid` head predicts each chain-neighbour’s total
  queue (obs index 12) at T+1. Loss = **variance-normalised MSE** (`_nmse`, keeps the
  scale stable across traffic intensities). Trains `graph_enc` + `trunk` + temp heads.
- Curriculum: first `--offpeak-episodes 200` are **OFFPEAK-only** (easy), then the full
  curriculum mix.
- Defaults: `--episodes 400`, `--lr 3e-4`. Saves best → `stage2_gatconv.pt` and
  `stage2_trunk.pt`.

### Stage 3 — full PPO (`stage3_train_ppo.py`) ← produces the production model

- **Prereq:** stage1 + stage2 checkpoints (unless `--resume`).
- Loads GRU, GATConv, trunk; actor/critic heads start fresh.
- **Differential learning rates** (one Adam, multiple param groups):

  | Param group | LR |
  |-------------|----|
  | `temporal_enc` (GRU) | 1e-4 |
  | `graph_enc` (GAT) | 2e-4 |
  | `trunk`, `actor_heads`, `local_critics`, `global_critic` | `--lr` = 3e-4 |

  Lower LR on pretrained encoders protects what Stages 1–2 learned.
- **Warm start freeze:** for the first `--freeze-episodes 100`, the GRU and GAT are
  frozen; at episode 100 they unfreeze.
- **Cosine LR decay** every episode from base LR down toward `--lr-min 1e-5`.
- **Entropy annealing:** `entropy_coef` is cosine-annealed from `--entropy-start 0.01`
  → `--entropy-end 0.0005` over the full run. Early episodes explore stochastically;
  late episodes sharpen toward the argmax-mode distribution used at deployment. This
  directly closes the train/deploy mismatch (the model is trained with policies close to
  argmax, not just with a fixed high-entropy policy).
- **Rollout & update:** collect until `rollout_length = 64` steps, then `ppo_update`
  with `ppo_epochs = 4`, `minibatch_size = 64`. The governor is active during both
  collection (`apply`) and update (`apply_stateless_batch`). Reward = `compute_reward`,
  advantages = `compute_gae` (both imported from `trafix_v2`).
- **Episode budget:** `--episodes 2000`, `--max-steps 3600`, `--decision-interval 10`,
  `--warmup 50` (env warm-up before the agent acts).
- **Greedy checkpoint selection:** every `--eval-interval 50` episodes a separate,
  fixed-seed `greedy_eval()` rollout is run: **argmax** (not sampled) with the governor
  ON and starvation overrides OFF — exactly the production inference path. The checkpoint
  with the best greedy reward that also keeps `peak_queue < baseline × (1 + peak_slack)`
  is saved as `trafix_v6_stage3_best.pt`. This prevents the common failure mode of
  choosing a checkpoint that looks good under stochastic sampling but degrades under
  deterministic argmax deployment.
- **Checkpoints:** every 100 episodes → `stage3_ep{N}.pt`; at the end →
  `trafix_v6_final.pt`; best greedy → `trafix_v6_stage3_best.pt`. Each checkpoint
  stores model + optimizer + episode + best_reward.
- Truncated episodes flush the remaining buffer with a proper bootstrap value computed
  from the last observed window (not zeros) — important for correct GAE on
  `max_steps`-cut episodes.

### Fine-tune: argmax collapse fix (`finetune_argmax.py`, optional)

- Starts from `trafix_v6_final.pt`. **LR 1e-5**, GRU+GAT **frozen** (only trunk + actor
  heads move, 27,108 / 101,476 params). Curriculum low-demand-heavy (45 % OFFPEAK).
  400 episodes. **Greedy rollout** (argmax + governor ON) during training, not sampled.
  Entropy cosine-annealed 0.01 → 0.0005. Best model gated by greedy reward **and**
  `peak_queue < baseline × 1.10` (high-traffic regression guard). Saves
  `trafix_v6_argmax_best.pt` and `trafix_v6_argmax_final.pt`.

### Fine-tune: morning-peak (`finetune_morning_peak.py`, optional)

- Starts from `trafix_v6_final.pt`. **LR 1e-5**, GRU+GAT **frozen**. Curriculum
  **70% MORNING_PEAK / 30% OFFPEAK**. 250 episodes. Best model gated by *lowest
  morning-peak queue* **and** *offpeak queue < 0.035* (anti-forgetting guard).

### Evaluation (`eval_stage3.py`)

Loads any stage-3 checkpoint and runs it across SUMO scenarios; reports queue
(`obs[:,12]`) and wait (`obs[:,19]`). `--greedy` uses argmax; `--gui` shows SUMO.

---

## 10. The SUMO environment & scenario generator

### `SumoEnvironment` (`backend/ai/train_v2.py`)

This is the TraCI wrapper every trainer uses (Stage 1/2/3 use it via
`ScenarioEnvironment`). Key behaviours:

- **Start:** launches `sumo`/`sumo-gui` with `--time-to-teleport -1` (teleporting
  **disabled** — forces the model to actually clear lanes rather than have SUMO
  magically remove stuck cars), `--waiting-time-memory 1000`, `--random`, and
  `--seed = base_seed + episode` (each episode a different but reproducible seed).
  Then warms up `warmup_steps` simulation steps before any action.
- **Observation (`get_observations`):** for each TLS it walks the controlled links,
  dedupes by incoming lane, classifies each lane’s **direction** from edge geometry
  (`_classify_edge_direction` compares the lane’s start point to the junction
  position) and **movement type** from the lane index (`{0:right, 1:through, 2:left}`),
  and accumulates `getLastStepVehicleNumber`. `queue_length = min(total*1.5, 200)`.
- **Manual duration tracking:** `_phase_held_since[tls]` records the step a green
  started; `phase_duration = step − held_since`. This is deliberate — calling
  `setPhase()` every decision interval resets SUMO’s own phase timer, so the SUMO-
  reported elapsed time would never exceed ~10 s. Manual tracking gives the true hold
  time the governor/reward depend on.
- **Actions (`apply_actions` / `_advance_transitions`):** model phase → target green;
  if it differs from current, start a 3-step yellow (`YELLOW_STEPS=3`) then switch.
  Yellows are never interrupted. `step()` applies the action then runs
  `decision_interval = 10` sim steps; `done` when SUMO empties or `max_steps` reached.
- **`build_edge_index`:** tries to read the graph from the net file via `sumolib`,
  else falls back to the hard-coded 5-junction topology
  `[[0,1,1,2,1,3,2,4,3,4],[1,0,2,1,3,1,4,2,4,3]]` (the same chain used by the model).

`TrainConfig` (dataclass) centralises all v2/env hyperparameters — see §15.

### `ScenarioGenerator` / `ScenarioEnvironment` (`trafix_v6/scenario_generator.py`)

Generates a fresh `.rou.xml` per episode so the agent never overfits one demand
pattern. **No `traci` import at module load** — SUMO is only touched in
`ScenarioEnvironment.start()`.

Five scenario types and their flow ranges (veh/hr):

| Type | Pattern | Flow |
|------|---------|------|
| OFFPEAK | uniform low flow on all OD pairs | base 200–500 |
| MORNING_PEAK | heavy inbound, light side | main 800–1200 / side 100–300 |
| EVENING_PEAK | heavy outbound, light side | main 800–1200 / side 100–300 |
| INCIDENT | OFFPEAK base, one junction’s fringe blocked for a window | base 200–500, block 50–150 s |
| PULSE | quiet window then a directional burst | burst 600–1000, quiet 20–80 |

**Curriculum** (`_CURRICULUM`, weights = [OFFPEAK, MORNING, EVENING, INCIDENT, PULSE]):

| From episode | OFFPEAK | MORNING | EVENING | INCIDENT | PULSE |
|-------------:|:-------:|:-------:|:-------:|:--------:|:-----:|
| 0 | 0.85 | 0 | 0 | 0 | 0.15 |
| 200 | 0.35 | 0.25 | 0.25 | 0 | 0.15 |
| 500 | 0.20 | 0.25 | 0.25 | 0.15 | 0.15 |
| 800 | 0.15 | 0.20 | 0.20 | 0.25 | 0.20 |

(The list is checked high-threshold-first, so early episodes are easy OFFPEAK-heavy and
difficulty ramps up.) Topology constants (`_JUNCTION_FRINGE_IN`, OD lists) are derived
from `sumo/map.net.xml`; every edge is validated against the net file at construction.

---

## 11. The live runner and its safety overrides

File: `sumo/run_sumo_live.py`. This is the process that actually drives the demo. It is
**not** the trainer — it just samples SUMO, calls the backend, and actuates lights.

Loop (every sim step): advance yellows → run emergency preemption → **every
`DECISION_INTERVAL=10` steps** build a telemetry batch, POST `/telemetry_batch`, and
apply the returned phases through yellow transitions (with `MIN_GREEN_THROUGH=10`,
`MIN_GREEN_LEFT=8`).

It layers **four heuristic safety overrides on top of the AI** (because a learned
policy can starve a movement):

| Override | Constant | Trigger |
|----------|----------|---------|
| Through-phase starvation | `STARVE_LIMIT = 8` | 8 consecutive non-through decisions ⇒ force the higher-demand through phase |
| Direction starvation | `DIRECTION_STARVE_LIMIT = 10` | one through direction monopolises 10 decisions while the other has cars ⇒ force the starved direction |
| Left-turn starvation | `LEFT_STARVE_LIMIT = 15` | a left phase with demand unserved for 15 decisions ⇒ force it |
| Fixed-time fallback | `FALLBACK_THRESHOLD = 3` | backend unreachable 3× in a row ⇒ cycle NS/EW on a 40-step timer so lights never freeze |

(8/10/15 decisions × 10 sim-steps ≈ 80/100/150 simulated seconds.)

**Emergency preemption** is a separate module (`sumo/emergency_preemption.py`,
`EmergencyPreemptionController`) deliberately decoupled from the sim loop so the
detection/actuation layer can later be swapped for CARLA+YOLO without touching the
loop. It is a per-junction state machine (`yellow → all-red → green → return-yellow`)
that overrides the AI for that junction while an emergency vehicle transits, and
reports metrics (transit steps, vehicles waited) which the runner POSTs to
`/emergency_event` for the dashboard.

> **Defense point:** if a reviewer says “the AI starved a left turn,” the honest answer
> is that the *live runner* has explicit overrides for exactly that — the model’s raw
> behaviour is corrected by both the RuleGovernor (in the backend) and these starvation
> guards (in the runner).

---

## 12. "What happens if I change X?" — defense crib sheet

This is the section to study for *“what happens when we change x with y.”*

### Observation / shape parameters — change these and weights break

| Change | Consequence |
|--------|-------------|
| `OBS_DIM` 20 → other | GRU input size changes ⇒ **checkpoint won’t load** (RuntimeError → heuristic fallback). Must retrain from Stage 1. Also breaks `parse_sumo_observations` index layout, the governor’s index constants, and Stage-1’s 12-count target. |
| `NUM_PHASES` 6 → 4 | Actor heads become `64→4`, phase one-hot is 6 bits ⇒ **shape mismatch on load**. Retrain. This is literally the v5→v6 difference. |
| `NUM_JUNCTIONS` 5 | Hard-coded everywhere (heads, critics, chain graph, `state_dict` in backend). Changing it is a structural rewrite, not a knob. |
| `T` window 30 → smaller | Less temporal context; the GRU still runs (T is dynamic), but behaviour drifts from training. The backend pre-fills 30; a different live T than training T degrades quality silently. |
| observation normaliser (junction-relative shares, ÷200, /120 cap 3.0) | These are baked into `parse_sumo_observations` and used by training and inference through the same shared function. Change one only in inference ⇒ the model sees a different distribution than it trained on ⇒ worse decisions, no error raised. Change in `parse_sumo_observations` ⇒ must retrain from Stage 1 because the GRU pretrain used the old distribution. The relative-share approach (`count / max(total, 1)`) is what the current production weights were trained on. |

### Reward weights — change these, retrain, behaviour shifts (no crash)

| Change | Effect |
|--------|--------|
| ↑ `queue` / `pressure` magnitude | More aggressive at draining queues, more switching. |
| `throughput` ↓ | Less incentive to actually clear cars. |
| `fairness` 0 → >0 | Re-enables lane-balance pressure (currently inert). |
| `green_wave` ↑ | Stronger neighbour-phase alignment; can over-coordinate and starve cross traffic. |
| `starvation` more negative | Switches away from a held direction sooner. |
| `phase_penalty`/`wait_penalty` | Trade-off between stability (fewer switches) and responsiveness. |

These only take effect on **re-training** — the deployed `.pt` already encodes the
old weights.

### PPO / training knobs — affect *how* it learns, not the I/O contract

| Change | Effect |
|--------|--------|
| `clip_eps` 0.2 ↑ | Larger policy steps, faster but less stable. |
| `entropy_coef` (now annealed 0.01→0.0005) | Stage 3 cosine-anneals entropy so early training explores and late training sharpens toward argmax. A flat high value keeps entropy high throughout → policy stays diffuse at deployment (argmax collapse risk). A flat low value kills exploration early → poor coverage of scenarios. The anneal is the designed trade-off. |
| `value_coef` 0.25 ↑ | Critic learns faster but can dominate the policy gradient. |
| `target_kl` 0.015 ↑ | Allows bigger updates before early-stop. |
| `gamma` 0.99 ↓ | Shorter horizon, more myopic (less green-wave planning). |
| `gae_lambda` 0.95 | Bias/variance of advantage estimate. |
| `rollout_length` 64 ↑ | More data per update, lower-variance gradients, slower wall-clock. |
| `freeze_episodes` 100 ↓ | Encoders start moving sooner ⇒ risk of wrecking the pretrained prior early. |
| differential LRs (1e-4/2e-4/3e-4) | Raise the encoder LRs and you may overwrite Stages 1–2; that’s the whole reason they’re lower. |

### Governor knobs — change these and behaviour changes *live, instantly* (no retrain)

| Change | Effect |
|--------|--------|
| `MIN_GREEN_THROUGH` 10 / `MIN_GREEN_LEFT` 8 | Longer min-green ⇒ smoother but laggier; shorter ⇒ flicker risk. |
| `MAX_GREEN_THROUGH` 90 / `MAX_GREEN_LEFT` 45 | Cap on how long one phase can hold before a forced switch. |
| `flicker_penalty` 3.0 ↑ | Stronger anti-oscillation; too high freezes the phase. |
| `pressure_boost` 1.0 / `pressure_thresh` 0.12 | Lower threshold ⇒ pressure boost fires more often, biasing toward the busiest movement. At 0.12 (the current production value) the boost fires for any movement with ≥ 12 % of total demand — effective even at very low traffic. The old value was 0.35. |
| `YELLOW_STEPS` 3 | Safety timing; changing it desyncs the yellow handling in env/runner. |

### Runner overrides — change these and you change the safety net, not the model

`STARVE_LIMIT 8`, `DIRECTION_STARVE_LIMIT 10`, `LEFT_STARVE_LIMIT 15`,
`FALLBACK_THRESHOLD 3`, `FALLBACK_CYCLE 40` — all in `run_sumo_live.py`. Loosening them
lets the AI’s raw (possibly starving) behaviour show through; tightening them makes the
heuristics dominate.

---

## 13. Corrections to retired numbers (old MODEL_EXPLANATION.md)

The previous `backend/ai/MODEL_EXPLANATION.md` (now **deleted** — this file replaces
it) was correct on architecture but **had drifted on several training numbers**. They
are recorded here so that anyone who saw the old doc, or an old slide quoting it, is not
caught out. Trust the code (and this doc):

| Item | Old doc said | Actual code |
|------|--------------|-------------|
| `rollout_length` | 256 | **64** (`stage3 --rollout-length`, `TrainConfig`) |
| `value_coef` | 0.5 | **0.25** |
| anti-flicker penalty | −10.0 | **3.0** (governor default & production) |
| Stage-1 window in docstring | “T=10” | constant `T_WINDOW = 30` is what runs |
| Stage-3 description | “buffer 256 steps” | 64 |
| differential LRs / freeze / cosine decay | not mentioned | present (GRU 1e-4, GAT 2e-4, heads 3e-4; freeze 100; cosine→1e-5) |
| `final == ep2000` | “identical” | **confirmed true** (byte-identical weights) |

Also worth stating plainly:

- The v2 model classes (`SpatioTemporalGNN`, `IntersectionCoordinator`,
  `CoordinatedPPOAgent`, `train_step`) were **removed** from `trafix_v2.py`. The file
  now contains only the shared infrastructure: `parse_sumo_observations`,
  `compute_reward`, `compute_gae`, and `RewardWeights`. Don’t look for a v2 model there.
- `SpatioTemporalGNN` (when it existed) contained **no GRU** despite the name — it was a
  2-layer GCN. GRU lives only in **v6** (`_TemporalEncoder`).
- `fairness` weight is **0.0** (term computed but disabled).

---

## 14. File / import map

```
ENTRY POINTS
  baslat.py              one-click: uvicorn main:app + sumo/run_sumo_live.py
  run.py                 uvicorn launcher
  main.py                FastAPI bridge → mounts backend.main.app, serves dashboards

BACKEND (inference)
  backend/main.py        /telemetry_batch endpoint; model selection; governor;
                         rolling window; heuristic fallback; DB logging
  backend/database.py    Postgres persistence (sessions, step_log, emergency_log)
  backend/ai/trafix_v2.py
        ├─ parse_sumo_observations()   ← SHARED 20-dim parser (all versions)
        ├─ NUM_NODE_FEATURES = 20
        ├─ compute_reward(), _compute_green_wave()   ← used by Stage-3 PPO
        ├─ compute_gae()
        └─ RewardWeights (dataclass)
        (v2 model classes SpatioTemporalGNN / IntersectionCoordinator /
         CoordinatedPPOAgent / train_step were removed — file is now infra-only)
  backend/ai/train_v2.py
        ├─ TrainConfig (dataclass)      ← env/v2 hyperparameters
        ├─ SumoEnvironment              ← TraCI wrapper used by ALL trainers
        ├─ build_edge_index()
        └─ MODEL_TO_SUMO_GREEN / SUMO_TO_MODEL_PHASE / LANE_TYPE

MODEL v6 (production)
  trafix_v6/trafix_v6.py       TraFixV6 (GRU→GAT→trunk→actors+hybrid critic)
  trafix_v6/rule_governor.py   RuleGovernor + sample_governed/evaluate_governed
  trafix_v6/scenario_generator.py  ScenarioGenerator + ScenarioEnvironment
  trafix_v6/stage1_pretrain_gru.py     → checkpoints/stage1_gru.pt
  trafix_v6/stage2_pretrain_gatconv.py → checkpoints/stage2_gatconv.pt, stage2_trunk.pt
  trafix_v6/stage3_train_ppo.py        → checkpoints/stage3_ep{N}.pt, trafix_v6_final.pt
  trafix_v6/finetune_morning_peak.py   optional fine-tune
  trafix_v6/eval_stage3.py             evaluation harness
  train_v6.py                          orchestrates stage1→2→3
  trafix_v6/checkpoints/   stage1_gru.pt, stage2_gatconv.pt, stage2_trunk.pt,
                           stage3_ep1000.pt, stage3_ep2000.pt, trafix_v6_final.pt

SUMO (simulation + live actuation)
  sumo/map.net.xml             5-junction, 3-lane network; 5 tlLogic × 12 phases
  sumo/training.sumocfg        net + training_demand.rou.xml
  sumo/demo.sumocfg            live demo config
  sumo/run_sumo_live.py        live loop → POSTs telemetry, applies decisions,
                               starvation/fallback overrides
  sumo/emergency_preemption.py EmergencyPreemptionController (decoupled state machine)

TESTS
  tests/mandatory/             121 tests, run without SUMO (pytest tests/mandatory -v)
  tests/metrics/, tests/analysis/   metric modules + AI-vs-baseline analysis
```

**Import chains worth knowing:**
- The v6 training scripts import `parse_sumo_observations`, `compute_reward`,
  `compute_gae`, `RewardWeights`, `NUM_NODE_FEATURES` **from `trafix_v2`**, and
  `TrainConfig`/`SumoEnvironment` **from `train_v2`** — i.e. v6 *reuses* the v2
  observation+reward+env code. That shared dependency is intentional and is why
  `trafix_v2.py` is still load-bearing even though the v2 *model* is unused.
- `backend/main.py` imports `parse_sumo_observations` from `trafix_v2` regardless of
  model version.

---

## 15. Every number in one table

### Model (TraFixV6)

| Param | Value | Where |
|-------|-------|-------|
| obs_dim | 20 | `OBS_DIM`, `NUM_NODE_FEATURES` |
| num_phases | 6 | `NUM_PHASES` |
| num_junctions | 5 | `NUM_JUNCTIONS` |
| window T | 30 | `_V6_T_WINDOW`, `T_WINDOW` |
| GRU hidden | 128 | `hidden_dim` |
| GAT heads × head_dim | 4 × 32 = 128 | `gat_heads`, `gat_head_dim` |
| trunk | 128 → 64 | `trunk_mid`, `trunk_out` |
| total params | 101,476 | (verified) |

### Production governor (backend/main.py)

| Param | Value |
|-------|-------|
| min_green through / left | 10 s / 8 s |
| max_green through / left | 90 s / 45 s |
| flicker_window | 2 |
| flicker_penalty | 3.0 |
| pressure_boost | 1.0 |
| pressure_thresh | **0.12** |

### Stage-3 PPO (the run that produced production weights)

| Param | Value |
|-------|-------|
| episodes | 2000 |
| max_steps / episode | 3600 |
| decision_interval | 10 sim-steps |
| warmup_steps | 50 |
| rollout_length | 64 |
| ppo_epochs | 4 |
| minibatch_size | 64 |
| gamma / gae_lambda | 0.99 / 0.95 |
| clip_eps / value_clip_eps | 0.2 / 0.2 |
| value_loss_coef | 0.25 |
| entropy_start → entropy_end | 0.01 → 0.0005 (cosine anneal) |
| eval_interval | 50 episodes |
| peak_slack | 0.10 (10 % above baseline peak_queue) |
| target_kl | 0.015 |
| max_log_ratio | 2.0 |
| max_grad_norm | 0.5 |
| LR: GRU / GAT / heads | 1e-4 / 2e-4 / 3e-4 |
| lr_min (cosine floor) | 1e-5 |
| freeze_episodes | 100 |

### Stage 1 / Stage 2 / Fine-tune

| Param | Stage 1 | Stage 2 | Fine-tune |
|-------|---------|---------|-----------|
| episodes | 300 | 400 | 250 |
| lr | 1e-3 | 3e-4 | 1e-5 |
| task | next-step 12-count MSE | neighbour-queue nMSE | PPO (frozen encoders) |
| trains | GRU + pred head | GAT + trunk (+heads) | trunk + actors |
| scheduler | ReduceLROnPlateau(0.5,15) | ReduceLROnPlateau(0.5,15) | — |
| special | random actions | 200 offpeak-only first | 70/30 morning/offpeak, gated save |

### Reward weights (RewardWeights)

| pressure | queue | throughput | fairness | phase | wait | green_wave | starvation | clear_bonus |
|---------:|------:|-----------:|---------:|------:|-----:|-----------:|-----------:|------------:|
| −0.30 | −0.25 | +0.25 | 0.00 | −0.08 | −0.05 | +0.20 | **−0.20** | **+0.06** |

### Live runner (run_sumo_live.py)

| Param | Value |
|-------|-------|
| DECISION_INTERVAL | 10 |
| YELLOW_STEPS / ALL_RED_STEPS | 3 / 2 |
| MIN_GREEN_THROUGH / LEFT | 10 / 8 |
| STARVE_LIMIT (through) | 8 |
| DIRECTION_STARVE_LIMIT | 10 |
| LEFT_STARVE_LIMIT | 15 |
| FALLBACK_THRESHOLD / CYCLE | 3 / 40 |
| MAX_EMERGENCY_GREEN | 60 |
| EMERGENCY_STUCK_SPEED / PUSH_AFTER | 0.1 m/s / 10 steps |

### SUMO launch flags (training)

`--step-length 1.0`, `--waiting-time-memory 1000`, `--time-to-teleport -1`
(teleport disabled), `--random`, `--seed = seed + episode`.

---

## 16. Design rationale — why each structure (and the alternatives we rejected)

This section is the *“why did you build it this way and not the obvious other way”*
companion to the rest of the document. Each entry follows the same shape: **what it is →
what it’s normally used for → why we chose it (or rejected the standard alternative).**
These are the questions a reviewer is most likely to push on.

### 16.1 Why reinforcement learning at all (and not fixed-time / actuated control)

- **What the alternatives are.** Classic traffic control is **fixed-time** (Webster’s
  formula picks a cycle from historical flows) or **actuated** (loop detectors extend a
  green while cars keep arriving). Commercial adaptive systems (SCATS, SCOOT) tune
  cycle/split/offset with hand-built optimisers.
- **Why RL here.** Those methods optimise one junction at a time against *average*
  conditions and need expert calibration per intersection. We want a single policy that
  (a) reacts to the live state, (b) **coordinates across 5 junctions**, and (c)
  generalises across demand patterns it wasn’t hand-tuned for. RL learns that
  coordination directly from the reward instead of us encoding it.
- **Honesty for the review.** We keep a **fixed-time fallback** (live runner) and a
  **heuristic fallback** (backend) precisely because RL is not trustworthy alone — the
  classic methods are the safety floor.

### 16.2 Why PPO (and not DQN / A2C / SAC)

- **Normal use.** PPO is the default on-policy actor-critic for discrete control —
  stable, few moving parts, robust to hyperparameters.
- **Why we chose it.** Our action space is **discrete** (6 phases) and we want a
  **stochastic policy** during training (exploration via entropy) but a deterministic
  one at deploy (argmax). PPO’s clipped objective gives stable updates without the
  replay-buffer brittleness and overestimation issues of **DQN**, and without the tuning
  pain of **SAC** (which is built for continuous actions anyway). **A2C/A3C** is the
  closest cousin but PPO’s trust-region clip + KL early-stop (`target_kl=0.015`) makes it
  far more stable for a small network on noisy traffic rollouts.
- **Trade-off we accept.** PPO is **on-policy** → less sample-efficient than DQN. We pay
  for that with a cheap simulator (SUMO) and 2000 episodes, which is affordable.

### 16.3 Why a GRU temporal encoder (and not LSTM / Transformer / frame-stacking / none)

- **What it does.** Summarises the last **30 seconds** of each junction into a 128-dim
  state, so the policy sees *trend* (is the queue filling or draining?) not just the
  instantaneous snapshot.
- **Normal alternatives.** **Frame-stacking** (concatenate the last N frames into the
  MLP) is the cheap option; **LSTM** is the heavier recurrent option; a **Transformer**
  encoder is the modern, expressive option.
- **Why GRU.** A GRU has **fewer parameters than an LSTM** (one fewer gate) and trains
  faster — and at 30 steps we don’t need the LSTM’s extra long-memory capacity. A
  Transformer is overkill for a 30-step, 20-feature sequence and would dominate the
  parameter budget (the whole model is only ~101k params). Frame-stacking would blow up
  the input width and throw away the sequential inductive bias. The GRU is the
  sweet-spot for “small, fast, captures short-horizon dynamics.”
- **Defense point.** This is the v5→v6 lineage: v2’s class is even *named*
  `SpatioTemporalGNN` but the GRU was removed there; v6 is where the temporal encoder is
  real and load-bearing.

### 16.4 Why a GATConv spatial graph (and not GCN / no graph / a different topology)

- **What it does.** Lets each junction **attend to its physical neighbours** so the
  policy can coordinate (green waves, not fighting each other). The 5 junctions are a
  **bidirectional chain `0–1–2–3–4`** matching the real road layout.
- **Normal alternatives.** **No graph** → 5 fully independent controllers that can’t
  coordinate. **GCN** (what v2 used) → fixed, degree-normalised neighbour averaging.
  **GAT** → *learned* attention weights over neighbours.
- **Why GAT over GCN.** GCN weights every neighbour equally (after normalisation); GAT
  **learns how much each neighbour matters** for the current state — e.g. a congested
  upstream junction should influence me more than an empty one. For coordination that
  data-dependent weighting is exactly what we want. (4 heads × 32 = 128 keeps it cheap.)
- **Why a chain, not all-to-all.** The physical network *is* a chain; connecting
  non-adjacent junctions would inject spurious coordination and add params for no signal.
  The edge index is a fixed buffer, not learned.

### 16.5 Why per-junction actor heads + a hybrid critic (CTDE)

- **What it is.** **Five independent actor heads** (each junction chooses its own phase)
  but a **shared encoder** and a **hybrid critic** `V_j = V_local_j + V_global`. This is
  the standard **CTDE** pattern — *centralised training, decentralised execution*.
- **Normal alternatives.** A **single joint policy** outputting one action over the
  product space (6⁵ = 7776 combos — explodes, won’t learn). Or **fully independent
  agents** with no shared critic (each sees only itself — can’t learn coordination, and
  the value target is non-stationary because the other agents keep changing).
- **Why hybrid critic.** A **global-only** critic (what v5 used) gives every junction the
  same value and washes out which junction is actually responsible for a good/bad
  outcome → high-variance credit assignment. A **local-only** critic ignores network
  effects. `local + global` gives each junction a junction-specific baseline *plus*
  network context — lower-variance advantages without a separate module per junction.
  This is the headline v5→v6 change.

### 16.6 Why 3-stage pretraining (and not end-to-end PPO from scratch)

- **What it is.** Stage 1 pretrains the GRU (predict next traffic state), Stage 2
  pretrains GAT+trunk (predict neighbour queues), Stage 3 runs PPO with those weights
  loaded and the encoders frozen for the first 100 episodes.
- **Normal alternative.** Just initialise everything randomly and run PPO end-to-end.
- **Why we pretrain.** PPO’s reward signal is **sparse and noisy**; asking a randomly
  initialised GRU+GAT to *simultaneously* learn “what traffic dynamics look like” **and**
  “a good control policy” is unstable and slow. Self-supervised pretraining (next-step /
  neighbour-queue prediction needs **no reward, no labels** — the simulator provides the
  targets) gives the encoders a useful representation *before* the fragile RL starts, so
  PPO only has to learn the policy head on top. The **freeze-then-unfreeze** and
  **differential learning rates** (GRU 1e-4 < GAT 2e-4 < heads 3e-4) exist to *protect*
  that pretrained prior from being wiped out by early, high-variance policy gradients.
- **Trade-off.** More moving parts and three scripts to run, versus much more stable
  convergence. For a 5-agent coordinated problem that stability is worth it.

### 16.7 Why the RuleGovernor (action masking, not just reward shaping)

- **What it is.** Hard + soft constraints applied to the **logits** before a phase is
  chosen: forbid (`-1e9`) illegal switches (min/max-green), discourage flicker, nudge
  toward pressure.
- **Normal alternative.** **Reward shaping** — penalise illegal behaviour in the reward
  and hope the policy learns to avoid it (“soft” constraints).
- **Why mask instead.** Reward shaping makes a rule *probable*, not *guaranteed* — during
  exploration the agent will still occasionally switch a light after 1 second, which in
  traffic is **unsafe and physically illegal**. Masking is the **safe-RL “shielding”**
  pattern: the constraint is enforced by construction, every step, training and live.
  It also shrinks the effective action space, which *speeds up* learning. We apply the
  governor consistently in **both** rollout collection and the PPO update
  (`apply_stateless_batch`) so the policy is optimised under the same constraints it acts
  under — otherwise the importance ratio would be biased.
- **Why min/max-green are phase-type-specific** (through 10/90 s, left 8/45 s): left
  turns are low-volume and shouldn’t hold the intersection as long as a through
  movement; the asymmetry is a domain fact, not a tuned knob.

### 16.8 Why a dense, multi-term reward (and the fairness term we left off)

- **What it is.** Reward = weighted sum of pressure, queue, throughput, phase-stability,
  wait, starvation (local) + green-wave (global). Fairness is present but **weight 0**.
- **Normal alternatives.** A **sparse** reward (e.g. only “total waiting time at episode
  end”) is cleaner and less biased — but gives almost no per-step learning signal, so the
  agent needs vastly more data and may never discover coordination.
- **Why dense.** Each term is a hand-designed hint that makes credit assignment tractable
  on short rollouts: queue/pressure say “smaller queues are better,” throughput rewards
  actually clearing cars, green-wave rewards inter-junction alignment, starvation/wait
  prevent abandoning a direction. The risk of dense shaping is **reward hacking** (the
  agent games a term); we mitigate that by keeping each weight small and letting the
  governor/overrides catch the pathological cases.
- **The fairness term specifically.** It computes the **coefficient of variation**
  (`std/mean`) of the 12 per-lane counts — high when queues are lopsided. Its *normal*
  use is to stop a throughput-maximiser from **permanently starving the quiet
  direction**. We set its weight to **0.00** because that exact failure mode is already
  covered three more directly: the **`starvation` reward term** (−0.20, per-movement
  share-based), the governor’s **max-green** hard switch, and the live runner’s
  **starvation overrides** (`STARVE_LIMIT`/`DIRECTION_STARVE_LIMIT`/`LEFT_STARVE_LIMIT`).
  On top of that, lane occupancy at a junction is *naturally* uneven (through lanes have
  more vehicles than left-turn lanes),
  so the CV signal is noisy and would fight throughput for little gain. It’s kept as a
  one-number toggle so the choice is reversible, but it is deliberately off.

### 16.9 Why act every 10 s with a min-green (and not every simulation second)

- **What it is.** `decision_interval = 10`: the agent observes and (maybe) switches once
  per 10 sim-seconds; min-green then enforces a floor on how often a real switch happens.
- **Why.** Per-second control would let the policy oscillate a light faster than cars can
  physically react — unsafe, and it shortens the effective horizon the GRU must reason
  over. Real signal controllers also operate on multi-second decision cadences. Ten
  seconds matches the min-green-through and keeps the rollout length manageable.

### 16.10 Why GAE (and not Monte-Carlo returns or one-step TD)

- **What it is.** Generalised Advantage Estimation with `γ=0.99, λ=0.95`.
- **Normal extremes.** **Monte-Carlo** (`λ=1`) is unbiased but high-variance — full-
  episode returns swing wildly with traffic randomness. **One-step TD** (`λ=0`) is
  low-variance but biased by an immature critic.
- **Why GAE.** `λ=0.95` interpolates: most of the variance reduction of bootstrapping,
  most of the low bias of long returns. We then **standardise** advantages per batch so
  the policy-gradient scale is stable across light and heavy traffic episodes.

### 16.11 Why teleport is disabled and why a scenario curriculum

- **`--time-to-teleport -1`.** By default SUMO *teleports* vehicles that have been stuck
  too long, silently removing the consequences of a bad policy. Disabling it **forces the
  model to actually serve every lane** — a gridlock stays a gridlock in the reward, which
  is the honest training signal we want.
- **Curriculum (`scenario_generator.py`).** Random demand from day one means early,
  untrained policies face incidents and pulses they can’t handle and get no learnable
  signal. The curriculum starts **85% OFFPEAK** and ramps difficulty (peaks, incidents,
  pulses) as the episode count grows — standard curriculum learning, so the agent masters
  easy coordination before hard, bursty scenarios. Fresh `.rou.xml` per episode prevents
  overfitting one fixed demand file.

---

*Generated for code-review preparation. All values read from source on branch `main`
(`87d872c`) and verified against the on-disk checkpoints and a live model import.*
