# TraFix — Presentation Source Pack (for an LLM with NO repo access)

> **HOW TO USE THIS FILE.** Paste this whole file into Claude (or any LLM) and ask it to
> generate a slide deck (Google Slides / PowerPoint / Marp / reveal.js — your choice). It is
> self-contained: every code excerpt the slides need is embedded below in fenced blocks
> marked `CODE SCREENSHOT`. The model does NOT need the repository.
>
> **Instructions to the generating model:**
> - Build one deck of ~40–55 slides, grouped into the 6 sections below (one section per
>   presenter). Start with a title slide and an agenda; end each section with a 3-question
>   "likely Q&A" slide using the questions provided.
> - Render every block tagged **`CODE SCREENSHOT`** as a syntax-highlighted code image/figure
>   (monospace, dark theme, filename shown as a caption). Keep the code verbatim.
> - Turn each **Bullets** list into the slide body; put **Notes** into speaker notes.
> - Anywhere you see `{{FILL AFTER RETRAIN: ...}}`, render a clearly-styled YELLOW placeholder
>   box reading "TO BE UPDATED AFTER TRAINING" with the described content — do NOT invent
>   numbers.
> - Use a consistent template (see "SLIDE TEMPLATES"). Keep ≤ 6 bullets/slide.

---

## PROJECT ONE-LINER

TraFix is a reinforcement-learning traffic-signal controller for a **5-junction** corridor
in **SUMO**. The production model **TraFix v6** reads a 30-second window of 20 features per
junction, runs **GRU (time) → GATConv (space) → shared MLP → 5 actor heads + hybrid critic**,
has its logits constrained by a **RuleGovernor**, and emits one of **6 phases** per junction
every **10 simulated seconds**. A FastAPI backend serves decisions; a live SUMO runner
actuates lights and adds safety overrides; an ambulance-preemption state machine can override
the AI; a 121-test mandatory suite plus Type-1/Type-2/unseen scenario tests evaluate it.

**Model lineage:** v2 (GCN + multi-head attention — *removed, no trained weights*) → v5
(GRU+GAT, 4 phases — *deleted earlier*) → **v6 (the only running model)**.

---

## SLIDE TEMPLATES (use these layouts)

- **T1 Title:** project name, subtitle, team, date.
- **T2 Section divider:** big section number + presenter topic.
- **T3 Bulleted content:** title + ≤6 bullets + optional small diagram.
- **T4 Code screenshot:** title + the code image (caption = filename) + 1–2 takeaway bullets.
- **T5 Diagram:** ASCII/box diagram rendered as a figure (e.g. the data-flow pipeline).
- **T6 Q&A:** "They might ask…" with 3 questions + one-line answers.

Reusable data-flow diagram (render as a figure wherever helpful):

```
SUMO telemetry ─► parse_sumo_observations ─► [30-frame rolling window]
   ─► TraFixV6 (GRU ─► GATConv ─► trunk ─► 5 actor heads + hybrid critic)
   ─► RuleGovernor.apply (min/max-green, pressure@0.12, anti-flicker)
   ─► argmax ─► (live runner: starvation overrides safety net) ─► yellow ─► green
```

---

# SECTION 1 — TraFix AI model & training (Presenter 1)

### Slide: The v6 architecture
**Bullets:**
- Per-junction actor-critic; ~101,476 parameters total.
- GRU encodes the 30-step time window; GATConv coordinates the 5 junctions over a fixed
  bidirectional **chain** graph 0–1–2–3–4; shared trunk → 5 actor heads + hybrid critic.
- Hybrid critic: `V_j = V_local_j + V_global` (CTDE — centralised training, decentralised
  execution).
**Notes:** GRU not LSTM (fewer params, 30 steps don't need long memory); GAT not GCN (learned
neighbour attention); chain not all-to-all (matches the physical road).

```python
# CODE SCREENSHOT — trafix_v6/trafix_v6.py  (forward pass)
def _encode(self, obs, edge_index):
    B = obs.shape[0]
    h = self.temporal_enc(obs)                      # GRU: [B,T,J,20] -> [B,J,128]
    h_flat = h.reshape(B * NUM_JUNCTIONS, self.hidden_dim)
    g = self.graph_enc(h_flat, self._batch_edge_index(edge_index, B))  # GATConv
    g = g.reshape(B, NUM_JUNCTIONS, -1)
    t = self.trunk(g)                               # [B,J,64]
    local_v  = torch.stack([self.local_critics[j](t[:, j, :])
                            for j in range(NUM_JUNCTIONS)], dim=1)
    global_v = self.global_critic(t.mean(dim=1)).unsqueeze(1)
    v = (local_v + global_v).squeeze(-1)            # hybrid critic -> [B,J]
    return t, v
```

### Slide: The reward function (why the AI behaves the way it does)
**Bullets:**
- Dense per-junction reward: pressure, queue, throughput, green-wave, phase/wait penalties.
- **NEW** per-movement **anti-starvation** term: penalises demand sitting in *unserved*
  movements, scaled by how long the current phase held — active even at low demand, zero when
  a movement has no demand.
- Small `clear_bonus` rewards actually clearing cars (non-zero gradient at low flow).
**Notes:** This is the root-cause fix for the argmax collapse — it teaches "serve every
movement that has demand," which the live-runner overrides used to do externally.

```python
# CODE SCREENSHOT — backend/ai/trafix_v2.py  (RewardWeights + anti-starvation)
@dataclass
class RewardWeights:
    pressure=-0.30; queue=-0.25; throughput=0.25; fairness=0.00
    phase_penalty=-0.08; wait_penalty=-0.05; green_wave=0.20
    starvation=-0.20      # per-movement anti-starvation (active at low demand)
    clear_bonus=0.06      # low-demand shaping: reward clearing any car

# inside compute_reward, per junction:
group_demand = [N_thru+S_thru, N_left, S_left, E_thru+W_thru, E_left, W_left]
total_dem = sum(group_demand)
starvation = 0.0
if total_dem > 0:                       # zero when no demand
    excess = min(dur / 45.0, 2.0)       # grows with how long current phase held
    unserved_share = (total_dem - group_demand[phase]) / total_dem
    starvation = unserved_share * excess
```

### Slide: 3-stage training pipeline
**Bullets:**
- Stage 1: pretrain the GRU by next-step traffic prediction (self-supervised, no reward).
- Stage 2: pretrain GATConv + trunk by neighbour-queue prediction (GRU frozen).
- Stage 3: full **PPO** with pretrained encoders, frozen 100 episodes then unfrozen;
  differential LRs (GRU 1e-4 < GAT 2e-4 < heads 3e-4), cosine decay.
- Orchestrated by `train_v6.py`; the governor is active during both rollout and update.
**Notes:** Pretraining gives stable encoders before the noisy RL signal arrives.

### Slide: PPO + GAE
```python
# CODE SCREENSHOT — trafix_v6/stage3_train_ppo.py  (clipped PPO objective)
log_ratio = (new_lp - mb_old).clamp(-max_log_ratio, max_log_ratio)
ratio = torch.exp(log_ratio)
surr1 = ratio * mb_adv
surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
policy_loss = -torch.min(surr1, surr2).mean()
# clipped value loss + entropy bonus, with KL early-stop (target_kl=0.015)
total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_loss
```
**Bullets:** GAE γ=0.99, λ=0.95, advantages standardised; clip 0.2; KL early-stop; grad-clip 0.5.

### Slide: The argmax problem and the fix
**Bullets:**
- Trained by **sampling** (+entropy) but deployed by **argmax** → the greedy policy collapsed:
  4/5 junctions locked onto one phase 83–97% of the time, starving the rest.
- Fix = **(reward)** anti-starvation + **(training)** entropy annealing toward ~0 +
  **(selection)** pick best checkpoint by a **greedy/argmax** rollout (governor on, overrides
  off) + **(governor)** `pressure_thresh` 0.35→0.12.
- Proven by a committed diagnostic that prints the per-junction phase distribution.

```text
CODE SCREENSHOT — diagnostic BEFORE baseline (argmax + governor, no overrides), type1_low
  junction   NS-thru  N-left  S-left  EW-thru  E-left  W-left
  J0            0%      2%      6%      2%      90%      0%   <- LOCKED
  J1            3%      0%      0%      0%      95%      2%   <- LOCKED
  J2            1%      0%      0%      2%      97%      0%   <- LOCKED
  J3           46%     20%      9%     22%      1%      2%
  J4            0%      0%      4%     13%      0%      83%   <- LOCKED
  OVERALL: COLLAPSED (a junction locks >70%)
```
{{FILL AFTER RETRAIN: AFTER table from the new checkpoint — should show no junction >~60-70%
and no starved movement; plus new best_reward.}}

```python
# CODE SCREENSHOT — trafix_v6/finetune_argmax.py  (greedy/argmax checkpoint selection)
masked = governor.apply(logits_list, obs_last)         # governor ON
actions_1d = torch.stack([torch.argmax(l, -1).reshape(()) for l in masked])  # NO overrides
# entropy_coef cosine-annealed start->end; "best" gated on greedy reward + a
# high-traffic no-regression guard. Writes trafix_v6_argmax_best.pt (never the final).
```

### Slide (T6): Likely Q&A — Presenter 1
- *Why GRU not Transformer?* 30-step, 20-feature sequence — GRU is the small/fast sweet spot.
- *Why did argmax collapse?* Optimised + checkpoint-selected under sampling; greedy mode never
  measured. Flat low-demand reward made argmax arbitrary & sticky.
- *Does the reward force serving empty roads?* No — the anti-starvation term is zero when a
  movement has no demand.

---

# SECTION 2 — Inference path & RuleGovernor (Presenter 2)

### Slide: Telemetry → decision
**Bullets:**
- `POST /telemetry_batch`: build obs → parse → push to a 30-frame rolling window (pre-filled
  on first call) → model forward → governor → softmax → **argmax** → return phases +
  confidence.
- Restart detection clears the window + governor when SUMO's step counter resets.
- Heuristic fallback keeps lights moving if no weights load.

```python
# CODE SCREENSHOT — backend/main.py  (v6 inference branch)
window_tensor = torch.stack(list(_v6_window)).unsqueeze(0)   # [1,T,5,20]
logits_list, _ = ai_agent(window_tensor)
obs_last = window_tensor[0, -1]
if _v6_governor is not None:
    logits_list = _v6_governor.apply(logits_list, obs_last)  # constrain logits
action_probs = torch.stack([torch.softmax(l, -1).squeeze(0) for l in logits_list], dim=0)
# next_phase = argmax(action_probs[j]); confidence = max softmax prob
```

### Slide: The 20-dim observation contract
**Bullets:**
- 12 per-lane counts (÷15 left/right, ÷30 through) + total queue/200 + 6-bit phase one-hot +
  duration (÷120, capped 3.0).
- Shared by every model version and every trainer — single source of truth.

```python
# CODE SCREENSHOT — backend/ai/trafix_v2.py  (parse_sumo_observations)
for key in _LANE_KEYS:           row.append(o.get(key, 0) / _NORM[key])   # 0-11
row.append(o.get("queue_length", 0.0) / 200.0)                           # 12
one_hot = [0.0]*6; one_hot[int(o.get("current_phase",0)) % 6] = 1.0
row.extend(one_hot)                                                       # 13-18
row.append(min(o.get("phase_duration",0.0)/120.0, 3.0))                  # 19
```

### Slide: The RuleGovernor (safe-RL shielding)
**Bullets:**
- Adds to the logits (0, a bonus, or −1e9 to forbid): **min-green** (block early switch),
  **max-green** (force a switch), **pressure boost** (nudge the busiest movement), **anti-
  flicker** (penalise A→B→A).
- Phase-type aware: through 10/90 s, left 8/45 s. `pressure_thresh` lowered **0.35 → 0.12**.
- Applied in BOTH training and inference so the policy acts under the constraints it ships with.

```python
# CODE SCREENSHOT — trafix_v6/rule_governor.py  (hard mask)
if duration < min_green:        # min-green: forbid every other phase
    for p in range(self.num_phases):
        if p != phase: mask[p] = _NEG_INF
elif duration > max_green:      # max-green: forbid staying
    mask[phase] = _NEG_INF
```

### Slide (T6): Likely Q&A — Presenter 2
- *What stops an illegal switch?* The governor masks it by construction every step (training
  + live) — shielding, not just reward shaping.
- *No model loaded?* Heuristic fallback: hold under min-green/no-demand, else pick NS vs EW by
  through-demand. Lights never freeze.
- *Why argmax not sampling at deploy?* Deterministic, reproducible decisions for a live system.

---

# SECTION 3 — Live runner, actuation & the safety net (Presenter 3)

### Slide: The live loop
**Bullets:**
- `run_sumo_live.py` every sim step: advance yellows → emergency preemption → every
  `DECISION_INTERVAL=10` steps POST telemetry, apply returned phases via 3-step yellow.
- Fixed-time fallback if the backend is unreachable 3× — lights never freeze.

### Slide: Starvation overrides (the safety net)
**Bullets:**
- Three heuristic guards layered on the AI: through-starvation (8), direction-starvation (10),
  left-turn starvation (15) decisions.
- Present here AND mirrored in the test runner so tests measure the controller that ships.
- The argmax fix's goal: make these **rarely-triggered** instead of load-bearing.

```python
# CODE SCREENSHOT — tests/utils/sumo_runner.py  (left-turn starvation override)
for lp in (1, 2, 4, 5):
    dsl[tls_id][lp] = 0 if model_phase == lp else dsl[tls_id][lp] + 1
if model_phase not in (1, 2, 4, 5):
    # force the most-overdue left that still has demand
    if best_lp is not None and dsl[tls_id][best_lp] >= _LEFT_STARVE_LIMIT:
        model_phase = best_lp
```
{{FILL AFTER RETRAIN: an override-trigger count from a live/test run showing they now fire rarely.}}

### Slide: The SUMO network
**Bullets:** 5 junctions, 3 lanes/approach, 5 `tlLogic` × 12 phases (6 green + 6 yellow);
teleporting disabled (`--time-to-teleport -1`) so gridlock stays visible in the reward/metrics.

### Slide (T6): Likely Q&A — Presenter 3
- *Your AI starved a left turn?* The runner has explicit STARVE/DIRECTION/LEFT guards; the
  retrained policy reduces how often they fire.
- *Why disable teleporting?* So the model can't "cheat" by having SUMO remove stuck cars.
- *How do lights never freeze?* Fixed-time fallback after 3 unreachable backend calls.

---

# SECTION 4 — Test framework (Presenter 4)

### Slide: The 121 mandatory tests
**Bullets:** FR + NFR + unit tests (model, observation, RuleGovernor, preemption), database,
and metrics-module tests — run without SUMO, must stay green.

### Slide: External scenario tests
**Bullets:** Type-1 (low/medium/high), Type-2 (morning/evening/incident/pulse), and *unseen*
scenarios (oscillating, supersaturation, stadium-exit, tidal-ramp…). Baseline vs AI.

### Slide: Runner integrity checks
**Bullets:**
- Gridlock detection (mean speed < 0.5 m/s for 300 s).
- Teleport sanity (authoritative total from `statistics.xml`, must be 0).
- Captures vehicles still in the network at sim end (no survivorship bias).

```python
# CODE SCREENSHOT — tests/utils/sumo_runner.py  (gridlock + teleport checks)
if mean_spd < _GRIDLOCK_SPEED_THRESHOLD:
    _low_speed_streak += 1
    if _low_speed_streak >= _GRIDLOCK_DURATION_S: _gridlock_detected = True
teleports = int(ET.parse(stats).getroot().find("teleports").get("total", 0))  # must be 0
```
{{FILL AFTER RETRAIN: Type-1/Type-2 pass summary with the new model — 121 green, 0 teleports, 0 gridlock.}}

### Slide (T6): Likely Q&A — Presenter 4
- *Why 121 "mandatory"?* They encode the FRs/NFRs and unit contracts; CI gate.
- *How is gridlock detected?* Sustained near-zero network speed.
- *Why teleports from statistics.xml?* It's the authoritative total; summary.xml is cumulative
  per-step and overcounts.

---

# SECTION 5 — Metrics & results/figures (Presenter 5)

### Slide: The 12 metrics
**Bullets:** throughput & completion %, queue (halting **and** crawl-aware "slow"), travel
time, teleports, junction fairness, waiting time, time-loss, network speed, stops/veh,
emissions, fuel, pollutants.

### Slide: Honest accounting (why these definitions)
**Bullets:**
- Travel time & throughput fold in cars **still in the network** at sim end → no survivorship
  bias for a controller that gridlocks.
- Crawl-aware queue (< 5 km/h) not just halting (< 0.1 m/s) → fair on continuous crawl.

```python
# CODE SCREENSHOT — tests/metrics/throughput.py  (completion accounting)
not_inserted = max(0, loaded - inserted)            # never even entered the net
cars_not_completed = still_running + not_inserted   # demand the controller failed to serve
completion_rate = arrived / total_demand
```
{{FILL AFTER RETRAIN: AFTER results tables/figures — throughput, completion %, slow-queue,
travel time at low/medium/high vs the baseline and the old model.}}

### Slide (T6): Likely Q&A — Presenter 5
- *Survivorship bias?* Unfinished cars are recorded via TraCI and folded into the averages.
- *Why crawl-aware queue?* Halting-only undercounts slow-moving congestion.
- *How is completion % computed?* arrived / total demand (incl. not-inserted).

---

# SECTION 6 — Frontend, ambulance, persistence & launchers (Presenter 6)

### Slide: Frontends
**Bullets:** `dashboard.html` (live junction states, confidences), `emergency.html`
(ambulance view), `index.html` (entry) — poll the FastAPI backend.

### Slide: Ambulance / emergency preemption
**Bullets:**
- Per-junction state machine: `None → yellow → all-red → green → return-yellow`, decoupled
  from the AI loop (so detection could later be CARLA+YOLO).
- Overrides the AI for that junction while the ambulance transits; reports transit metrics to
  `/emergency_event`.

```python
# CODE SCREENSHOT — sumo/emergency_preemption.py  (per-junction state machine)
class EmergencyPreemptionController:
    # stages: None -> "yellow" -> "allred" -> "green" -> "return_yellow" -> None
    def __init__(self, traci_mod, tls_ids, log, *, yellow_steps=3, all_red_steps=2,
                 max_emergency_green=60, stuck_speed=0.1, min_green_through=10):
        ...
    def is_active(self, tls_id):    # AI/fallback skip this junction while active
        return self.stage.get(tls_id) is not None
```

### Slide: Persistence & launch
**Bullets:**
- `backend/database.py`: optional Postgres logging of sessions / per-step decisions /
  emergency events.
- `baslat.py` one-click: starts `uvicorn main:app` + `sumo/run_sumo_live.py`.

### Slide (T6): Likely Q&A — Presenter 6
- *How does preemption work / why decoupled?* Independent state machine via TraCI; swappable
  detection layer; overrides the AI per junction.
- *What is persisted?* Sessions, step decisions+telemetry, emergency events (if a DB exists).
- *How to run the demo?* `python baslat.py` (GUI) / `--no-gui`.

---

## APPENDIX — numbers to refresh AFTER training (single source of truth)
{{FILL AFTER RETRAIN: new checkpoint path & best_reward · diagnostic AFTER tables (type1_low,
type1_medium) · high-traffic before/after (throughput, completion %, slow-queue, travel time)
· override-trigger frequency · Type-1/Type-2 regression summary. All other content above is
final.}}
