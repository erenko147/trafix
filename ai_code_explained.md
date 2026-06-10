# ai_code_explained.md — The AI Code, Explained From Zero

> **Who this is for.** Someone who knows little or nothing about machine learning or
> traffic simulation and wants to understand **every piece of AI code in this project** —
> what each file is for, and what each function inside it actually does and why.
>
> **How this differs from the other docs.** `master_ai.md` is the dense reference ("what is
> the model and why"), `system_explained.md` is the data-flow ("how the pieces talk"), and
> `code_reference.md` is a flat file:line index. **This file is the teaching version**: it
> starts with a plain-language dictionary of every term, then explains each file and
> function in order, building from the ground up.
>
> **Read it in order the first time.** Part 1 (terminology) makes Parts 3–6 readable. If a
> sentence later uses a word like "logits," "GAE," or "phase," it was defined in Part 1.
>
> **A note on "current code".** A few things changed recently and this doc reflects the
> code as it is now (not the older description in `master_ai.md`): the observation's 12 lane
> features are now **demand shares** (not raw counts); the reward has a new **per-movement
> anti-starvation** term and a **clear bonus**; the governor's `pressure_thresh` is **0.12**
> in every place it's constructed; the **legacy v2 model was deleted** (so `trafix_v2.py` is
> now only shared observation/reward/advantage code); and there is a new training script
> `finetune_argmax.py`. Where the *deployed weights* (`trafix_v6_final.pt`) still predate a
> change, that is called out.

---

## Table of Contents

- [Part 1 — Terminology, in plain language](#part-1--terminology-in-plain-language)
  - [1A. Traffic & simulation words](#1a-traffic--simulation-words)
  - [1B. Machine-learning & reinforcement-learning words](#1b-machine-learning--reinforcement-learning-words)
- [Part 2 — The 60-second picture of what the AI does](#part-2--the-60-second-picture-of-what-the-ai-does)
- [Part 3 — The decision-time AI (the model, the governor, the senses)](#part-3--the-decision-time-ai)
  - [3.1 `trafix_v6/trafix_v6.py` — the neural network](#31-trafix_v6trafix_v6py--the-neural-network)
  - [3.2 `backend/ai/trafix_v2.py` — observation, reward, advantage](#32-backendaitrafix_v2py--observation-reward-advantage)
  - [3.3 `trafix_v6/rule_governor.py` — the rule governor](#33-trafix_v6rule_governorpy--the-rule-governor)
- [Part 4 — The training world (environment & scenarios)](#part-4--the-training-world)
  - [4.1 `backend/ai/train_v2.py` — the SUMO environment](#41-backendaitrain_v2py--the-sumo-environment)
  - [4.2 `trafix_v6/scenario_generator.py` — making traffic to train on](#42-trafix_v6scenario_generatorpy--making-traffic-to-train-on)
- [Part 5 — The training scripts (how the brain is taught)](#part-5--the-training-scripts)
  - [5.1 `train_v6.py` — the orchestrator](#51-train_v6py--the-orchestrator)
  - [5.2 `trafix_v6/stage1_pretrain_gru.py`](#52-trafix_v6stage1_pretrain_grupy)
  - [5.3 `trafix_v6/stage2_pretrain_gatconv.py`](#53-trafix_v6stage2_pretrain_gatconvpy)
  - [5.4 `trafix_v6/stage3_train_ppo.py`](#54-trafix_v6stage3_train_ppopy)
  - [5.5 `trafix_v6/finetune_argmax.py`](#55-trafix_v6finetune_argmaxpy)
  - [5.6 `trafix_v6/finetune_morning_peak.py`](#56-trafix_v6finetune_morning_peakpy)
  - [5.7 `trafix_v6/eval_stage3.py`](#57-trafix_v6eval_stage3py)
- [Part 6 — Running the AI for real (serving & actuation)](#part-6--running-the-ai-for-real)
  - [6.1 `backend/main.py` — the inference server](#61-backendmainpy--the-inference-server)
  - [6.2 `sumo/run_sumo_live.py` — the live driver](#62-sumorun_sumo_livepy--the-live-driver)
  - [6.3 `sumo/emergency_preemption.py` — ambulance override](#63-sumoemergency_preemptionpy--ambulance-override)
  - [6.4 Launchers & plumbing: `baslat.py`, `main.py`, `run.py`, `backend/database.py`](#64-launchers--plumbing)

---

# Part 1 — Terminology, in plain language

You don't need to memorize these. Skim them once, then refer back. Each term is defined
the way it's actually used *in this project*.

## 1A. Traffic & simulation words

- **SUMO** — "Simulation of Urban MObility," a free traffic simulator. It moves virtual
  cars along roads, obeys traffic lights, and lets us measure things like queues and
  travel time. Everything the AI is trained and tested on happens inside SUMO.
- **TraCI** — "Traffic Control Interface." The live remote-control API for SUMO: a Python
  program can step the simulation one second at a time, *read* what's happening (how many
  cars on a lane), and *write* commands (set this traffic light to green). All our code
  talks to SUMO through TraCI.
- **Junction / intersection** — a crossroads where roads meet and a traffic light controls
  who goes. This project has **5 junctions**, named J0–J4 (sometimes shown as K1–K5).
- **Lane / edge** — a road in SUMO is an **edge**; each edge has one or more **lanes**.
  An "incoming lane" at a junction is one that feeds cars *into* that junction.
- **Movement** — the direction a car wants to go through a junction: **through** (straight),
  **left**, or **right**. In this map the lane index encodes it: lane 0 = right, lane 1 =
  through, lane 2 = left.
- **Phase** — one setting of a traffic light: which movements get green right now. This
  project's AI uses **6 phases** (called "model phases"):
  `0 = NS-through, 1 = N-left, 2 = S-left, 3 = EW-through, 4 = E-left, 5 = W-left`.
  (SUMO internally uses 12 phases — 6 greens, each followed by a yellow — but the AI only
  speaks the 6-phase language; code maps between them.)
- **Green / yellow / red** — standard light colors. A **yellow** is always inserted for a
  few seconds between two different greens so cars can stop safely; the AI never skips it.
- **Min-green / max-green** — the minimum time a green must stay on (so lights don't flicker)
  and the maximum it may stay on (so one direction can't hog the junction forever).
- **Queue** — cars waiting (stopped or crawling) at a junction. Lower is better.
- **Throughput** — how many cars complete their trips per hour. Higher is better.
- **Phase duration / hold time** — how long the current phase has been green. The code
  tracks this itself, because SUMO's own timer resets every time we command a phase.
- **Decision interval** — how often the AI is allowed to act: **every 10 simulated seconds**.
- **Telemetry** — the bundle of numbers describing the current state of all junctions (lane
  counts, queues, current phase, hold time) that gets sent to the AI each decision.
- **Route file (`.rou.xml`)** — a file telling SUMO which cars appear when and where they
  go. Different route files = different traffic patterns (light, rush hour, an incident…).
- **Net file (`.net.xml`)** — the road map: junctions, edges, lanes, and traffic-light logic.
- **Teleport** — SUMO's emergency hack: if a car is stuck too long (or collides) it can be
  "teleported" out of the jam. We **disable** time-based teleporting so the simulation
  honestly shows congestion.

## 1B. Machine-learning & reinforcement-learning words

- **Model / neural network** — a big mathematical function with millions of tunable numbers
  ("weights") that turns an input (the traffic state) into an output (which phase to pick).
  Our model is called **TraFix v6**.
- **Reinforcement learning (RL)** — teaching a model by *trial and error*: it acts, gets a
  **reward** (a score), and adjusts its weights to earn more reward over time. No one tells
  it the "right answer"; it discovers good behavior from the reward.
- **Agent / policy** — the decision-maker. The **policy** is the rule "given this state,
  choose this action." Training = improving the policy.
- **PPO (Proximal Policy Optimization)** — the specific RL algorithm used here. It's a
  popular, stable method for improving a policy in small, safe steps so training doesn't
  blow up. Details below (clipping, KL, advantage) are just the knobs that keep it stable.
- **Observation (obs)** — the numbers the model sees about one junction. Here it's a vector
  of **20 numbers** (see §3.2). The model reads all 5 junctions at once.
- **Action** — what the model outputs: for each junction, one of the 6 phases.
- **Logits** — the raw, unnormalized scores the model gives each of the 6 phases. Bigger =
  the model prefers that phase. They can be any number (e.g. -3.1, 0.7, 5.2).
- **Softmax** — a function that turns logits into **probabilities** that add up to 1
  (e.g. logits → `[0.7, 0.05, …]`). Lets us read the model's preference as percentages.
- **Argmax** — "pick the single highest." `argmax` of the logits = the model's top choice,
  deterministically. **This is what we use when actually running the AI.**
- **Sampling** — instead of always taking the top choice, roll a weighted die using the
  softmax probabilities. Used **during training** so the agent explores alternatives.
  (Training samples; deployment uses argmax — a distinction that matters a lot here.)
- **Confidence** — the softmax probability of the chosen phase (e.g. 0.82). Just for display.
- **Actor / critic (actor-critic)** — two jobs the network does at once. The **actor**
  picks actions (outputs logits). The **critic** estimates "how good is this state?" (a
  value number) — used only during training to judge whether an action did better than
  expected.
- **Value** — the critic's estimate of future reward from a state. Not used at deployment.
- **Reward** — the score the agent earns each step, hand-designed from traffic goals (short
  queues good, clearing cars good, starving a direction bad…). The model is trained to
  maximize the long-run sum of rewards. See §3.2.
- **Advantage** — "how much better than expected was this action?" = actual outcome minus
  the critic's prediction. Positive advantage → do more of that action.
- **GAE (Generalized Advantage Estimation)** — a careful recipe for computing the advantage
  that balances bias and noise. `gamma` (0.99) = how far into the future to care; `lambda`
  (0.95) = the bias/noise trade-off. Just think "the math that scores each action."
- **Entropy** — a measure of how *undecided* the policy is (high entropy = spread out over
  many phases, low = confident in one). An **entropy bonus** during training nudges the
  policy to stay exploratory so it doesn't lock in too early.
- **GRU (Gated Recurrent Unit)** — a type of network layer that reads a **sequence over
  time** and summarizes it. We feed it the last 30 seconds of a junction so the model can
  tell "is this queue growing or shrinking?", not just the instantaneous snapshot.
- **GAT / GATConv (Graph Attention Network)** — a layer that lets each junction **look at
  its neighbors** and decide how much each neighbor matters. This is how 5 junctions
  *coordinate* (e.g. set up "green waves") instead of acting blindly alone.
- **Graph / edge index** — the junctions are arranged as a **chain** `0–1–2–3–4`; the
  "edge index" is just the list of which junctions are neighbors. GAT uses it.
- **MLP (multilayer perceptron) / Linear layer / ReLU** — the plain building blocks of a
  network. A **Linear** layer multiplies inputs by weights; **ReLU** is a simple "keep
  positives, zero out negatives" function that lets the network learn non-straight-line
  patterns. An **MLP** is a few Linear+ReLU layers stacked.
- **Checkpoint (`.pt` file)** — a saved snapshot of the model's weights. The production one
  is `trafix_v6/checkpoints/trafix_v6_final.pt`.
- **Rollout** — a stretch of simulation where the agent acts and we record (state, action,
  reward, value) at each step, to learn from afterward.
- **Episode** — one full simulation run used for training (here up to 3600 sim-seconds).
- **Pretraining / fine-tuning** — *Pretraining* teaches part of the network a useful skill
  before the hard RL starts. *Fine-tuning* gently adjusts an already-trained model for a
  specific goal without retraining from scratch.
- **Governor (RuleGovernor)** — a hand-written safety layer that edits the model's logits
  *before* a phase is chosen, to enforce traffic laws (min/max green, no flicker) and nudge
  toward congested directions. It's not learned; it's rules. See §3.3.
- **Masking** — forcing the model not to choose something by setting that option's logit to
  a hugely negative number (`-1e9`), so after softmax its probability is ~0. The governor
  "masks" illegal phases.
- **Starvation** — when one movement never gets a green and its cars pile up forever. A
  known failure mode of the raw model under argmax; the live runner has overrides to prevent
  it, and the reward now penalizes it directly.

---

# Part 2 — The 60-second picture of what the AI does

Every 10 simulated seconds, for all 5 junctions at once:

1. **Sense** — read each junction's traffic into a 20-number observation (§3.2).
2. **Remember** — keep the last 30 seconds of observations (a GRU reads this).
3. **Coordinate** — let junctions look at their neighbors (a GAT does this).
4. **Decide** — output 6 logits per junction (the actor).
5. **Constrain** — the governor edits those logits to obey traffic law (§3.3).
6. **Choose** — take the argmax → one phase per junction.
7. **Act** — switch the SUMO lights toward those phases (through a safe yellow).

Training (Part 5) repeats this millions of times in simulation, scoring each decision with
the reward (§3.2) and nudging the weights via PPO to earn more reward. Serving (Part 6)
loads the trained weights and does steps 1–7 for real.

```
   SUMO  ──TraCI──►  observation (20 nums × 5 junctions)
                           │
                  GRU (time) → GAT (neighbors) → shared MLP
                           │
                    actor heads → 6 logits per junction
                           │
                    RuleGovernor (edit logits: laws + nudges)
                           │
                    argmax → 1 phase per junction
                           │
   SUMO  ◄──TraCI──  set lights (green via yellow)
```

---

# Part 3 — The decision-time AI

These three files are the heart of the AI: the network that decides, the code that turns
SUMO into numbers and scores decisions, and the rule layer that keeps decisions legal.

## 3.1 `trafix_v6/trafix_v6.py` — the neural network

**Purpose.** Defines **TraFix v6**, the actor-critic network that maps the traffic state to
a phase choice. This is "the brain." It's ~101,000 weights.

**Top-level constants**
- `OBS_DIM = 20` — each junction is described by 20 numbers.
- `NUM_PHASES = 6` — six possible greens.
- `NUM_JUNCTIONS = 5` — five intersections.

**`_make_chain_edge_index(n)`** — Builds the neighbor list for the graph: the 5 junctions
form a line `0–1–2–3–4`, connected both ways. Returns the pairs of (junction, neighbor) the
GAT layer needs. Stored once as a fixed buffer; never learned.

**`class _TemporalEncoder` (the GRU / "memory over time")**
- *Purpose:* compress the last 30 seconds of one junction into a single 128-number summary
  that captures trend (filling vs draining), not just the current instant.
- **`__init__`** — creates one GRU shared by all junctions.
- **`forward(obs)`** — input is `[batch, 30 timesteps, 5 junctions, 20 features]`. It
  reshapes so all junctions are processed in one efficient batch, runs the GRU, and returns
  the final hidden summary `[batch, 5 junctions, 128]`.

**`class _GraphEncoder` (the GAT / "look at neighbors")**
- *Purpose:* let each junction blend in information from its chain neighbors so the network
  can coordinate.
- **`__init__`** — creates a `GATConv` with 4 attention "heads" of 32 each (= 128 out).
  Multiple heads = multiple independent ways of weighing neighbors.
- **`forward(x, edge_index)`** — runs the attention over the chain graph; each junction's
  128-number vector now reflects its neighbors' states.

**`class _SharedTrunk` (the shared MLP)**
- *Purpose:* squeeze the 128 graph features down to a compact 64-number representation that
  both the actor and the critic reuse.
- **`forward(x)`** — two `Linear → ReLU` layers: 128 → 128 → 64.

**`class TraFixV6` (the full model)** — wires the three pieces above together and adds the
decision heads.
- **`__init__`** — builds the temporal encoder, the graph encoder, the trunk, then:
  - `actor_heads`: 5 small `Linear(64→6)` layers — one per junction — that output the 6
    phase logits for that junction.
  - `local_critics`: 5 `Linear(64→1)` — each junction's own value estimate.
  - `global_critic`: one `Linear(64→1)` on the *average* of all junctions — a network-wide
    value estimate. The final value per junction is `local + global` (a "hybrid critic":
    junction-specific baseline plus shared context).
- **`_encode(obs, edge_index)`** — the internal pipeline: temporal → graph → trunk, then
  computes the hybrid critic value. Returns the 64-number trunk features and the value.
- **`_batch_edge_index(edge_index, batch_size)`** — a plumbing helper so the graph works
  when several states are processed at once (offsets neighbor indices per item).
- **`forward(obs)`** — *the main entry point.* Runs `_encode`, then the 5 actor heads.
  Returns `(logits_list, value)` where `logits_list` is 5 tensors of 6 logits each. This is
  what both training and serving call.
- **`get_action(obs)`** — used **in training rollouts**: runs `forward`, then **samples** a
  phase per junction (the exploratory die-roll) and returns the actions, their
  log-probabilities, and the value.
- **`evaluate_actions(obs, actions)`** — used **in the PPO update**: given states and the
  actions that were taken, recompute their log-probabilities, the policy entropy, and the
  value. PPO needs these to figure out how to adjust the weights.
- **`__repr__`** — prints a human-readable summary of the architecture and parameter count.

> **Key point for later:** the model itself only *samples* in `get_action` (training).
> "Deployment uses argmax" is implemented by the callers (the backend and test runner),
> not inside the model.

## 3.2 `backend/ai/trafix_v2.py` — observation, reward, advantage

**Purpose.** The **shared "senses and scoring"** code used by the model, every training
script, and the live backend. (Despite the `_v2` name, the old v2 *model* that lived here
was deleted; this file is now purely the observation parser, the reward function, and the
GAE advantage math.) Keeping it in one place guarantees training and serving see the data
the same way.

**`parse_sumo_observations(obs_list, device)`** — *the single source of truth for the
20-number observation.* Takes a list of per-junction dictionaries (lane counts, queue,
current phase, hold time) and returns a `[5, 20]` number grid. The 20 numbers are:
- **[0–11] the 12 lane "demand shares".** Each lane's car count is divided by the total of
  all 12 lanes at that junction. So "1 car out of 5" and "4 cars out of 20" both read as
  0.20. *Why shares?* Because absolute counts are tiny at low traffic (0–3 cars) and the
  model couldn't tell phases apart there; shares stay meaningful at any traffic level. (This
  is a recent change — `master_ai.md` still describes the old raw-count version.)
- **[12] queue ÷ 200** — total queue scaled into roughly 0–1.
- **[13–18] phase one-hot** — six 0/1 flags marking which phase is currently green.
- **[19] hold time ÷ 120**, capped at 3.0 — how long the current phase has been green, so a
  6-minute hold still looks different from a 2-minute hold.

**`class RewardWeights`** — a small settings bundle holding the weight (importance) of each
reward term. Current values and meaning:
- `pressure = -0.30` — penalize total demand sitting at the junction.
- `queue = -0.25` — penalize standing queue.
- `throughput = +0.25` — reward clearing cars (relative reduction).
- `fairness = 0.00` — a "lane imbalance" term, kept but **switched off** (it was noisy; the
  starvation term below does the anti-starvation job better).
- `phase_penalty = -0.08` — small penalty for changing phase (discourages thrashing).
- `wait_penalty = -0.05` — penalize holding one phase too long (> 60 s).
- `green_wave = +0.20` — reward neighboring junctions aligning their through-phases.
- `starvation = -0.20` — **the per-movement anti-starvation penalty** (new; see below).
- `clear_bonus = +0.06` — **a small reward for clearing any car at all** (new; low-demand
  shaping).

**`_intersection_total(o)`** — helper: total of all 12 lane counts at one junction.

**`_compute_green_wave(cur, prev)`** — computes the global "green wave" bonus: along the
directed neighbor edges `[(0,1),(1,2),(1,3),(3,4)]`, reward cases where an upstream junction
is running a through-phase **with a real platoon of cars** (≥ 5) and the downstream junction
is aligned and its queue is draining. Returns one number added equally to every junction.

**`compute_reward(current_obs, previous_obs, previous_actions, current_actions, weights)`**
— *the scoring function.* For each junction it computes the seven local terms and combines
them with the weights, then adds the global green-wave bonus. Returns one reward number per
junction. The two notable recent terms:
- **Per-movement anti-starvation** (term 7). It looks at how much demand is sitting in the
  movements that are **not** currently being served, as a *share* of total demand, scaled by
  how long the current phase has been held. Properties that matter: it's **active even at
  low traffic** (it's share-based), it covers **all six** movements (not just NS-vs-EW), and
  it's **zero when an unserved movement has no cars** (so it never punishes skipping an empty
  road). In plain terms: *"if cars are waiting on a movement you're ignoring, you lose
  points, more so the longer you ignore them."* This teaches the model, during training, to
  do what the live safety-overrides do by force.
- **Clear bonus** (term 8). A small positive reward for the absolute number of cars removed
  from the junction this step (capped). At very low traffic the regular `throughput` term
  goes flat/noisy; this keeps a gentle "clearing cars is good" signal alive.

> **Important:** changing reward weights only affects behavior **after retraining**. The
> currently deployed `trafix_v6_final.pt` was trained before these new terms, so they take
> effect only once a model is retrained/fine-tuned with them (see §5.5).

**`compute_gae(rewards, values, next_value, gamma, lam)`** — *the advantage math.* Given the
per-step rewards and the critic's value estimates over a rollout, it computes the
**advantage** (how much better than expected each action was) and the **returns** (targets
for the critic), using GAE with `gamma=0.99`, `lambda=0.95`. It also standardizes them
(mean 0, spread 1) so PPO's updates stay stable. PPO consumes these.

## 3.3 `trafix_v6/rule_governor.py` — the rule governor

**Purpose.** A **hand-written safety/sanity layer** that sits between the model's raw logits
and the final phase choice. It can't be learned away — it enforces traffic law every single
step, in both training and serving, by **adding numbers to the logits** (zero = no change,
a bonus = encourage, `-1e9` = forbid). The model proposes; the governor vetoes/nudges.

**Index constants (`_IDX_*`)** — name the 20 observation slots so the code can read, e.g.,
"north-through share" or "current phase" without magic numbers.

**Min/max-green constants** — `MIN_GREEN_THROUGH=10s`, `MIN_GREEN_LEFT=8s`,
`MAX_GREEN_THROUGH=90s`, `MAX_GREEN_LEFT=45s`. Through-phases (the high-volume ones) may hold
longer; left turns less.

**`_decode_obs(obs_j)`** — reads one junction's current phase (from the one-hot) and its hold
time in seconds (from slot 19 × 120) out of the observation.

**`class RuleGovernor`** — holds the rule settings and a short memory of recent phases (for
the anti-flicker rule).
- **`__init__`** — stores min/max-green, the anti-flicker window/penalty, and the pressure
  boost/threshold. *Production builds it with `pressure_thresh=0.12`* (recently lowered from
  0.35 so the pressure nudge fires even at balanced low traffic — note the constructor's
  *default* is still 0.35, but every caller passes 0.12).
- **`reset()`** — clear the recent-phase memory; called at the start of each episode / when
  SUMO restarts.
- **`update_state(actions_1d)`** — record the phases just chosen, so the anti-flicker rule
  can see the last couple of choices.
- **`_hard_mask(phase, duration)`** — the **legal-timing rule**. If the current phase has
  been held less than its min-green, forbid switching (set every *other* phase to `-1e9`,
  forcing a hold). If it's been held past its max-green, forbid *staying* (set the current
  phase to `-1e9`, forcing a switch). Otherwise do nothing.
- **`_pressure_bonus(obs_j)`** — the **congestion nudge**. Finds the movement group with the
  largest demand share; if it exceeds `pressure_thresh`, add a small bonus to that phase's
  logit so the model leans toward serving the busiest movement. (Because observations are now
  shares, it reads slots 0–11 directly.)
- **`_flicker_penalty(j)`** — the **anti-oscillation rule**. If the junction just went A→B,
  subtract a penalty from going *back* to A, so it doesn't ping-pong A→B→A→B.
- **`apply(logits_list, obs_last)`** — the **full** governor: hard mask + pressure bonus +
  (stateful) anti-flicker. Used during live serving and training rollouts.
- **`apply_stateless(logits_list, obs_last)`** — hard mask + pressure only (no flicker
  memory). Used where tracking flicker history isn't safe.
- **`apply_stateless_batch(logits_list, obs_last_batch)`** — same as above but for a whole
  minibatch at once; used **inside the PPO update** so the policy is evaluated under the same
  constraints it acted under.

**`sample_governed(masked_logits)`** — given governed logits, **sample** one phase per
junction (the exploratory die-roll) and return the phases plus their log-probabilities. Used
in training.

**`evaluate_governed(masked_logits, actions_batch)`** — recompute log-probabilities and
entropy of already-taken actions under the governed distribution. Used in the PPO update so
the math is consistent with the governor.

> **At deployment we take `argmax` of the governed logits, not `sample_governed`** — the
> backend and test runner do that. Why this matters is the subject of the AI-malfunction
> analysis; the short version is that the greedy (argmax) policy can collapse onto one phase
> per junction unless trained for it, which is why the live runner also has starvation
> overrides and the reward now has the anti-starvation term.

---

# Part 4 — The training world

To learn by trial and error, the agent needs a *world* to act in and *traffic* to face.
These two files provide both.

## 4.1 `backend/ai/train_v2.py` — the SUMO environment

**Purpose.** Wraps SUMO/TraCI into a clean "environment" object that every training script
drives: start a simulation, read observations, apply the agent's phases (with safe yellows),
step forward, and report metrics. (Like `trafix_v2.py`, the dead v2 *trainer* that used to
live here was removed; what remains is the environment plus shared phase-mapping tables.)

**Phase-map tables**
- `MODEL_TO_SUMO_GREEN` — converts a model phase (0–5) to SUMO's green index (0,2,4,6,8,10).
- `SUMO_TO_MODEL_PHASE` — the reverse (SUMO's 12 phases folded back to the model's 6).
- `LANE_TYPE` — lane index → movement (`0=right, 1=through, 2=left`).

**`class TrainConfig`** — a settings bundle (a dataclass) holding all environment/training
knobs: which SUMO config, seed, decision interval (10), warmup steps (50), max steps (3600),
rollout length (64), number of actions (6), etc. Scripts create one and tweak fields.

**`class SumoEnvironment`** — the environment.
- **`__init__(cfg)`** — stores the config; doesn't start SUMO yet.
- **`start(episode)`** — launches SUMO (headless or GUI) with the right flags
  (teleporting disabled, a per-episode random seed so each episode is different but
  reproducible), then runs the warmup steps before the agent is allowed to act.
- **`close()`** — shuts SUMO down cleanly.
- **`is_running()`** — whether SUMO is still alive.
- **`get_observations()`** — *the senses.* For each junction it walks the controlled lanes,
  figures out each lane's compass direction (from geometry) and movement (from lane index),
  counts the cars, computes the queue, and reads the current phase and the manually-tracked
  hold time. Returns the list of per-junction dicts that `parse_sumo_observations` expects.
- **`_classify_edge_direction(edge_id, jx, jy)`** — geometry helper: compares a lane's start
  point to the junction center to label it north/south/east/west.
- **`apply_actions(actions)`** — takes the agent's chosen model phases and begins switching
  the SUMO lights toward them; if a phase actually changes it starts a 3-step **yellow**
  first (never an abrupt green-to-green).
- **`_advance_transitions()`** — counts down running yellows and flips to the target green
  when a yellow finishes.
- **`step(actions)`** — the main loop body: apply the actions, then advance the simulation by
  one decision interval (10 sim-seconds), and report the new observations plus whether the
  episode is done (SUMO emptied or max steps reached).
- **`get_metrics()`** — queue/wait/throughput numbers for logging during training.

## 4.2 `trafix_v6/scenario_generator.py` — making traffic to train on

**Purpose.** Generates a **fresh traffic pattern (route file) for each training episode** so
the agent sees endless variety and can't memorize one fixed demand. Also wraps the
environment so scripts can inject these route files.

**`class ScenarioType` (an Enum)** — the five traffic patterns:
`OFFPEAK` (light, uniform), `MORNING_PEAK` (heavy inbound), `EVENING_PEAK` (heavy outbound),
`INCIDENT` (a road blocked for a while), `PULSE` (a quiet period then a sudden burst).

**`class ScenarioGenerator`**
- **`__init__`** — remembers the road map and output folder; sets up a seeded random source.
- **`generate(scenario_type, episode)`** — builds the chosen pattern for that episode and
  writes a `.rou.xml`, returning its path.
- **`sample(episode)`** — pick a scenario via the curriculum (below) and generate it.
- **`curriculum_schedule(episode)`** — implements **curriculum learning**: early episodes are
  mostly easy OFFPEAK, and harder scenarios (peaks, incidents, pulses) are mixed in as
  training progresses, so the agent masters easy coordination before hard cases.
- **`summary(...)`** — a human-readable description of a generated scenario.
- **`_gen_offpeak / _gen_morning_peak / _gen_evening_peak / _gen_incident / _gen_pulse`** —
  the five builders that emit the actual car flows for each pattern.
- **`_write_rou_xml(out_path, flows)`** — serializes the flows into SUMO's route-file format.
- **`_check_edge / _parse_net_edges / _make_rng`** — validate that every road referenced
  actually exists in the net file, and create the per-episode seeded randomness.

**`class ScenarioEnvironment`** — a thin wrapper around `SumoEnvironment` that (a) delays
importing SUMO until you actually start it, and (b) lets a script swap in a new route file
between episodes. Its `set_route_file`, `start`, `close`, and `__getattr__` (which forwards
everything else to the underlying environment) are plumbing. `_import_sumo_env` is the lazy
importer.

---

# Part 5 — The training scripts

These are run *offline* to produce the weights. You don't need them to serve the model; you
need them to *create or improve* it. They all reuse Parts 3–4.

## 5.1 `train_v6.py` — the orchestrator

**Purpose.** The "run everything" launcher for the original 3-stage training. Running
`python train_v6.py` does stage 1 → 2 → 3 in order; `--stage N` runs just one.
- **`_header(title)` / `_run(cmd, label)`** — pretty console banners and a subprocess
  wrapper that runs each stage script and checks it succeeded.
- **`stage1 / stage2 / stage3(args)`** — invoke the three stage scripts with the right
  arguments.
- **`main()`** — parse arguments and run the requested stage(s).

## 5.2 `trafix_v6/stage1_pretrain_gru.py`

**Purpose.** **Stage 1 — teach the GRU about traffic dynamics before any RL.** It's a
self-supervised warm-up: the GRU learns to *predict the next traffic state* from the last 30
seconds, with **no reward and random actions**. This gives the time-encoder a useful sense of
"how traffic evolves" before the fragile RL begins.
- **`make_env_config(...)`** — build the `TrainConfig` for this stage.
- **`train(args)`** — the loop: run episodes with random phases, feed the GRU the 30-second
  window, have a small temporary head predict the next frame's lane counts, and minimize the
  prediction error (MSE). Saves the best GRU weights to `checkpoints/stage1_gru.pt`.
- **`parse_args()`** — command-line options (episodes, learning rate, …).

## 5.3 `trafix_v6/stage2_pretrain_gatconv.py`

**Purpose.** **Stage 2 — teach the GAT + trunk to reason about neighbors**, still without
RL. It loads and **freezes** the Stage-1 GRU, then trains the graph layers on an auxiliary
task: predict each neighbor's queue. This warms up the coordination machinery.
- **`_nmse(pred, target)`** — a "variance-normalized" error measure that stays well-scaled
  across light and heavy traffic.
- **`make_env_config(...)`** — stage config.
- **`train(args)`** — freeze the GRU, train GAT+trunk (plus temporary prediction heads) to
  predict neighbor queues; saves `stage2_gatconv.pt` and `stage2_trunk.pt`.
- **`parse_args()`** — options.

## 5.4 `trafix_v6/stage3_train_ppo.py`

**Purpose.** **Stage 3 — the real reinforcement learning** that produces the production
model. It loads the pretrained GRU/GAT/trunk, attaches fresh actor/critic heads, and runs
**PPO** against the reward in `trafix_v2.py`. This file also now contains the machinery for
the "make argmax good" fixes (entropy annealing + greedy selection).
- **`_entropy_coef(episode, total, start, end)`** — **entropy annealing**: smoothly lowers
  the exploration bonus from a higher start to a near-zero end over training, so the policy
  starts exploratory and gradually **sharpens into a confident, well-defined choice** (which
  is what argmax then reads).
- **`greedy_eval(model, env, generator, governor, device, …)`** — evaluates the model using
  **argmax** (the deployment behavior) with the governor on but **no starvation overrides**,
  and reports metrics including a "worst_lock" number (how badly any junction is collapsing
  onto a single phase). Used to pick the best checkpoint *by how good the greedy policy is*,
  not by how good the random-sampled policy is.
- **`class RolloutBuffer`** — stores the (state, action, log-prob, reward, value) tuples
  collected during a rollout. `add` appends one step, `clear` empties it, `__len__` reports
  how many steps are stored, used to decide when to run an update.
- **`ppo_update(model, optimizer, buffer, next_value, …, governor)`** — *the learning step.*
  It computes advantages (via `compute_gae`), then for several epochs over minibatches it:
  re-evaluates the actions under the current (governed) policy, forms PPO's **clipped**
  objective (so the policy only moves a little), adds a **clipped value loss** for the critic
  and the **entropy bonus**, checks a **KL early-stop** (bail out if the policy moved too
  far), and takes a gradient step. This is the inner engine of all the PPO scripts.
- **`save_checkpoint(...)`** — writes model + optimizer + episode + best score to a `.pt`.
- **`train(args)`** — the full Stage-3 loop: warm-start (briefly freeze the pretrained
  encoders), differential learning rates (smaller for pretrained parts), cosine learning-rate
  decay, collect rollouts of length 64, call `ppo_update`, periodically `greedy_eval`, and
  save the best/periodic checkpoints (`stage3_ep{N}.pt`, `trafix_v6_final.pt`).
- **`parse_args()`** — the many training knobs (episodes, clip, KL target, entropy schedule,
  learning rates, …).

## 5.5 `trafix_v6/finetune_argmax.py`

**Purpose.** **The targeted fix for the "argmax collapse" problem** (the bundle of fixes
called Steps 1–3 in the fix plan), as a *fine-tune* of the existing production checkpoint
rather than a from-scratch retrain. It freezes the GRU+GAT and gently retrains the trunk +
heads with: the new anti-starvation reward (Step 1, from `trafix_v2.py`), entropy annealing +
**greedy/argmax checkpoint selection** (Step 2), and a **low-demand-heavy curriculum**
(Step 3) so the policy is actually shaped where it used to collapse. It writes **new**
checkpoints and never overwrites `trafix_v6_final.pt` (promotion is a manual, validated step).
- **`_TRAIN_MIX`** — the training scenario mix, OFFPEAK-heavy (45%) but keeping peaks so high
  traffic isn't forgotten.
- **`_EVAL_SET`** — a fixed, reproducible set of scenarios for the greedy evaluation.
- **`_sample_scenario_type()`** — randomly pick a training scenario per the mix.
- **`_entropy_coef(...)`** — cosine entropy anneal (same idea as Stage 3).
- **`greedy_eval(...)`** — argmax rollout (governor on, overrides off) used to select the
  best checkpoint; also reports `worst_lock` (collapse metric) and `peak_queue` (a
  high-traffic guard so a fix at low traffic can't silently wreck high traffic).
- **`finetune(args)`** — the loop: load the production model, freeze encoders, train trunk +
  heads with PPO under the new reward and annealed entropy on the low-demand-heavy
  curriculum, periodically run `greedy_eval`, and save the best greedy checkpoint **only if**
  it doesn't regress peak traffic. Saves `trafix_v6_argmax_best.pt` / `_final.pt`.
- **`parse_args()`** — options, including `--smoke` (a 2-episode code check), the entropy
  start/end, and `--pressure-thresh` (defaults to 0.12 to match production).

## 5.6 `trafix_v6/finetune_morning_peak.py`

**Purpose.** An earlier, **optional** fine-tune that nudges the final model to handle morning
rush better without forgetting off-peak. Same shape as the other fine-tunes (its own
`RolloutBuffer`, `ppo_update`, `quick_eval`, `finetune`, `parse_args`): very low learning
rate, encoders frozen, a 70%-morning/30%-offpeak curriculum, and a "don't forget" guard that
only saves a new model if off-peak performance stays good.

## 5.7 `trafix_v6/eval_stage3.py`

**Purpose.** **Measure** a trained checkpoint across scenarios without changing it.
- **`run_episode(model, env, device, episode_idx, greedy)`** — run one scenario and collect
  queue/wait metrics; `greedy=True` uses argmax (deployment behavior).
- **`evaluate(args)`** — loop over scenarios, average the metrics, print a report; `_mean` is
  a small averaging helper inside it.
- **`parse_args()`** — options (`--greedy`, `--gui`, which checkpoint, …).

---

# Part 6 — Running the AI for real

Once trained, the weights are served by the backend and driven against a live SUMO. These
files are "system" more than "AI," but they're where the AI's decisions actually happen, so
here's what each does at a level that connects to Parts 3–5. (`system_explained.md` covers
them in full detail.)

## 6.1 `backend/main.py` — the inference server

**Purpose.** A small web service (FastAPI) that **owns the trained model** and answers "given
this telemetry, what phase should each junction run?" It never touches SUMO directly.
- **`load_model()`** — at startup, read which version to load (defaults to v6), build
  `TraFixV6`, load `trafix_v6_final.pt`, set it to evaluation mode, and build the production
  `RuleGovernor` (with `pressure_thresh=0.12`). If weights are missing it falls back to a
  simple rule so the lights never freeze. (The old v2/v3 code path was removed; an unknown
  version now errors out.)
- **`class Telemetry` / `class TelemetryBatch`** — the shapes of the incoming data (one
  junction, and a batch of all junctions for one step).
- **`receive_telemetry_batch(batch, …)`** — *the decision endpoint* (`POST /telemetry_batch`).
  It stores the raw telemetry, parses it into the 20-number observation, appends to a
  30-frame memory window, runs the model, applies the governor, takes **argmax** to choose a
  phase per junction, records the confidence, logs to the database in the background, and
  returns the chosen phases. This is steps 1–6 of Part 2.
- **`_build_obs_list()`** — assembles the 5-junction observation list (zero-filling any
  missing junction).
- **`startup_event` / `shutdown_event`** — load the model and open/close the database session.
- **`receive_emergency_event` / `get_emergency_metrics`** — receive and report ambulance
  preemption events (for the dashboard).
- **`get_state` / `get_last_decisions` / `get_db_summary`** — read-only endpoints for the
  dashboard and for checks.

## 6.2 `sumo/run_sumo_live.py` — the live driver

**Purpose.** The process that actually runs a live SUMO demo: every second it reads the
junctions, every 10 seconds it asks the backend for phases, and it actuates the lights —
**plus it adds the hand-coded safety overrides** that catch the model's known weaknesses.
- It reads lane counts (`collect_lane_obs`), builds the telemetry batch, POSTs it to the
  backend, and applies the returned phases through safe yellow transitions.
- **Starvation overrides** — `STARVE_LIMIT` (force a through-phase after too many
  left-only decisions), `DIRECTION_STARVE_LIMIT` (force the neglected through direction), and
  `LEFT_STARVE_LIMIT` (force a starved left). These are the *external* version of what the
  new anti-starvation reward teaches the model to do *internally*; they exist because the
  raw argmax policy can starve a movement.
- **Fixed-time fallback** — if the backend is unreachable several times, cycle NS/EW on a
  timer so the lights never freeze.

## 6.3 `sumo/emergency_preemption.py` — ambulance override

**Purpose.** A self-contained state machine that detects an emergency vehicle approaching a
junction and overrides the AI for that junction (yellow → all-red → green for the
ambulance's approach → hand control back), while measuring how long the ambulance took and
how many cars waited. It's deliberately separate from the main loop so it could later be
swapped for a camera/vision detector without touching the rest.

## 6.4 Launchers & plumbing

- **`baslat.py`** — one-click launcher: starts the backend (with the chosen model version)
  and then the live SUMO driver, wiring them together on a local port.
- **`main.py`** (repo root) — a thin bridge that exposes the backend's API plus the three
  dashboard HTML pages.
- **`run.py`** — a minimal alternative launcher for the backend server alone.
- **`backend/database.py`** — *not AI logic*; it just saves each decision and emergency event
  to PostgreSQL (and degrades gracefully to "no database" if none is available). Listed here
  only because `master_ai.md` mentions it.

---

*Companion to `master_ai.md` (reference), `system_explained.md` (data flow), and
`code_reference.md` (symbol index). This file is the from-scratch teaching version. Where it
describes behavior that differs from `master_ai.md` (share-based observation, the new
anti-starvation/clear reward terms, the removed v2 model, `pressure_thresh=0.12`,
`finetune_argmax.py`), this file reflects the **current** source; the deployed
`trafix_v6_final.pt` only gains the new-reward behavior after a retrain/fine-tune is run and
promoted.*
