# TraFix v6 — AI Model Explanation

> **Multi-Intersection Traffic Signal Control using Temporal-Graph Actor-Critic Reinforcement Learning**

This document explains the architecture, training pipeline, and testing framework of the TraFix v6 AI model, which optimises traffic signal timing across 5 interconnected intersections using a GRU temporal encoder, a GATConv graph encoder, and a hybrid PPO actor-critic trained in three stages.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [Observation & Data Flow](#3-observation--data-flow)
4. [Rule Governor](#4-rule-governor)
5. [Training Pipeline](#5-training-pipeline)
6. [Test Suite](#6-test-suite)
7. [File Structure](#7-file-structure)
8. [Hyperparameters Reference](#8-hyperparameters-reference)

---

## 1. High-Level Overview

TraFix v6 is a reinforcement learning agent that controls traffic light phases at 5 intersections simultaneously. It connects to [SUMO](https://sumo.dlr.de/) via the TraCI API, observes real-time traffic conditions as a rolling 30-step window, and outputs a phase decision for each junction every simulation second.

### What's New in v6 vs v5

| Feature | v5 | v6 |
|---------|----|----|
| Observation dim | 10 | **20** (12 per-lane counts + queue + 6-bit phase one-hot + duration) |
| Phases per junction | 4 | **6** (NS-through, N-left, S-left, EW-through, E-left, W-left) |
| Lane width | 2-lane roads | **3-lane roads** |
| Spatial encoder | GATConv | GATConv (same, 4 heads × 32 = 128) |
| Critic | Global only | **Hybrid** (per-junction local + global mean-pool) |
| Training | 2-stage | **3-stage** (GRU pretrain → GAT pretrain → full PPO) |

### Decision Flow

```
SUMO TraCI
  → parse_sumo_observations()        20-dim per junction
  → Rolling window [T=30, J=5, 20]
  → GRU temporal encoder             [B, J, 128]
  → GATConv graph encoder            [B, J, 128]  (4 heads × 32)
  → Shared MLP trunk                 [B, J, 64]
  → 5 × Actor head                   5 × [B, 6] logits
  → RuleGovernor.apply()             mask/adjust logits
  → argmax → phase decisions         5 integers in [0, 5]
```

---

## 2. Architecture Deep Dive

### 2.1 `_TemporalEncoder` — GRU

```
Input:  [B, T=30, J=5, obs_dim=20]
Output: [B, J=5, hidden_dim=128]
```

- All 5 junctions are processed **in a single batched GRU call** — the `[B, J]` dimensions are flattened into the batch dimension before the GRU, then reshaped back.
- Single GRU layer, `hidden_dim=128`, `batch_first=True`.
- Only the **final hidden state** `h_n` is returned (the 30-step temporal summary).

### 2.2 `_GraphEncoder` — GATConv

```
Input:  [B×J, 128]  node features on a chain graph
Output: [B×J, 128]  (4 heads × 32 dims, concatenated)
```

- Uses a **fixed chain graph** `0–1–2–3–4` (bidirectional), matching the physical junction layout.
- `GATConv(in=128, out=32, heads=4, concat=True)` — 4 attention heads each producing 32-dim output, concatenated to 128.
- For batched inference, the edge index is offset per graph in the batch.

### 2.3 `_SharedTrunk` — MLP

```
Input:  [B, J, 128]
Output: [B, J, 64]

Linear(128 → 128) → ReLU → Linear(128 → 64) → ReLU
```

Shared across all junctions; projects the graph-encoded features into the final representation space used by both actor and critic heads.

### 2.4 Actor Heads (5 independent)

```
5 × Linear(64 → 6)   →   raw logits per junction
```

Each junction has its own linear head outputting 6 logits (one per phase). The RuleGovernor then masks/adjusts these before argmax.

### 2.5 Hybrid Critic

```
Local:   5 × Linear(64 → 1)    →  V_local[j]   per-junction
Global:  Linear(64 → 1)        →  V_global      on mean-pooled trunk
Value:   V[j] = V_local[j] + V_global
```

The per-junction value `V[j]` combines a junction-specific local estimate with a network-wide global estimate — giving the critic both local and global context without requiring separate modules.

### 2.6 Input Features (20-dim per junction)

| Index | Feature | Normalisation |
|-------|---------|--------------|
| 0 | `north_left` lane count | ÷ 15 |
| 1 | `north_through` lane count | ÷ 30 |
| 2 | `north_right` lane count | ÷ 15 |
| 3 | `south_left` | ÷ 15 |
| 4 | `south_through` | ÷ 30 |
| 5 | `south_right` | ÷ 15 |
| 6 | `east_left` | ÷ 15 |
| 7 | `east_through` | ÷ 30 |
| 8 | `east_right` | ÷ 15 |
| 9 | `west_left` | ÷ 15 |
| 10 | `west_through` | ÷ 30 |
| 11 | `west_right` | ÷ 15 |
| 12 | `queue_length` | ÷ 200 |
| 13–18 | `current_phase` one-hot (6 bits) | — |
| 19 | `phase_duration` | min(÷ 120, 3.0) |

Parsing is handled by `parse_sumo_observations()` in `backend/ai/trafix_v2.py`, which remains the shared observation utility across all model versions.

### 2.7 Phase Definitions

| Phase | Movement |
|-------|---------|
| 0 | NS-through (North↕South straight) |
| 1 | N-left turn |
| 2 | S-left turn |
| 3 | EW-through (East↔West straight) |
| 4 | E-left turn |
| 5 | W-left turn |

---

## 3. Observation & Data Flow

### 3.1 Forward Pass

```python
# 1. Parse SUMO telemetry → normalised tensor
obs = parse_sumo_observations(raw_obs_list)       # [J=5, 20]

# 2. Accumulate rolling window (T=30 steps, 1 step = 1 sim-second)
window = torch.stack([obs] * T).unsqueeze(0)      # [1, T=30, J=5, 20]

# 3. Temporal encoding via GRU
h = temporal_enc(window)                          # [1, J=5, 128]

# 4. Spatial encoding via GATConv
h_flat = h.reshape(J, 128)
g = graph_enc(h_flat, edge_index)                 # [J=5, 128]

# 5. Shared trunk
t = trunk(g.reshape(1, J, 128))                   # [1, J=5, 64]

# 6. Per-junction actor logits
logits = [actor_heads[j](t[:, j, :]) for j in range(J)]   # 5 × [1, 6]

# 7. Rule Governor constrains logits
logits = rule_governor.apply(logits, obs)         # 5 × [1, 6] adjusted

# 8. Greedy phase selection
phases = [logits[j].argmax().item() for j in range(J)]    # [0..5] × 5
```

### 3.2 Action Selection Modes

| Mode | Method | Used when |
|------|--------|-----------|
| Training | `Categorical(logits).sample()` | Exploration during PPO rollouts |
| Evaluation | `logits.argmax()` | Live deployment, deterministic |

---

## 4. Rule Governor

`trafix_v6/rule_governor.py` applies **hard traffic-domain constraints** to the raw actor logits before a phase is selected. This prevents the AI from making physically illegal or unsafe decisions.

### Constraints Applied

| Rule | Mechanism |
|------|-----------|
| **Min-green** | If the current phase has been active < 10 s (through) or < 8 s (left-turn), its logit is set to `-∞` for all *other* phases — forcing the model to hold the current phase. |
| **Max-green** | If the current phase has been active > 90 s (through) or > 45 s (left-turn), its logit is set to `-∞` — forcing a switch. |
| **Anti-flicker** | If the model tries to revert to the previous phase (A→B→A), a `-10.0` penalty is applied to prevent rapid toggling. |
| **Pressure bonus** | A `+bonus` is added to the logit of the phase that serves the most-loaded direction (NS or EW), based on normalised vehicle counts. |

The Governor also tracks state across steps via `update_state()` and can be `reset()` between episodes.

---

## 5. Training Pipeline

TraFix v6 uses a **3-stage curriculum** to stabilise training of the joint GRU + GAT model.

### Stage 1 — GRU Pretraining (`stage1_pretrain_gru.py`)

Trains only the `_TemporalEncoder` (GRU) on synthetic traffic sequences using supervised regression targets derived from a heuristic controller.

```
Output: checkpoints/stage1_gru.pt
```

### Stage 2 — GATConv Pretraining (`stage2_pretrain_gatconv.py`)

Loads Stage 1 weights (GRU frozen), trains the `_GraphEncoder` + trunk on graph-structured inputs, again via supervised targets.

```
Output: checkpoints/stage2_gatconv.pt
         checkpoints/stage2_trunk.pt
```

### Stage 3 — Full PPO Training (`stage3_train_ppo.py`)

Loads Stage 1 and Stage 2 weights. Runs full PPO with all components unfrozen, using **differential learning rates** (lower LR for pretrained components, higher for actor/critic heads).

```
PPO structure (per episode):
  1. Start SUMO simulation
  2. Warm up 50 steps (agent observes, does not act)
  3. Collect rollout:
       Every step: observe → Governor-constrained sample → apply → reward
       Buffer size: 256 steps
  4. Compute GAE advantages (γ=0.99, λ=0.95)
  5. Run 4 PPO epochs over buffer
  6. Log metrics, save checkpoint every 100 episodes

Output: checkpoints/stage3_ep{N}.pt   (every 100 episodes)
         checkpoints/trafix_v6_final.pt (episode 2000)
         checkpoints/stage3_ep1000.pt  (kept for comparison)
```

### PPO Objective

```
ratio   = exp(new_log_prob - old_log_prob)
surr1   = ratio × advantage
surr2   = clamp(ratio, 1-0.2, 1+0.2) × advantage
L_policy = -min(surr1, surr2).mean()

L_total = L_policy + 0.5 × L_value - 0.01 × H(π)
```

The entropy coefficient `0.01` (higher than v5's `0.005`) prevents premature collapse over 6 phases.

### GAE

```
δ_t = r_t + γ × V(s_{t+1}) - V(s_t)
A_t = δ_t + (γλ) × A_{t+1}          (reversed accumulation)
```

Advantages are normalised (mean=0, std=1) before each PPO update.

---

## 6. Test Suite

Tests live in `tests/mandatory/` and run without SUMO. All 121 tests pass.

| File | Coverage |
|------|---------|
| `test_unit_model.py` | GRU/GATConv output shapes, no NaN, determinism in eval, checkpoint loading |
| `test_unit_rule_governor.py` | Min/max green enforcement, anti-flicker penalty, pressure bonus |
| `test_unit_observation.py` | `parse_sumo_observations` shape, normalisation, one-hot correctness, edge cases |
| `test_unit_preemption.py` | Emergency preemption state machine transitions |
| `test_nfr.py` | NFR-01 latency (p99 < 1000 ms), NFR-02 heuristic fallback, NFR-05 cross-platform imports |
| `test_fr.py` | FR-01 telemetry keys, FR-02 phase validity, FR-03 preemption override, FR-04 DB graceful, FR-06 yellow phase |
| `test_metrics_modules.py` | All 12 simulation metric compute functions |

Run all:
```bash
python -m pytest tests/mandatory/ -v
```

---

## 7. File Structure

```
backend/ai/
├── trafix_v2.py          # parse_sumo_observations(), NUM_NODE_FEATURES,
│                         # compute_reward() — shared observation utilities
│                         # also the legacy v2 CoordinatedPPOAgent model
├── train_v2.py           # TrainConfig, SumoEnvironment — used by v6 training scripts
└── MODEL_EXPLANATION.md  # This document

trafix_v6/
├── trafix_v6.py          # TraFixV6 model (GRU + GATConv + PPO actor-critic)
├── rule_governor.py      # RuleGovernor — hard traffic constraints on logits
├── stage1_pretrain_gru.py
├── stage2_pretrain_gatconv.py
├── stage3_train_ppo.py
├── eval_stage3.py
├── finetune_morning_peak.py
├── scenario_generator.py
└── checkpoints/
    ├── stage1_gru.pt         # Stage 1 pretrained GRU weights
    ├── stage2_gatconv.pt     # Stage 2 pretrained GATConv weights
    ├── stage2_trunk.pt       # Stage 2 pretrained trunk weights
    ├── stage3_ep1000.pt      # Mid-training checkpoint (checkpoint comparison)
    ├── stage3_ep2000.pt      # End-of-training checkpoint
    └── trafix_v6_final.pt    # Production model (identical to ep2000)

sumo/
├── map.net.xml           # 5-junction 3-lane road network
├── run_sumo_live.py      # Live simulation loop (TraCI → backend → SUMO)
└── emergency_preemption.py

backend/
├── main.py               # FastAPI server — telemetry intake, AI inference, DB logging
└── database.py           # PostgreSQL persistence (sessions, step_log, emergency_log)

tests/mandatory/          # 121 mandatory tests (all pass)
tests/reports/            # Analysis charts — AI vs baseline, roundabout comparison
```

---

## 8. Hyperparameters Reference

### Model

| Parameter | Value | Description |
|-----------|-------|-------------|
| `obs_dim` | 20 | Features per junction per timestep |
| `num_phases` | 6 | Discrete phases per junction |
| `num_junctions` | 5 | Fixed network size |
| `T` (window) | 30 | Timesteps fed to GRU |
| `hidden_dim` | 128 | GRU hidden size |
| `gat_heads` | 4 | GAT attention heads |
| `gat_head_dim` | 32 | Dims per GAT head (total = 128) |
| `trunk_mid` | 128 | MLP hidden size |
| `trunk_out` | 64 | MLP output size (input to actor/critic) |

### Training (Stage 3 PPO)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `episodes` | 2000 | Total training episodes |
| `rollout_length` | 256 | Steps per PPO update |
| `ppo_epochs` | 4 | PPO passes per rollout |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE λ |
| `clip_eps` | 0.2 | PPO clip range |
| `entropy_coef` | 0.01 | Entropy bonus weight |
| `value_coef` | 0.5 | Value loss weight |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `lr` | 3e-4 | Learning rate (Adam) |
| `warmup_steps` | 50 | SUMO warm-up before agent acts |

### Rule Governor

| Parameter | Value |
|-----------|-------|
| `min_green_through` | 10 s |
| `min_green_left` | 8 s |
| `max_green_through` | 90 s |
| `max_green_left` | 45 s |
| `anti_flicker_penalty` | −10.0 |
| `pressure_bonus` | adaptive (based on normalised queue counts) |

---

*Last updated: 2026-05-25*
