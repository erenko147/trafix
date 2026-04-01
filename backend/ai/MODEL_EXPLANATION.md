# TraFix v2 — AI Model Explanation

> **Coordinated Multi-Intersection Traffic Signal Control using Deep Reinforcement Learning**

This document explains the architecture, training pipeline, and testing framework of the TraFix v2 AI model, which optimizes traffic signal timing across multiple interconnected intersections using a combination of Graph Neural Networks, Recurrent Networks, and Attention-based coordination within a Proximal Policy Optimization (PPO) reinforcement learning framework.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [Data Flow Pipeline](#3-data-flow-pipeline)
4. [Reward Function Design](#4-reward-function-design)
5. [Training Pipeline](#5-training-pipeline)
6. [Test Suite](#6-test-suite)
7. [File Structure](#7-file-structure)
8. [Code Review Findings](#8-code-review-findings)
9. [Hyperparameters Reference](#9-hyperparameters-reference)

---

## 1. High-Level Overview

TraFix v2 is a reinforcement learning (RL) agent that controls traffic light phases at multiple intersections simultaneously. It connects to [SUMO](https://sumo.dlr.de/) (Simulation of Urban Mobility) via the TraCI API, observes real-time traffic conditions, and decides which traffic light phase each intersection should activate.

### What Makes v2 Special

| Feature | Purpose |
|---|---|
| **Graph Convolutional Network (GCN)** | Captures spatial relationships between intersections |
| **Gated Recurrent Unit (GRU)** | Maintains temporal memory across time steps |
| **Multi-Head Self-Attention** | Coordinates decisions across intersections (green wave) |
| **PPO with Entropy Bonus** | Stable training with exploration incentives |
| **Multi-component Reward** | Balances pressure, queues, throughput, fairness & stability |

### Decision Flow

```
SUMO Observations → Normalization → GCN (spatial) → GRU (temporal)
    → Multi-Head Attention (coordination) → Actor (actions) + Critic (value)
```

---

## 2. Architecture Deep Dive

### 2.1 Model Definition — `trafix_v2.py`

The model is implemented as `CoordinatedPPOAgent`, composed of three main modules:

#### a) SpatioTemporalGNN (GCN + GRU)

```
Input: (N, 7) node features + (2, E) edge index + (1, N, H) hidden state
Output: (N, H) features + (1, N, H) new hidden state
```

- **Two GCN layers** with a residual (skip) connection: the second GCN output is added back to the first, helping gradient flow and preserving local intersection information.
- **Layer Normalization** after GCN layers stabilizes training.
- **GRU** processes each intersection as an independent sequence element, maintaining a per-intersection hidden state across time steps. This lets the model remember traffic patterns over time.

#### b) IntersectionCoordinator (Multi-Head Attention)

```
Input: (N, H) features from GNN
Output: (N, H) coordinated features
```

- Uses **4-head self-attention** so each intersection can "see" all other intersections' states.
- Follows a **Transformer-style** architecture: self-attention → residual + LayerNorm → FFN → residual + LayerNorm.
- The FFN uses **GELU** activation (smoother than ReLU, common in modern transformers).
- This is what enables **green wave coordination** — e.g., if Intersection 1 turns green for eastbound traffic, Intersection 2 downstream can anticipate the incoming vehicles.

#### c) Actor-Critic Heads

- **Actor**: `Linear(H, H/2) → ReLU → Linear(H/2, 4) → Softmax` — outputs a probability distribution over 4 phases for each intersection.
- **Critic**: `Linear(H, H/2) → ReLU → Linear(H/2, 1)` — takes the mean of all coordinated features (global view) and outputs a single state value estimate for the entire network.

### 2.2 Input Features (7 per intersection)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `north_count` | Vehicles approaching from north |
| 1 | `south_count` | Vehicles approaching from south |
| 2 | `east_count` | Vehicles approaching from east |
| 3 | `west_count` | Vehicles approaching from west |
| 4 | `queue_length` | Total queue length (halted vehicles × vehicle length) |
| 5 | `current_phase` | Currently active traffic light phase |
| 6 | `phase_duration` | How long the current phase has been active |

### 2.3 Normalization Strategy

The `parse_sumo_observations()` function normalizes raw SUMO data:

- **Vehicle counts** (indices 0–3): divided by the max count across all intersections (clamped min=1)
- **Queue length** (index 4): divided by its max value (clamped min=1)
- **Phase index** (index 5): divided by `max(4, actual_max)` — normalized to [0, 1]
- **Phase duration** (index 6): divided by 120.0 seconds (assumed max phase length)

### 2.4 Graph Topology

The default topology models a 5-intersection asymmetric network:

```
  0 — 1 — 2
      |   |
      3 — 4
```

The `build_edge_index()` function in `train_v2.py` first tries to automatically extract the topology from the SUMO `.net.xml` file using `sumolib`. If that fails, it falls back to this hardcoded layout. All edges are bidirectional.

---

## 3. Data Flow Pipeline

### 3.1 Forward Pass

```python
# 1. Parse SUMO JSON → normalized tensor
x = parse_sumo_observations(raw_obs)               # (5, 7)

# 2. Spatial aggregation via GCN
h = ReLU(GCN1(x, edges))                           # (5, 128)
h = h + ReLU(GCN2(h, edges))                       # skip connection
h = LayerNorm(h)

# 3. Temporal memory via GRU
features, new_hidden = GRU(h, hidden_state)         # (5, 128), (1, 5, 128)

# 4. Inter-intersection coordination via Attention
coordinated = MultiheadAttention(features)          # (5, 128)

# 5. Decision making
action_probs = Softmax(Actor(coordinated))          # (5, 4) — per-intersection
state_value  = Critic(Mean(coordinated))            # (1,)  — global
```

### 3.2 Action Selection

During training, actions are **sampled** from the categorical distribution (for exploration). During evaluation, **argmax** gives deterministic decisions.

Each intersection independently selects from 4 phases:

| Action | Phase |
|--------|-------|
| 0 | North-South green |
| 1 | South-North green |
| 2 | East-West green |
| 3 | West-East green |

---

## 4. Reward Function Design

The reward function (`compute_reward()`) combines six components to guide the agent toward optimal traffic management:

### Component Breakdown

| Component | Weight | Signal | Description |
|---|---|---|---|
| **Pressure** | -0.4 | Penalty | Total vehicle count across all intersections (normalized). Fewer vehicles waiting = better. |
| **Queue** | -0.3 | Penalty | Total queue lengths. Shorter queues = better. |
| **Throughput** | +0.3 | Reward | Decrease in total vehicles compared to previous step. More cars passing through = better. |
| **Fairness** | -0.15 | Penalty | Coefficient of variation of directional vehicle counts. Penalizes ignoring one direction too long. |
| **Phase Stability** | -0.10 | Penalty | Fraction of intersections that changed phase. Discourages unnecessary switching. |
| **Wait Penalty** | -0.05 | Penalty | Extra penalty when any intersection stays on the same phase > 60 seconds. |

### Reward Formula

```
R = -0.4 × pressure - 0.3 × queue + 0.3 × throughput
    - 0.15 × fairness - 0.10 × phase_changes - 0.05 × wait_penalty
```

> **Design Philosophy**: The reward balances *efficiency* (throughput, pressure) against *fairness* (no direction should be starved) and *stability* (avoid rapid phase flipping while preventing overly long waits).

---

## 5. Training Pipeline

### 5.1 Training Script — `train_v2.py`

The training loop follows a standard **on-policy PPO** structure with SUMO as the environment:

```
For each episode (0 to 500):
    1. Start SUMO simulation
    2. Warm up for 50 simulation steps
    3. Collect rollout:
        - Every 10 sim-seconds: observe → select action → apply to SUMO → compute reward
        - Store (obs, action, log_prob, reward, value, hidden) in buffer
    4. When buffer reaches 64 steps: compute GAE advantages → run 4 PPO epochs
    5. Log metrics, save checkpoints
```

### 5.2 Key Training Components

#### GAE (Generalized Advantage Estimation)

Uses `γ=0.99` and `λ=0.95` to compute advantage estimates that balance bias and variance:

```python
δ_t = r_t + γ × V(s_{t+1}) - V(s_t)        # TD error
A_t = δ_t + (γλ) × A_{t+1}                  # GAE accumulation (reversed)
```

Advantages are normalized (mean=0, std=1) before PPO updates.

#### PPO Clipped Objective

```python
ratio = exp(new_log_prob - old_log_prob)
surr1 = ratio × advantage
surr2 = clamp(ratio, 1-ε, 1+ε) × advantage   # ε = 0.2
policy_loss = -min(surr1, surr2).mean()
```

#### Total Loss

```
L_total = L_policy + 0.5 × L_value - 0.02 × H(π)
```

Where `H(π)` is the entropy bonus that encourages exploration.

### 5.3 Learning Rate Schedule

- **Cosine Warmup**: Linear warmup for the first 10% of episodes (up to 20), then cosine decay from `3e-4` down to `1e-5`.
- **Entropy Decay**: The entropy coefficient decays by `0.9995×` per episode from `0.02` to a minimum of `0.005`, gradually reducing exploration.

### 5.4 SUMO Environment Interface (`SumoEnvironment`)

- **Observation Collection**: For each traffic light, it reads vehicle counts per direction (using lane geometry to infer direction via angle calculation), queue lengths, current phase, and phase duration.
- **Action Application**: Converts model actions (phase indices) to SUMO phase changes, respecting the actual number of phases in the SUMO logic.
- **Metrics**: Collects average speed, average waiting time, total vehicles, and halting count.

### 5.5 Checkpointing

| File | When Saved |
|---|---|
| `best_model.pth` | Whenever mean episode reward improves |
| `checkpoint_epN.pth` | Every 25 episodes |
| `coordinated_agent_weights.pth` | Every 25 episodes + at training end |
| `final_model.pth` | At training completion |

Each checkpoint stores: model state dict, optimizer state dict, episode number, best reward, and config.

---

## 6. Test Suite

### 6.1 Test Script — `test_model.py`

The test suite runs **9 comprehensive tests** without requiring SUMO, using predefined traffic scenarios:

| Test | What It Validates |
|---|---|
| **1. Model Info** | All modules exist (GNN, Coordinator, Actor, Critic), parameters are trainable |
| **2. Forward Pass** | Correct input/output shapes across 4 scenarios, normalization in [0,1], valid action range [0,4) |
| **3. Reward Function** | Heavy traffic < normal reward; night > normal; throughput direction; phase stability penalty; wait penalty; fairness penalty |
| **4. Coordination Effect** | Attention layer actually modifies features (diff > 0); bottleneck intersection representation changes |
| **5. Consistency** | Deterministic forward pass (argmax) gives same results across 5 trials; value estimates are stable |
| **6. Entropy & Exploration** | Probability distributions are valid (≥0, sum=1); entropy > 0 (model explores); entropy bonus is active |
| **7. Edge Cases** | Empty traffic (all zeros), extreme traffic (999 vehicles), single spike, GRU hidden state stability over 10 steps |
| **8. GAE Computation** | Advantages are computed without NaN; normalization works (mean ≈ 0) |
| **9. PPO Loss** | Loss computation without NaN; gradients flow through the entire network |

### 6.2 Test Scenarios

| Scenario | Description |
|---|---|
| Normal Traffic | Moderate, varied traffic across 5 intersections |
| Heavy Traffic (K3 bottleneck) | Intersection K3 has 35+28+15+20 = 98 vehicles, 180m queue |
| Night Traffic | Very low volume (0–2 vehicles per direction), long phase durations |
| East-West Dominant | One-directional pressure (18–25 vehicles E/W vs 0–2 N/S) |

---

## 7. File Structure

```
backend/ai/
├── trafix_v2.py              # Model architecture + reward + GAE + train_step
├── train_v2.py               # SUMO integration + training loop + CLI
├── test_model.py             # 9-test comprehensive validation suite
└── MODEL_EXPLANATION.md      # This document

sumo/
├── training.sumocfg          # SUMO config (references map.net.xml + training_demand.rou.xml)
├── map.net.xml               # Road network definition
└── training_demand.rou.xml   # Vehicle demand/routes for training

training_outputs/
├── training.log              # Training session logs
└── training_log.csv          # Per-episode metrics (reward, loss, speed, etc.)

*.pth                         # Saved model weights
├── core_agent_weights.pth
└── core_agent_weights_latest.pth
```

---

## 8. Code Review Findings

### ✅ Strengths

1. **Well-structured architecture**: Clean separation between spatial (GCN), temporal (GRU), and coordination (Attention) modules. The modular design makes each component independently testable.

2. **Robust SUMO integration**: The `SumoEnvironment` class handles edge cases well — checking for TLS existence, direction inference via lane geometry angles, safe metric collection with try/except, and proper phase count validation.

3. **Comprehensive test suite**: The 9 tests cover model structure, forward pass correctness, reward behavior, coordination effects, determinism, exploration, edge cases, GAE, and gradient flow — all without requiring SUMO.

4. **Smart normalization**: Vehicle counts use adaptive max-normalization rather than fixed ranges, letting the model handle both low and high traffic without manual tuning.

5. **Production-ready training**: Includes checkpoint resume, cosine warmup LR schedule, entropy decay, rolling average tracking, CSV + JSON logging, and consecutive failure detection.

6. **Flexible import paths**: Both `train_v2.py` and `test_model.py` handle multiple import paths (package import, relative import, direct run), making them usable in different contexts.

### ⚠️ Observations

1. **Training log shows no completed episodes**: The `training.log` file contains 5 initialization attempts (2026-03-26) that all stopped at "Cihaz: cpu" without any episode output. This suggests SUMO failed to start during ` env.start()` — likely a missing/misconfigured SUMO configuration file path. The weight files (`core_agent_weights.pth`) appear to be from a previous successful training session on a different machine.

2. **Default CLI path mismatch**: The `TrainConfig` dataclass defaults to `sumo_cfg="sumo/training.sumocfg"` (project root relative), but the CLI parser defaults to `"training.sumocfg"` (current directory relative). When run from the `sumo/` directory this works, but from project root it would need `--sumo-cfg sumo/training.sumocfg`.

3. **Weight file naming**: The code saves to `coordinated_agent_weights.pth` but the existing weight files are named `core_agent_weights.pth` — the test script's `find_weights()` function correctly searches for both names, but this naming inconsistency could cause confusion.

4. **CPU-only training**: All training runs show "Cihaz: cpu". For a model with GCN + GRU + Attention, GPU acceleration would significantly speed up training with larger networks.

5. **Single-threaded SUMO**: Each episode runs a full SUMO instance sequentially. Parallel environment sampling (vectorized envs) could improve training throughput.

---

## 9. Hyperparameters Reference

| Parameter | Value | Description |
|---|---|---|
| `hidden_dim` | 128 | Size of all hidden representations |
| `num_actions` | 4 | Traffic light phases per intersection |
| `num_heads` | 4 | Attention heads in coordinator |
| `lr` | 3e-4 | Initial learning rate (Adam) |
| `lr_min` | 1e-5 | Minimum learning rate |
| `gamma` | 0.99 | Discount factor for future rewards |
| `gae_lambda` | 0.95 | GAE lambda for advantage estimation |
| `clip_eps` | 0.2 | PPO clipping range |
| `entropy_coef` | 0.02 → 0.005 | Entropy bonus (decays over training) |
| `value_coef` | 0.5 | Value loss weight in total loss |
| `max_grad_norm` | 0.5 | Gradient clipping threshold |
| `episodes` | 500 | Total training episodes |
| `max_steps_per_episode` | 3600 | Max simulation seconds per episode |
| `decision_interval` | 10 | Agent decides every 10 sim-seconds |
| `rollout_length` | 64 | Steps before PPO update |
| `ppo_epochs` | 4 | PPO passes over each rollout |
| `warmup_steps` | 50 | SUMO warmup before agent starts |

---

*Document generated: 2026-04-01*
