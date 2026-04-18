"""
TraFix Simple — Eğitim Scripti
================================
Kullanım:
  python backend/ai/train_simple.py
  python backend/ai/train_simple.py --episodes 300
  python backend/ai/train_simple.py --resume
"""

import os
import sys
import json
import time
import math
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# ── SUMO ──
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    for candidate in [
        "C:\\Program Files (x86)\\Eclipse\\Sumo\\tools",
        "C:\\Program Files\\Eclipse\\Sumo\\tools",
        "/usr/share/sumo/tools",
        "/usr/local/share/sumo/tools",
    ]:
        if os.path.isdir(candidate):
            sys.path.append(candidate)
            break

try:
    import traci
except ImportError:
    raise ImportError("SUMO TraCI bulunamadı. SUMO_HOME ortam değişkenini ayarlayın.")

# ── Dinamik talep ──
try:
    from sumo.generate_demand import generate_dynamic_demand as _gen_demand
    _HAS_DYNAMIC_DEMAND = True
except ImportError:
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sumo"))
        from generate_demand import generate_dynamic_demand as _gen_demand
        _HAS_DYNAMIC_DEMAND = True
    except ImportError:
        _HAS_DYNAMIC_DEMAND = False

# ── Model ──
try:
    from backend.ai.trafix_simple import (
        SimplePPOAgent, parse_sumo_observations,
        compute_reward, compute_gae, train_step, RewardWeights, NUM_NODE_FEATURES,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from trafix_simple import (
        SimplePPOAgent, parse_sumo_observations,
        compute_reward, compute_gae, train_step, RewardWeights, NUM_NODE_FEATURES,
    )


# ══════════════════════════════════════════════════
#  Konfigürasyon
# ══════════════════════════════════════════════════

@dataclass
class TrainConfig:
    sumo_cfg:          str   = "sumo/training.sumocfg"
    net_file:          str   = "sumo/map.net.xml"
    output_dir:        str   = "training_outputs_simple"
    checkpoint_path:   str   = "coordinated_agent_weights_simple.pth"

    episodes:          int   = 300
    max_steps:         int   = 3600
    decision_interval: int   = 10       # eğitim = inference (run_sumo_live.py ile aynı)
    warmup_steps:      int   = 50
    ppo_epochs:        int   = 4
    rollout_length:    int   = 32       # küçük rollout → sık güncelleme

    hidden_dim:        int   = 64       # v2/v3'ten küçük → hızlı eğitim
    num_actions:       int   = 4
    lr:                float = 3e-4
    lr_min:            float = 1e-5
    gamma:             float = 0.99
    gae_lambda:        float = 0.95
    clip_eps:          float = 0.2
    entropy_coef:      float = 0.05    # entropi çöküşünü önler
    entropy_coef_min:  float = 0.01
    entropy_decay:     float = 0.9998  # yavaş bozunma
    value_coef:        float = 0.5
    max_grad_norm:     float = 0.5

    reward_weights: RewardWeights = field(default_factory=RewardWeights)

    save_interval:     int   = 25
    log_interval:      int   = 5

    gui:               bool  = False
    sumo_step_length:  float = 1.0
    seed:              int   = 42
    resume:            bool  = False


# ══════════════════════════════════════════════════
#  SUMO Ortamı
# ══════════════════════════════════════════════════

class SumoEnvironment:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.tls_ids: List[str] = []
        self.num_nodes = 0
        self._step_count = 0

    def start(self, episode: int = 0):
        cfg_path = os.path.abspath(self.cfg.sumo_cfg)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"SUMO config bulunamadı: {cfg_path}\n"
                f"Proje kökünden çalıştırın: python backend/ai/train_simple.py"
            )

        sumo_binary = "sumo-gui" if self.cfg.gui else "sumo"
        traci.start([
            sumo_binary, "-c", cfg_path,
            "--step-length", str(self.cfg.sumo_step_length),
            "--waiting-time-memory", "1000",
            "--no-warnings", "true",
            "--seed", str(self.cfg.seed + episode),
        ])
        self._step_count = 0
        self.tls_ids = sorted(traci.trafficlight.getIDList())
        self.num_nodes = len(self.tls_ids)

        if self.num_nodes == 0:
            raise RuntimeError("SUMO ağında trafik ışığı bulunamadı.")

        for _ in range(self.cfg.warmup_steps):
            traci.simulationStep()
            self._step_count += 1

    def close(self):
        try:
            traci.close()
        except Exception:
            pass

    @property
    def is_running(self) -> bool:
        try:
            return traci.simulation.getMinExpectedNumber() > 0
        except Exception:
            return False

    def get_observations(self) -> List[Dict]:
        obs = []
        for idx, tls_id in enumerate(self.tls_ids):
            controlled = traci.trafficlight.getControlledLanes(tls_id)
            unique     = list(dict.fromkeys(controlled))

            direction_counts = {"north": 0, "south": 0, "east": 0, "west": 0}
            total_queue = 0.0

            for i, lane in enumerate(unique):
                vc = traci.lane.getLastStepVehicleNumber(lane)
                q  = traci.lane.getLastStepHaltingNumber(lane) * \
                     max(traci.lane.getLastStepLength(lane), 7.5)

                dir_name = ["north", "south", "east", "west"][i % 4]
                try:
                    shape = traci.lane.getShape(lane)
                    if len(shape) >= 2:
                        dx = shape[-1][0] - shape[0][0]
                        dy = shape[-1][1] - shape[0][1]
                        angle = math.degrees(math.atan2(dy, dx)) % 360
                        if   45  <= angle < 135: dir_name = "north"
                        elif 135 <= angle < 225: dir_name = "west"
                        elif 225 <= angle < 315: dir_name = "south"
                        else:                    dir_name = "east"
                except Exception:
                    pass

                direction_counts[dir_name] += vc
                total_queue += q

            current_phase = traci.trafficlight.getPhase(tls_id)
            try:
                prog_dur   = traci.trafficlight.getPhaseDuration(tls_id)
                next_sw    = traci.trafficlight.getNextSwitch(tls_id)
                sim_time   = traci.simulation.getTime()
                elapsed    = prog_dur - max(0, next_sw - sim_time)
                phase_dur  = max(0.0, elapsed)
            except Exception:
                phase_dur = 0.0

            obs.append({
                "intersection_id": idx,
                "north_count":  direction_counts["north"],
                "south_count":  direction_counts["south"],
                "east_count":   direction_counts["east"],
                "west_count":   direction_counts["west"],
                "queue_length": total_queue,
                "current_phase": current_phase % self.cfg.num_actions,
                "phase_duration": phase_dur,
            })
        return obs

    def apply_actions(self, actions: torch.Tensor):
        for i, tls_id in enumerate(self.tls_ids):
            desired = int(actions[i].item())
            current = traci.trafficlight.getPhase(tls_id)

            logic = traci.trafficlight.getAllProgramLogics(tls_id)
            if logic:
                desired = desired % len(logic[0].phases)

            if desired == current:
                continue

            # Güvenlik: Green → Green atlamasını engelle
            if current == 0 and desired in [2, 3]:
                desired = 1
            elif current == 2 and desired in [0, 1]:
                desired = 3

            traci.trafficlight.setPhase(tls_id, desired)

    def step(self, actions: Optional[torch.Tensor] = None) -> Tuple[List[Dict], bool]:
        if actions is not None:
            self.apply_actions(actions)

        for _ in range(self.cfg.decision_interval):
            if not self.is_running:
                return self.get_observations(), True
            traci.simulationStep()
            self._step_count += 1

        done = not self.is_running or self._step_count >= self.cfg.max_steps
        return self.get_observations(), done

    def get_metrics(self) -> Dict[str, float]:
        try:
            vehicles = traci.vehicle.getIDList()
            if not vehicles:
                return {"avg_speed": 0, "avg_waiting": 0, "total_vehicles": 0}
            speeds  = [traci.vehicle.getSpeed(v) for v in vehicles]
            waiting = [traci.vehicle.getWaitingTime(v) for v in vehicles]
            return {
                "avg_speed":      sum(speeds)  / len(speeds),
                "avg_waiting":    sum(waiting) / len(waiting),
                "total_vehicles": len(vehicles),
            }
        except Exception:
            return {"avg_speed": 0, "avg_waiting": 0, "total_vehicles": 0}


# ══════════════════════════════════════════════════
#  Rollout Buffer
# ══════════════════════════════════════════════════

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.observations:  List[torch.Tensor] = []
        self.actions:       List[torch.Tensor] = []
        self.log_probs:     List[torch.Tensor] = []
        self.rewards:       List[torch.Tensor] = []
        self.values:        List[torch.Tensor] = []
        self.hidden_states: List[torch.Tensor] = []

    def add(self, obs, action, log_prob, reward, value, hidden):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value.detach())
        self.hidden_states.append(hidden.detach())

    def __len__(self):
        return len(self.rewards)

    def to_dict(self, next_value: torch.Tensor) -> Dict:
        return {
            "observations":  self.observations,
            "actions":       self.actions,
            "log_probs":     self.log_probs,
            "rewards":       self.rewards,
            "values":        self.values,
            "hidden_states": self.hidden_states,
            "next_value":    next_value,
        }


# ══════════════════════════════════════════════════
#  Yardımcılar
# ══════════════════════════════════════════════════

class TrainingLogger:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "training_log.csv"
        self.history: List[Dict] = []

        with open(self.csv_path, "w") as f:
            f.write("episode,reward_mean,policy_loss,value_loss,entropy,avg_speed,avg_waiting,lr,entropy_coef,timestamp\n")

        log_path = self.output_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path, encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ],
        )

    def log_episode(self, episode: int, data: Dict):
        self.history.append({"episode": episode, **data})
        with open(self.csv_path, "a") as f:
            f.write(
                f"{episode},{data.get('reward_mean',0):.6f},"
                f"{data.get('policy_loss',0):.6f},{data.get('value_loss',0):.6f},"
                f"{data.get('entropy',0):.6f},{data.get('avg_speed',0):.4f},"
                f"{data.get('avg_waiting',0):.4f},{data.get('lr',0):.8f},"
                f"{data.get('entropy_coef',0):.6f},{datetime.now().isoformat()}\n"
            )


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_episodes, total_episodes, lr_min):
        self.optimizer        = optimizer
        self.warmup_episodes  = warmup_episodes
        self.total_episodes   = total_episodes
        self.lr_min           = lr_min
        self.base_lr          = optimizer.param_groups[0]["lr"]

    def step(self, episode: int) -> float:
        if episode < self.warmup_episodes:
            lr = self.base_lr * (episode + 1) / self.warmup_episodes
        else:
            progress = (episode - self.warmup_episodes) / max(1, self.total_episodes - self.warmup_episodes)
            lr = self.lr_min + 0.5 * (self.base_lr - self.lr_min) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


def _save_checkpoint(agent, optimizer, episode, best_reward, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({
        "model_state_dict":     agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode":              episode,
        "best_reward":          best_reward,
        "config": {
            "hidden_dim":   agent.hidden_dim,
            "num_actions":  agent.num_actions,
            "entropy_coef": agent.entropy_coef,
            "model":        "simple",
        },
    }, path)


# ══════════════════════════════════════════════════
#  Ana Eğitim Döngüsü
# ══════════════════════════════════════════════════

def train(cfg: TrainConfig):
    logger = TrainingLogger(cfg.output_dir)
    logging.info("=" * 55)
    logging.info("  TraFix Simple — Eğitim Başlıyor")
    logging.info("=" * 55)

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"  Cihaz: {device}")

    # Kavşak sayısını öğren
    env = SumoEnvironment(cfg)
    env.start(episode=0)
    num_nodes = env.num_nodes
    env.close()
    logging.info(f"  Kavşak sayısı: {num_nodes}")

    # Ajan
    agent = SimplePPOAgent(
        num_node_features=NUM_NODE_FEATURES,
        hidden_dim=cfg.hidden_dim,
        num_actions=cfg.num_actions,
        entropy_coef=cfg.entropy_coef,
        value_coef=cfg.value_coef,
        clip_eps=cfg.clip_eps,
        max_grad_norm=cfg.max_grad_norm,
    ).to(device)

    # Checkpoint'tan devam
    start_episode = 0
    if cfg.resume and os.path.exists(cfg.checkpoint_path):
        try:
            ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=True)
            state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            agent.load_state_dict(state)
            start_episode = ckpt.get("episode", 0)
            logging.info(f"  Checkpoint yüklendi: episode {start_episode}")
        except Exception as e:
            logging.warning(f"  Checkpoint yüklenemedi, sıfırdan başlanıyor: {e}")

    optimizer = optim.Adam(agent.parameters(), lr=cfg.lr)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_episodes=min(20, cfg.episodes // 10),
        total_episodes=cfg.episodes,
        lr_min=cfg.lr_min,
    )

    best_reward    = float("-inf")
    reward_history = deque(maxlen=50)
    current_entropy_coef = cfg.entropy_coef

    logging.info(f"  Parametre sayısı: {sum(p.numel() for p in agent.parameters()):,}")
    logging.info(f"  Episode: {start_episode} → {cfg.episodes}")
    logging.info("=" * 55)

    for episode in range(start_episode, cfg.episodes):
        ep_start = time.time()

        # Entropi katsayısı azalt
        current_entropy_coef = max(
            cfg.entropy_coef_min,
            cfg.entropy_coef * (cfg.entropy_decay ** episode),
        )
        agent.entropy_coef = current_entropy_coef
        current_lr = scheduler.step(episode)

        # Her episode farklı trafik yoğunluğu
        if _HAS_DYNAMIC_DEMAND:
            try:
                _gen_demand()
            except Exception as e:
                logging.warning(f"  Talep üretilemedi: {e}")

        # SUMO başlat
        try:
            env.start(episode=episode)
        except FileNotFoundError as e:
            logging.error(str(e))
            sys.exit(1)
        except Exception as e:
            logging.error(f"  Episode {episode}: SUMO başlatılamadı: {e}")
            failures = getattr(train, "_failures", 0) + 1
            train._failures = failures
            if failures >= 3:
                logging.error("  3 ardışık hata — durduruluyor.")
                sys.exit(1)
            continue
        train._failures = 0

        agent.train()
        hidden     = agent.init_hidden(num_nodes, device)
        buffer     = RolloutBuffer()
        obs        = env.get_observations()
        prev_obs   = None
        prev_actions = None

        ep_rewards: List[float] = []
        ep_metrics: List[Dict]  = []
        total_pl = total_vl = total_ent = 0.0
        update_count = 0
        step = 0
        done = False

        while not done:
            x = parse_sumo_observations(obs, device)

            actions, log_probs, value, new_hidden = agent.select_actions(x, hidden)
            next_obs, done = env.step(actions)

            reward = compute_reward(
                current_obs=next_obs,
                previous_obs=prev_obs,
                previous_actions=prev_actions,
                current_actions=actions,
                weights=cfg.reward_weights,
            ).to(device)

            ep_rewards.append(reward.item())
            buffer.add(x, actions, log_probs, reward, value, hidden)

            if len(buffer) >= cfg.rollout_length:
                with torch.no_grad():
                    next_x = parse_sumo_observations(next_obs, device)
                    _, next_value, _ = agent(next_x, new_hidden)

                losses = train_step(agent, optimizer, buffer.to_dict(next_value.detach()), cfg.ppo_epochs)
                total_pl  += losses["policy"]
                total_vl  += losses["value"]
                total_ent += losses["entropy"]
                update_count += 1
                buffer.clear()

            if step % 10 == 0:
                ep_metrics.append(env.get_metrics())

            prev_obs     = obs
            prev_actions = actions
            obs          = next_obs
            hidden       = new_hidden
            step        += 1

        # Kalan buffer
        if len(buffer) > 1:
            with torch.no_grad():
                x = parse_sumo_observations(obs, device)
                _, next_value, _ = agent(x, hidden)
            losses = train_step(agent, optimizer, buffer.to_dict(next_value.detach()), cfg.ppo_epochs)
            total_pl  += losses["policy"]
            total_vl  += losses["value"]
            total_ent += losses["entropy"]
            update_count += 1

        env.close()

        mean_reward = sum(ep_rewards) / max(len(ep_rewards), 1)
        reward_history.append(mean_reward)
        rolling_avg  = sum(reward_history) / len(reward_history)
        avg_metrics  = {}
        if ep_metrics:
            for k in ep_metrics[0]:
                avg_metrics[k] = sum(m[k] for m in ep_metrics) / len(ep_metrics)

        log_data = {
            "reward_mean":   mean_reward,
            "policy_loss":   total_pl / max(update_count, 1),
            "value_loss":    total_vl / max(update_count, 1),
            "entropy":       total_ent / max(update_count, 1),
            "lr":            current_lr,
            "entropy_coef":  current_entropy_coef,
            **avg_metrics,
        }
        logger.log_episode(episode, log_data)

        if episode % cfg.log_interval == 0:
            logging.info(
                f"EP {episode:>4d}/{cfg.episodes} | "
                f"R={mean_reward:+.4f} (avg50={rolling_avg:+.4f}) | "
                f"π={log_data['policy_loss']:.4f} V={log_data['value_loss']:.4f} "
                f"H={log_data['entropy']:.4f} | "
                f"spd={avg_metrics.get('avg_speed',0):.1f} "
                f"wait={avg_metrics.get('avg_waiting',0):.1f} | "
                f"LR={current_lr:.2e} εH={current_entropy_coef:.4f} | "
                f"{time.time()-ep_start:.1f}s"
            )

        if mean_reward > best_reward:
            best_reward = mean_reward
            _save_checkpoint(agent, optimizer, episode, best_reward,
                             os.path.join(cfg.output_dir, "best_model.pth"))

        if (episode + 1) % cfg.save_interval == 0:
            _save_checkpoint(agent, optimizer, episode, best_reward, cfg.checkpoint_path)
            logging.info(f"  → Checkpoint: episode {episode + 1}")

    _save_checkpoint(agent, optimizer, cfg.episodes - 1, best_reward, cfg.checkpoint_path)
    _save_checkpoint(agent, optimizer, cfg.episodes - 1, best_reward,
                     os.path.join(cfg.output_dir, "final_model.pth"))

    logging.info("=" * 55)
    logging.info("  EĞİTİM TAMAMLANDI")
    logging.info(f"  En iyi ödül  : {best_reward:.6f}")
    logging.info(f"  Checkpoint   : {cfg.checkpoint_path}")
    logging.info("=" * 55)


# ══════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="TraFix Simple Eğitim")
    p.add_argument("--sumo-cfg",   default="sumo/training.sumocfg")
    p.add_argument("--output-dir", default="training_outputs_simple")
    p.add_argument("--checkpoint", default="coordinated_agent_weights_simple.pth")
    p.add_argument("--episodes",   type=int,   default=300)
    p.add_argument("--hidden-dim", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--gui",        action="store_true")
    p.add_argument("--resume",     action="store_true")
    p.add_argument("--seed",       type=int,   default=42)
    args = p.parse_args()

    return TrainConfig(
        sumo_cfg        = args.sumo_cfg,
        output_dir      = args.output_dir,
        checkpoint_path = args.checkpoint,
        episodes        = args.episodes,
        hidden_dim      = args.hidden_dim,
        lr              = args.lr,
        gui             = args.gui,
        resume          = args.resume,
        seed            = args.seed,
    )


if __name__ == "__main__":
    train(parse_args())
