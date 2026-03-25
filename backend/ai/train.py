"""
TraFix — SUMO Üzerinde PPO Eğitim Scripti
============================================
5 kavşaklı asimetrik ağda GCN+GRU tabanlı PPO ajanını eğitir.

Gereksinimler:
  • SUMO kurulu ve SUMO_HOME ortam değişkeni tanımlı
  • demo.sumocfg ve ilişkili net/route dosyaları mevcut

Kullanım:
  cd x:\\trafix\\sumo
  python -m backend.ai.train          # veya doğrudan: python train.py
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import warnings

# TraCI'nin kendi icindeki gereksiz deprecation uyarisini gizle
warnings.filterwarnings("ignore", category=UserWarning, module="traci")

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("HATA: SUMO_HOME ortam değişkeni bulunamadı!")

import traci  # noqa: E402

# Model import — doğrudan çalıştırılırsa relative path ayarlanır
try:
    from backend.ai.model import Core_PPO_Agent
except ImportError:
    # train.py doğrudan çalıştırılıyorsa
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from backend.ai.model import Core_PPO_Agent


# ──────────────────────────────────────────────
#  Konfigürasyon
# ──────────────────────────────────────────────
CONFIG = {
    "sumo_cfg": os.path.join(os.path.dirname(__file__), "..", "..", "..", "sumo", "training.sumocfg"),
    "num_features": 7,      # FEATURE_ORDER ile uyumlu
    "hidden_dim": 64,
    "num_actions": 4,       # 4 farklı trafik ışığı fazı
    "num_nodes": 5,         # 5 kavşak
    "lr": 3e-4,
    "gamma": 0.99,          # indirim faktörü
    "gae_lambda": 0.95,     # GAE lambda
    "clip_eps": 0.2,        # PPO clip epsilon
    "ppo_epochs": 4,        # her güncelleme adımında kaç epoch
    "episodes": 100,
    "max_steps": 3600,      # episode başına max simülasyon adımı
    "save_path": "core_agent_weights.pth",
}

# Kavşaklar arası bağlantı (backend'deki edge_index ile aynı topoloji)
EDGE_INDEX = torch.tensor([
    [0, 1, 0, 2, 1, 3, 2, 3, 2, 4, 1, 0, 2, 0, 3, 1, 3, 2, 4, 2],
    [1, 0, 2, 0, 3, 1, 3, 2, 4, 2, 0, 1, 0, 2, 1, 3, 2, 3, 2, 4],
], dtype=torch.long)

# Backend'deki 7 özellik sırası
FEATURE_ORDER = [
    "north_count", "south_count", "east_count", "west_count",
    "queue_length", "current_phase", "phase_duration",
]


# ──────────────────────────────────────────────
#  SUMO Yardımcı Fonksiyonlar
# ──────────────────────────────────────────────
def lane_to_edge_id(lane_id: str) -> str:
    return lane_id.rsplit("_", 1)[0]


def get_incoming_edges(tls_id: str) -> list:
    """Trafik ışığının kontrol ettiği incoming edge'leri bulur."""
    edges = set()
    try:
        for group in traci.trafficlight.getControlledLinks(tls_id):
            for link in group:
                if not link:
                    continue
                edge = lane_to_edge_id(link[0])
                if not edge.startswith(":"):
                    edges.add(edge)
    except Exception:
        pass
    return list(edges)


def classify_direction(junction_id: str, edge_ids: list) -> dict:
    """Edge'leri junction konumuna göre N/S/E/W sınıflandırır."""
    dmap = {"north": None, "south": None, "east": None, "west": None}
    try:
        jx, jy = traci.junction.getPosition(junction_id)
    except Exception:
        return dmap

    for eid in edge_ids:
        try:
            shape = traci.lane.getShape(f"{eid}_0")
            if not shape:
                continue
            x0, y0 = shape[0]
            dx, dy = x0 - jx, y0 - jy
            if abs(dx) > abs(dy):
                dmap["west" if dx < 0 else "east"] = eid
            else:
                dmap["south" if dy < 0 else "north"] = eid
        except Exception:
            pass
    return dmap


def build_intersection_map() -> dict:
    """Tüm trafik ışıklı kavşakları ve yön haritalarını döner."""
    imap = {}
    for tls_id in traci.trafficlight.getIDList():
        edges = get_incoming_edges(tls_id)
        dmap = classify_direction(tls_id, edges)
        imap[tls_id] = {"edges": edges, "directions": dmap}
    return imap


def edge_vehicle_count(edge_id) -> int:
    if edge_id is None:
        return 0
    try:
        return int(traci.edge.getLastStepVehicleNumber(edge_id))
    except Exception:
        return 0


def edge_waiting_time(edge_id) -> float:
    if edge_id is None:
        return 0.0
    try:
        return float(traci.edge.getWaitingTime(edge_id))
    except Exception:
        return 0.0


# ──────────────────────────────────────────────
#  State / Reward
# ──────────────────────────────────────────────
def get_state_from_sumo(intersection_map: dict, tls_ids: list) -> torch.Tensor:
    """
    Simülasyondan 5 kavşak × 7 feature tensörü oluşturur.
    Eğer ağda 5'ten az kavşak varsa, kalan satırlar sıfır kalır.
    """
    state = torch.zeros(CONFIG["num_nodes"], CONFIG["num_features"])

    for idx, tls_id in enumerate(tls_ids[:CONFIG["num_nodes"]]):
        data = intersection_map[tls_id]
        d = data["directions"]

        n_count = edge_vehicle_count(d["north"])
        s_count = edge_vehicle_count(d["south"])
        e_count = edge_vehicle_count(d["east"])
        w_count = edge_vehicle_count(d["west"])

        q_length = sum(
            edge_waiting_time(d[k]) for k in ("north", "south", "east", "west")
        )

        try:
            current_phase = int(traci.trafficlight.getPhase(tls_id))
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
            phase_duration = float(logic.phases[current_phase].duration)
        except Exception:
            current_phase, phase_duration = 0, 0.0

        state[idx] = torch.tensor([
            n_count, s_count, e_count, w_count,
            q_length, current_phase, phase_duration,
        ], dtype=torch.float32)

    return state


def calculate_reward(intersection_map: dict) -> float:
    """
    Ödül = ağdaki toplam bekleme süresinin negatifi.
    Düşük bekleme → yüksek ödül.
    """
    total_wait = 0.0
    for data in intersection_map.values():
        for edge_id in data["edges"]:
            total_wait += edge_waiting_time(edge_id)
    return -total_wait


# ──────────────────────────────────────────────
#  PPO Güncelleme
# ──────────────────────────────────────────────
def compute_gae(rewards, values, gamma, lam):
    """Generalized Advantage Estimation."""
    advantages = []
    gae = 0.0
    # Son adımdan geriye doğru
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages


def ppo_update(agent, optimizer, log_probs_old, values, rewards, states,
               edge_index, hidden_states, actions_taken):
    """
    PPO clipped surrogate güncelleme — mini-batch sampling ile hızlandırılmış.
    3600 adımın hepsini işlemek yerine her epoch'ta 64 rastgele örnek kullanır.
    """
    import random

    gamma = CONFIG["gamma"]
    lam = CONFIG["gae_lambda"]
    clip_eps = CONFIG["clip_eps"]
    mini_batch_size = 64  # Her epoch'ta kaç örnek işlenecek

    # Değerleri Python listesinden çıkar
    values_list = [v.item() for v in values]
    advantages = compute_gae(rewards, values_list, gamma, lam)

    # Tensörlere dönüştür
    advantages_t = torch.tensor(advantages, dtype=torch.float32)
    returns_t = advantages_t + torch.tensor(values_list[:len(advantages)], dtype=torch.float32)
    old_log_probs_t = torch.stack(log_probs_old).detach()

    # Normalize advantages
    if advantages_t.std() > 1e-8:
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    all_indices = list(range(len(rewards)))

    for _ in range(CONFIG["ppo_epochs"]):
        # Rastgele mini-batch seç (3600 yerine 64 örnek)
        batch = random.sample(all_indices, min(mini_batch_size, len(all_indices)))

        total_policy_loss = 0.0
        total_value_loss = 0.0

        for t in batch:
            action_probs, state_value, _ = agent(states[t], edge_index, hidden_states[t])

            dist = Categorical(action_probs)
            new_log_prob = dist.log_prob(actions_taken[t]).mean()

            ratio = torch.exp(new_log_prob - old_log_probs_t[t])
            surr1 = ratio * advantages_t[t]
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_t[t]
            policy_loss = -torch.min(surr1, surr2)

            value_loss = F.mse_loss(state_value.squeeze(), returns_t[t])

            total_policy_loss += policy_loss
            total_value_loss += value_loss

        loss = (total_policy_loss + 0.5 * total_value_loss) / len(batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
        optimizer.step()


# ──────────────────────────────────────────────
#  Ana Eğitim Döngüsü
# ──────────────────────────────────────────────
def main_training_loop():
    cfg = CONFIG

    # Model
    agent = Core_PPO_Agent(cfg["num_features"], cfg["hidden_dim"], cfg["num_actions"])
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg["lr"])

    sumo_cfg_path = os.path.abspath(cfg["sumo_cfg"])
    if not os.path.exists(sumo_cfg_path):
        sys.exit(f"HATA: SUMO config dosyası bulunamadı: {sumo_cfg_path}")

    print(f"SUMO config: {sumo_cfg_path}")
    print(f"Model: {cfg['num_features']} features, {cfg['hidden_dim']} hidden, {cfg['num_actions']} actions")
    print(f"Eğitim: {cfg['episodes']} episode × {cfg['max_steps']} adım")
    print("=" * 60)

    best_reward = -math.inf

    for ep in range(cfg["episodes"]):

        # Her episode basinda rastgele yeni bir trafik seli olustur
        gen_script = os.path.join(os.path.dirname(__file__), "..", "..", "..", "sumo", "generate_demand.py")
        import subprocess
        subprocess.run([sys.executable, gen_script], check=False)

        # SUMO'yu baslat
        traci.start(["sumo", "-c", CONFIG["sumo_cfg"], "--no-step-log", "true", "--no-warnings", "true"])
        
        # Kavşak haritasını kur
        intersection_map = build_intersection_map()
        tls_ids = sorted(intersection_map.keys())

        if not tls_ids:
            print("UYARI: Trafik ışıklı kavşak bulunamadı!")
            traci.close()
            continue

        # Episode verileri
        hidden_state = agent.init_hidden(cfg["num_nodes"])
        log_probs, values, rewards = [], [], []
        states, hidden_states, actions_taken = [], [], []

        step = 0
        while step < cfg["max_steps"] and traci.simulation.getMinExpectedNumber() > 0:

            # 1. Durumu al
            current_state = get_state_from_sumo(intersection_map, tls_ids)

            # 2. Ajan karar versin
            action_probs, state_value, new_hidden = agent(
                current_state, EDGE_INDEX, hidden_state
            )

            # Her kavşak için ayrı aksiyon örnekle
            dist = Categorical(action_probs)     # (num_nodes, num_actions)
            actions = dist.sample()              # (num_nodes,)

            # 3. Aksiyonları SUMO'ya uygula
            for i, tls_id in enumerate(tls_ids[:cfg["num_nodes"]]):
                try:
                    phase = actions[i].item()
                    # Faz sayısını aşmayacak şekilde clamp
                    logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
                    max_phase = len(logic.phases) - 1
                    phase = min(phase, max_phase)
                    traci.trafficlight.setPhase(tls_id, phase)
                except Exception:
                    pass

            # 4. Simülasyonu 1 adım ilerlet
            traci.simulationStep()

            # 5. Ödülü hesapla
            reward = calculate_reward(intersection_map)

            # Geçmişi RAM'de tut
            log_probs.append(dist.log_prob(actions).mean())
            values.append(state_value.detach())
            rewards.append(reward)
            states.append(current_state.detach())
            hidden_states.append(hidden_state.detach())
            actions_taken.append(actions.detach())

            hidden_state = new_hidden.detach()
            step += 1

        traci.close()
        import time
        time.sleep(1) # Portun tam kapanmasi icin ufak bekleme
        sys.stdout.flush()

        ep_reward = sum(rewards)
        print(f"Episode {ep + 1:3d}/{cfg['episodes']}  |  "
              f"Adım: {step:4d}  |  Toplam Ödül: {ep_reward:10.2f}")

        # PPO güncellemesi
        if len(rewards) > 0:
            ppo_update(
                agent, optimizer,
                log_probs, values, rewards,
                states, EDGE_INDEX, hidden_states, actions_taken,
            )

        # En iyi modeli kaydet
        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(agent.state_dict(), cfg["save_path"])
            print(f"  ↳ Yeni en iyi model kaydedildi! (Ödül: {best_reward:.2f})")

        # Her durumda son modeli de kaydet
        torch.save(agent.state_dict(), "core_agent_weights_latest.pth")

    print("=" * 60)
    print(f"Eğitim tamamlandı. En iyi ödül: {best_reward:.2f}")
    print(f"Model dosyası: {cfg['save_path']}")


if __name__ == "__main__":
    main_training_loop()
