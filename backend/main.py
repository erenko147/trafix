"""
TraFix Backend — FastAPI
========================
Telemetri alır, AI modeline sorar, faz kararı döner.
Dashboard için GET /state endpoint'i sunar.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import deque
import torch
import os
import logging

# AI model import
# TRAFIX_MODEL_VERSION=v2 (default) → trafix_v2 + coordinated_agent_weights.pth
# TRAFIX_MODEL_VERSION=v3           → trafix_v3 + coordinated_agent_weights_v3.pth
_MODEL_VERSION = os.environ.get("TRAFIX_MODEL_VERSION", "v2").strip().lower()

if _MODEL_VERSION == "v3":
    from backend.ai.trafix_v3 import CoordinatedPPOAgent, parse_sumo_observations
    _WEIGHT_FILENAME = "coordinated_agent_weights_v3.pth"
    _USE_GRAPH = True
    _USE_V5 = False
elif _MODEL_VERSION == "simple":
    from backend.ai.trafix_simple import SimplePPOAgent as CoordinatedPPOAgent, parse_sumo_observations
    _WEIGHT_FILENAME = "coordinated_agent_weights_simple.pth"
    _USE_GRAPH = False
    _USE_V5 = False
elif _MODEL_VERSION == "v5":
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(__file__))))
    from trafix_v5.trafix_v5 import TraFixV5
    from trafix_v5.rule_governor import RuleGovernor
    from backend.ai.trafix_v2 import parse_sumo_observations
    _WEIGHT_FILENAME = "trafix_v5/checkpoints/trafix_v5_final.pt"
    _USE_GRAPH = False
    _USE_V5 = True
    CoordinatedPPOAgent = None  # unused for v5
else:  # v2 default
    from backend.ai.trafix_v2 import CoordinatedPPOAgent, parse_sumo_observations
    _WEIGHT_FILENAME = "coordinated_agent_weights.pth"
    _USE_GRAPH = True
    _USE_V5 = False

logger = logging.getLogger("trafix")

app = FastAPI(title="TraFix API", version="2.0")

# ==========================================
# CORS Ayarları
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# GLOBAL STATE (In-Memory Veritabanı)
# ==========================================
state_dict = {
    "0": {},
    "1": {},
    "2": {},
    "3": {},
    "4": {},
}

# ==========================================
# VERİ ŞEMASI
# ==========================================
class Telemetry(BaseModel):
    intersection_id: int
    north_count: int
    south_count: int
    east_count: int
    west_count: int
    queue_length: float
    current_phase: int
    phase_duration: float


# ==========================================
# MODEL KONFİGÜRASYONU
# ==========================================
NUM_FEATURES = 10  # 4 counts + queue + 4 one-hot phase bits + duration
HIDDEN_DIM = 128
NUM_ACTIONS = 4
NUM_NODES = 5
NUM_HEADS = 4

FEATURE_ORDER = [
    "north_count",
    "south_count",
    "east_count",
    "west_count",
    "queue_length",
    "current_phase",
    "phase_duration",
]

# Kavşaklar arası bağlantı — map.net.xml'den çıkarılan 5-kavşak topolojisi
#   0 — 1 — 2
#       |   |
#       3 — 4
# Eğitimle (train_v2.py build_edge_index) aynı grafı kullanmalı
edge_index = torch.tensor([
    [0, 1, 1, 2, 1, 3, 2, 4, 3, 4],
    [1, 0, 2, 1, 3, 1, 4, 2, 4, 3],
], dtype=torch.long)

# ==========================================
# AI MODEL — Başlangıçta None, startup'ta yüklenir
# ==========================================
ai_agent = None
obs_history: deque = deque(maxlen=500)  # kept for compatibility but unused after fixed-scale normalisation
last_decisions_cache: list = []  # /last_decisions endpoint için önbellek

# v5-specific: sliding temporal window [T, 5, obs_dim]
_V5_T_WINDOW = 10
_v5_window: deque = deque(maxlen=_V5_T_WINDOW)
_v5_governor: "RuleGovernor | None" = None


def load_model():
    """Eğitilmiş model ağırlıklarını diskten yükler."""
    global ai_agent

    base_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(base_dir)

    print(f"[INFO] Model versiyonu: {_MODEL_VERSION.upper()} | Aranan ağırlık: {_WEIGHT_FILENAME}")

    if _USE_V5:
        global _v5_governor
        weight_paths = [
            os.path.join(project_root, _WEIGHT_FILENAME),
            _WEIGHT_FILENAME,
        ]
        agent = TraFixV5(obs_dim=NUM_FEATURES, num_phases=NUM_ACTIONS)
        for path in weight_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                try:
                    ckpt = torch.load(abs_path, map_location="cpu", weights_only=True)
                    state = ckpt.get("model_state_dict", ckpt)
                    agent.load_state_dict(state)
                    agent.eval()
                    ai_agent = agent
                    _v5_governor = RuleGovernor(
                        num_junctions=NUM_NODES,
                        num_phases=NUM_ACTIONS,
                        min_green_s=10.0,
                        max_green_s=90.0,
                        flicker_window=2,
                        flicker_penalty=3.0,
                        pressure_boost=1.0,
                    )
                    print(f"[OK] TraFixV5 modeli yuklendi: {abs_path}")
                    print(f"[OK] RuleGovernor aktif (min_green=10s, max_green=90s)")
                    return True
                except RuntimeError as e:
                    print(f"[WARN] V5 agirlik dosyasi uyumsuz: {abs_path} — {e}")
                    continue
        print("[WARN] TraFixV5 agirlik dosyasi bulunamadi. Heuristic fallback aktif olacak.")
        return False

    agent = CoordinatedPPOAgent(
        num_node_features=NUM_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_actions=NUM_ACTIONS,
        num_heads=NUM_HEADS,
    )

    # Olası weight dosya yolları (öncelik sırasıyla)
    weight_paths = [
        os.path.join(base_dir, "ai", _WEIGHT_FILENAME),
        os.path.join(base_dir, "..", _WEIGHT_FILENAME),
        _WEIGHT_FILENAME,
        os.path.join(base_dir, "ai", "core_agent_weights.pth"),
        os.path.join(base_dir, "..", "core_agent_weights.pth"),
        "core_agent_weights.pth",
    ]

    for path in weight_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            try:
                state_dict = torch.load(abs_path, map_location="cpu", weights_only=True)
                # Handle checkpoint format (dict with 'model_state_dict' key)
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                agent.load_state_dict(state_dict)
                agent.eval()
                ai_agent = agent
                logger.info(f"AI modeli yüklendi: {abs_path}")
                print(f"[OK] AI modeli yuklendi: {abs_path}")
                return True
            except RuntimeError as e:
                logger.warning(f"Agirlik dosyasi uyumsuz: {abs_path} — {e}")
                print(f"[WARN] Agirlik dosyasi uyumsuz (eski model?): {abs_path}")
                continue

    logger.warning("AI model agirlik dosyasi bulunamadi. Heuristic fallback aktif.")
    print("[WARN] AI model agirlik dosyasi bulunamadi. Heuristic fallback aktif olacak.")
    return False


@app.on_event("startup")
async def startup_event():
    """Sunucu başlarken modeli yüklemeye çalışır."""
    load_model()


# ==========================================
# GRAPH BUILDER
# ==========================================
def graph_builder() -> torch.Tensor:
    """state_dict → (5, 7) tensör."""
    rows = []
    for i in range(NUM_NODES):
        node = state_dict.get(str(i), {})
        row = [float(node.get(f, 0.0)) for f in FEATURE_ORDER]
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


from typing import List

class TelemetryBatch(BaseModel):
    step: int
    intersections: List[Telemetry]

# ==========================================
# POST /telemetry_batch — Toplu Telemetri Al, Toplu Karar Dön
# ==========================================
@app.post("/telemetry_batch")
async def receive_telemetry_batch(batch: TelemetryBatch):
    global last_decisions_cache

    # 1. Gelen veriyi state_dict'e kaydet
    # current_phase in the incoming data must be in 0–3 model-action space
    # (0=N-only, 1=E-only, 2=S-only, 3=W-only), NOT the 0–7 SUMO phase space.
    for data in batch.intersections:
        state_dict[str(data.intersection_id)] = data.dict()

    decisions = []

    # ─── AI MODEL AKTİF ─────────────────────────────────
    if ai_agent is not None:
        # Build observation list from all known intersections
        obs_list = []
        for i in range(NUM_NODES):
            node = state_dict.get(str(i), {})
            if node:
                obs_list.append(node)
            else:
                obs_list.append({
                    "intersection_id": i,
                    "north_count": 0, "south_count": 0,
                    "east_count": 0, "west_count": 0,
                    "queue_length": 0.0,
                    "current_phase": 0, "phase_duration": 0.0,
                })

        # Fixed-scale normalisation (matches training)
        node_features = parse_sumo_observations(obs_list)  # [5, obs_dim]

        with torch.no_grad():
            if _USE_V5:
                # Maintain temporal window and run TraFixV5
                if len(_v5_window) == 0:
                    for _ in range(_V5_T_WINDOW):
                        _v5_window.append(node_features.detach())
                else:
                    _v5_window.append(node_features.detach())
                # [1, T, 5, obs_dim]
                window_tensor = torch.stack(list(_v5_window)).unsqueeze(0)
                logits_list, _ = ai_agent(window_tensor)

                # Apply governor rules to logits
                obs_last = window_tensor[0, -1]  # [J, obs_dim]
                if _v5_governor is not None:
                    logits_list = _v5_governor.apply(logits_list, obs_last)

                # Stack softmax probs to [5, num_phases]
                action_probs = torch.stack(
                    [torch.softmax(l, dim=-1).squeeze(0) for l in logits_list], dim=0
                )
            elif _USE_GRAPH:
                action_probs, _ = ai_agent(node_features, edge_index)
            else:
                action_probs, _ = ai_agent(node_features)

            chosen_phases = []
            for data in batch.intersections:
                idx = data.intersection_id
                if 0 <= idx < action_probs.shape[0]:
                    next_phase = int(torch.argmax(action_probs[idx]).item())
                    confidence = round(float(action_probs[idx].max().item()), 3)
                else:
                    next_phase = data.current_phase
                    confidence = 0.0
                chosen_phases.append(next_phase)
                decisions.append({
                    "intersection_id": idx,
                    "next_phase": next_phase,
                    "confidence": confidence,
                    "total_vehicles": data.north_count + data.south_count + data.east_count + data.west_count,
                    "queue_length": round(data.queue_length, 1),
                })

            # Update governor state so anti-flicker tracks this step
            if _USE_V5 and _v5_governor is not None:
                chosen_tensor = torch.tensor(chosen_phases, dtype=torch.long)
                _v5_governor.update_state(chosen_tensor)

        last_decisions_cache = decisions
        return {"decisions": decisions}

    # ─── HEURİSTİC FALLBACK ─────────────────────────────
    # AI modeli yoksa matematiksel kural tabanlı karar
    for data in batch.intersections:
        directions = {
            0: data.north_count,
            1: data.south_count,
            2: data.east_count,
            3: data.west_count,
        }

        best_dir = max(directions, key=directions.get)
        max_cars = directions[best_dir]

        # Minimum 10 saniye yanma kuralı
        if data.phase_duration < 10.0 or max_cars == 0:
            next_phase = data.current_phase
        else:
            # 0=North, 1=South -> Phase 0 (NS Green)
            # 2=East, 3=West -> Phase 2 (EW Green)
            next_phase = 0 if best_dir in [0, 1] else 2

        decisions.append({
            "intersection_id": data.intersection_id,
            "next_phase": next_phase,
            "confidence": 0.0,
            "total_vehicles": data.north_count + data.south_count + data.east_count + data.west_count,
            "queue_length": round(data.queue_length, 1),
        })

    last_decisions_cache = decisions
    return {"decisions": decisions}


# ==========================================
# GET /last_decisions — Architecture Sayfası İçin Son Kararlar
# ==========================================
@app.get("/last_decisions")
async def get_last_decisions():
    return {"decisions": last_decisions_cache}


# ==========================================
# GET /state — Dashboard İçin Durum Verisi
# ==========================================
@app.get("/state")
async def get_state():
    return state_dict