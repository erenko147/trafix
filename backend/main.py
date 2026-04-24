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

_MODEL_VERSION = os.environ.get("TRAFIX_MODEL_VERSION", "v2").strip().lower()

if _MODEL_VERSION == "v3":
    from backend.ai.trafix_v3 import CoordinatedPPOAgent, parse_sumo_observations
    _WEIGHT_FILENAME = "coordinated_agent_weights_v3.pth"
    _USE_GRAPH = True
elif _MODEL_VERSION == "simple":
    from backend.ai.trafix_simple import SimplePPOAgent as CoordinatedPPOAgent, parse_sumo_observations
    _WEIGHT_FILENAME = "coordinated_agent_weights_simple.pth"
    _USE_GRAPH = False
else:  # v2 default
    from backend.ai.trafix_v2 import CoordinatedPPOAgent, parse_sumo_observations
    _WEIGHT_FILENAME = "coordinated_agent_weights.pth"
    _USE_GRAPH = True

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
NUM_FEATURES = 7
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
hidden_state = None  # GRU hidden state — istekler arası korunur
obs_history: deque = deque(maxlen=500)  # rolling araç sayısı geçmişi — adaptif normalizasyon için
last_decisions_cache: list = []  # /last_decisions endpoint için önbellek


def _detect_arch(keys) -> str:
    """Agirlik dosyasinin anahtar isimlerinden model mimarisini tespit et."""
    key_str = " ".join(keys)
    if "gconv_gru" in key_str:
        return "v3"
    if "gcn1" in key_str or "gcn2" in key_str:
        return "v2"
    if "attn_norm" in key_str or ("proj" in key_str and "gru" in key_str and "gcn" not in key_str):
        return "simple"
    return "unknown"


def _make_agent(version: str):
    """Versiyon stringine gore agent + parser ikilisi don."""
    if _MODEL_VERSION == "simple":
        return (
            CoordinatedPPOAgent(
                num_node_features=NUM_FEATURES,
                hidden_dim=64,
                num_actions=NUM_ACTIONS,
            ),
            parse_sumo_observations,
        )
    if _MODEL_VERSION == "v3":
        return (
            CoordinatedPPOAgent(
                num_node_features=NUM_FEATURES,
                hidden_dim=HIDDEN_DIM,
                num_actions=NUM_ACTIONS,
                num_heads=NUM_HEADS,
            ),
            parse_sumo_observations,
        )
    return (
        CoordinatedPPOAgent(
            num_node_features=NUM_FEATURES,
            hidden_dim=HIDDEN_DIM,
            num_actions=NUM_ACTIONS,
            num_heads=NUM_HEADS,
        ),
        parse_sumo_observations,
    )


def load_model():
    """Egitilmis model agirliklarini diskten yukler. Mimariyi otomatik algilar."""
    global ai_agent, hidden_state, parse_sumo_observations

    base_dir = os.path.dirname(__file__)

    # Tum olasi .pth dosya yollarini topla (tekrarsiz)
    candidates = [
        os.path.join(base_dir, "ai", "coordinated_agent_weights.pth"),
        os.path.join(base_dir, "..", "coordinated_agent_weights.pth"),
        os.path.join(base_dir, "ai", "coordinated_agent_weights_v3.pth"),
        os.path.join(base_dir, "..", "coordinated_agent_weights_v3.pth"),
        os.path.join(base_dir, "ai", "core_agent_weights.pth"),
        os.path.join(base_dir, "..", "core_agent_weights.pth"),
        os.path.join(base_dir, "..", "coordinated_agent_weights_simple.pth"),
        os.path.join(base_dir, "..", "training_outputs_simple", "best_model.pth"),
    ]
    # Tekrar eden mutlak yollari kaldir, var olmayanlari at
    seen = set()
    weight_paths = []
    for p in candidates:
        ap = os.path.abspath(p)
        if ap not in seen and os.path.exists(ap):
            seen.add(ap)
            weight_paths.append(ap)

    if not weight_paths:
        print("[WARN] Hicbir agirlik dosyasi bulunamadi. Heuristic fallback aktif.")
        return False

    for abs_path in weight_paths:
        try:
            sd = torch.load(abs_path, map_location="cpu", weights_only=True)
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]

            detected = _detect_arch(list(sd.keys()))
            agent, parser = _make_agent(detected)

            agent.load_state_dict(sd)
            agent.eval()
            ai_agent = agent
            hidden_state = agent.init_hidden(NUM_NODES)
            parse_sumo_observations = parser

            print(f"[OK] Model yuklendi: {abs_path}")
            print(f"[OK] Algilanan mimari: {detected.upper()} "
                  f"(istenen: {_MODEL_VERSION.upper()})")
            return True

        except Exception as e:
            print(f"[WARN] {os.path.basename(abs_path)} yuklenemedi: {e}")
            continue

    print("[WARN] Agirlik dosyalari yuklenemedii. Heuristic fallback aktif.")
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
    global hidden_state, obs_history, last_decisions_cache

    # 1. Gelen veriyi state_dict'e kaydet
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
                # Default empty observation for missing intersections
                obs_list.append({
                    "intersection_id": i,
                    "north_count": 0, "south_count": 0,
                    "east_count": 0, "west_count": 0,
                    "queue_length": 0.0,
                    "current_phase": 0, "phase_duration": 0.0,
                })

        # Rolling normalisation: bu batch'teki araç sayılarını geçmişe ekle
        for node in obs_list:
            obs_history.append(
                node["north_count"] + node["south_count"] +
                node["east_count"]  + node["west_count"]
            )
        _count_max = float(max(obs_history)) if obs_history else None

        # Normalize data exactly as during training
        node_features = parse_sumo_observations(obs_list, count_max=_count_max)

        with torch.no_grad():
            if _USE_GRAPH:
                action_probs, _, new_hidden = ai_agent(node_features, edge_index, hidden_state)
            else:
                action_probs, _, new_hidden = ai_agent(node_features, hidden_state)
            hidden_state = new_hidden

            for data in batch.intersections:
                idx = data.intersection_id
                if 0 <= idx < action_probs.shape[0]:
                    next_phase = int(torch.argmax(action_probs[idx]).item())
                    confidence = round(float(action_probs[idx].max().item()), 3)
                else:
                    next_phase = data.current_phase
                    confidence = 0.0
                decisions.append({
                    "intersection_id": idx,
                    "next_phase": next_phase,
                    "confidence": confidence,
                    "total_vehicles": data.north_count + data.south_count + data.east_count + data.west_count,
                    "queue_length": round(data.queue_length, 1),
                })

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