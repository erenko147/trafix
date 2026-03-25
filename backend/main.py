"""
TraFix Backend — FastAPI
========================
Telemetri alır, AI modeline sorar, faz kararı döner.
Dashboard için GET /state endpoint'i sunar.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import os
import logging

# AI model import
from backend.ai.model import Core_PPO_Agent

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
HIDDEN_DIM = 64
NUM_ACTIONS = 4
NUM_NODES = 5

FEATURE_ORDER = [
    "north_count",
    "south_count",
    "east_count",
    "west_count",
    "queue_length",
    "current_phase",
    "phase_duration",
]

# Kavşaklar arası bağlantı (asimetrik 5-kavşak topolojisi)
edge_index = torch.tensor([
    [0, 1, 0, 2, 1, 3, 2, 3, 2, 4, 1, 0, 2, 0, 3, 1, 3, 2, 4, 2],
    [1, 0, 2, 0, 3, 1, 3, 2, 4, 2, 0, 1, 0, 2, 1, 3, 2, 3, 2, 4],
], dtype=torch.long)

# ==========================================
# AI MODEL — Başlangıçta None, startup'ta yüklenir
# ==========================================
ai_agent = None
hidden_state = None  # GRU hidden state — istekler arası korunur


def load_model():
    """Eğitilmiş model ağırlıklarını diskten yükler."""
    global ai_agent, hidden_state

    agent = Core_PPO_Agent(NUM_FEATURES, HIDDEN_DIM, NUM_ACTIONS)

    # Olası weight dosya yolları (öncelik sırasıyla)
    weight_paths = [
        os.path.join(os.path.dirname(__file__), "ai", "core_agent_weights.pth"),
        os.path.join(os.path.dirname(__file__), "..", "core_agent_weights.pth"),
        "core_agent_weights.pth",
    ]

    for path in weight_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            agent.load_state_dict(torch.load(abs_path, map_location="cpu"))
            agent.eval()
            ai_agent = agent
            hidden_state = agent.init_hidden(NUM_NODES)
            logger.info(f"AI modeli yüklendi: {abs_path}")
            print(f"[OK] AI modeli yuklendi: {abs_path}")
            return True

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


# ==========================================
# POST /telemetry — Telemetri Al, Karar Dön
# ==========================================
@app.post("/telemetry")
async def receive_telemetry(data: Telemetry):
    global hidden_state

    # 1. Gelen veriyi state_dict'e kaydet
    state_dict[str(data.intersection_id)] = data.dict()

    # ─── AI MODEL AKTİF ─────────────────────────────────
    if ai_agent is not None:
        node_features = graph_builder()

        with torch.no_grad():
            action_probs, _, new_hidden = ai_agent(
                node_features, edge_index, hidden_state
            )
            hidden_state = new_hidden

            # Bu kavşak için en yüksek olasılıklı fazı seç
            idx = data.intersection_id
            if 0 <= idx < action_probs.shape[0]:
                next_phase = int(torch.argmax(action_probs[idx]).item())
            else:
                next_phase = data.current_phase

        return {
            "intersection_id": data.intersection_id,
            "next_phase": next_phase,
        }

    # ─── HEURİSTİC FALLBACK ─────────────────────────────
    # AI modeli yoksa matematiksel kural tabanlı karar
    directions = {
        0: data.north_count,
        1: data.south_count,
        2: data.east_count,
        3: data.west_count,
    }

    best_phase = max(directions, key=directions.get)
    max_cars = directions[best_phase]

    # Minimum 10 saniye yanma kuralı
    if data.phase_duration < 10.0:
        next_phase = data.current_phase
    elif max_cars == 0:
        next_phase = data.current_phase
    else:
        next_phase = best_phase

    return {
        "intersection_id": data.intersection_id,
        "next_phase": next_phase,
    }


# ==========================================
# GET /state — Dashboard İçin Durum Verisi
# ==========================================
@app.get("/state")
async def get_state():
    return state_dict