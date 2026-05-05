"""
TraFix v5 — Dashboard API
==========================
FastAPI sunucusu: eval ve training döngüsünden gerçek zamanlı state alır,
eski dashboard.html ile uyumlu /state ve /last_decisions endpoint'lerini sunar.

Kullanım (doğrudan çağrılmaz; eval_stage3.py veya stage3_train_ppo.py tarafından başlatılır):
  from trafix_v5.api import update_state_batch, start_server
"""

import threading
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

_lock = threading.Lock()

shared_state: dict = {str(i): {
    "intersection_id": str(i),
    "north_count": 0,
    "south_count": 0,
    "east_count": 0,
    "west_count": 0,
    "queue_length": 0.0,
    "current_phase": 0,
    "phase_duration": 0.0,
} for i in range(5)}

last_decisions: list = []

app = FastAPI(title="TraFix v5 Dashboard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    path = _FRONTEND_DIR / "dashboard.html"
    if path.exists():
        return HTMLResponse(path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>dashboard.html bulunamadı</h1>", status_code=404)


@app.get("/architecture", response_class=HTMLResponse)
def architecture():
    path = _FRONTEND_DIR / "index.html"
    if path.exists():
        return HTMLResponse(path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html bulunamadı</h1>", status_code=404)


@app.get("/state")
def get_state():
    with _lock:
        return JSONResponse(dict(shared_state))


@app.get("/last_decisions")
def get_last_decisions():
    with _lock:
        return JSONResponse(list(last_decisions))


def update_state_batch(
    x_next: torch.Tensor,
    actions: torch.Tensor,
    logits_list: List[torch.Tensor],
) -> None:
    """
    Her simülasyon adımından sonra çağrılır.

    x_next     : [J, obs_dim] — parse_sumo_observations() çıktısı (normalize edilmiş)
    actions    : [J]          — seçilen faz indeksleri
    logits_list: List[J × Tensor[1, 4]] — model.forward() çıktısı
    """
    new_decisions = []
    new_state = {}

    for j in range(x_next.shape[0]):
        obs = x_next[j].cpu()
        action = int(actions[j].item())

        if j < len(logits_list):
            probs = F.softmax(logits_list[j].squeeze(0).cpu(), dim=-1)
            confidence = float(probs.max().item())
        else:
            confidence = 0.0

        new_state[str(j)] = {
            "intersection_id": str(j),
            "north_count":  int(obs[0].item() * 30),
            "south_count":  int(obs[1].item() * 30),
            "east_count":   int(obs[2].item() * 30),
            "west_count":   int(obs[3].item() * 30),
            "queue_length": float(obs[4].item() * 150),
            "current_phase": action,
            "phase_duration": float(obs[9].item() * 120),
        }

        new_decisions.append({
            "intersection_id": j,
            "next_phase": action,
            "confidence": round(confidence, 4),
        })

    with _lock:
        shared_state.update(new_state)
        last_decisions.clear()
        last_decisions.extend(new_decisions)


def start_server(port: int = 8000) -> None:
    """Uvicorn'u arka plan thread'inde başlatır (daemon → ana process kapanınca durur)."""
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    print(f"[TraFix API] Dashboard: http://localhost:{port}")
