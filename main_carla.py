"""
TraFix + CARLA — FastAPI giriş noktası
Mevcut AI backend'ine endpoint'ler ekler:
  GET  /camera/{id}        → MJPEG kamera akışı
  GET  /carla/status       → Ko-sim durumu
  GET  /camera/{id}/roi    → Kameranın ROI poligonunu getir
  POST /camera/{id}/roi    → ROI poligonu ayarla  body: {"polygon": [[x,y], ...]}
  DELETE /camera/{id}/roi  → ROI poligonunu kaldır
Startup'ta SUMO–CARLA ko-sim thread'ini başlatır.
"""
import asyncio
import os
import time
import threading
from typing import Any

from fastapi import Body
from fastapi.responses import StreamingResponse
from backend.main import app  # AI model + tüm mevcut route'lar

_stop = threading.Event()
_thread: threading.Thread | None = None


def _start(port: int) -> None:
    time.sleep(4)  # uvicorn hazır olsun
    from carla_cosim import run_cosim
    no_gui = os.environ.get("SUMO_GUI", "0") != "1"
    run_cosim(api_port=port, stop_event=_stop, no_sumo_gui=no_gui)


@app.on_event("startup")
async def _on_start() -> None:
    global _thread
    port = int(os.environ.get("TRAFIX_API_PORT", "8000"))
    _thread = threading.Thread(target=_start, args=(port,), daemon=True, name="cosim")
    _thread.start()
    print(f"[CoSim] Thread başlatıldı (port={port})")


@app.on_event("shutdown")
async def _on_stop() -> None:
    _stop.set()
    if _thread:
        _thread.join(timeout=5.0)


@app.get("/camera/{iid}", tags=["CARLA"])
async def camera_stream(iid: int):
    """CARLA kamera MJPEG akışı — <img src='/camera/0'> ile kullanılır."""
    from carla_integration import get_frame

    async def _gen():
        try:
            while True:
                frame = get_frame(iid)
                if frame:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        _gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
    )


@app.get("/carla/status", tags=["CARLA"])
async def carla_status():
    from carla_integration import CARLA_STATUS
    return CARLA_STATUS


@app.get("/camera/{iid}/roi", tags=["CARLA"])
async def get_roi(iid: int):
    """Kameranın mevcut ROI poligonunu döndürür."""
    from carla_cosim import ROI_POLYGONS
    return {"camera_id": iid, "polygon": ROI_POLYGONS.get(iid)}


@app.post("/camera/{iid}/roi", tags=["CARLA"])
async def set_roi(iid: int, body: dict[str, Any] = Body(...)):
    """
    ROI poligonu ayarlar.  İstenen sayıda köşe verilebilir (min 3).

    Body örneği (640×640 piksel koordinatları):
      {"polygon": [[100, 200], [300, 150], [450, 400], [80, 420]]}

    Polygon boş liste veya eksik ise ROI kaldırılır.
    """
    from carla_cosim import ROI_POLYGONS
    pts = body.get("polygon") or []
    if len(pts) >= 3:
        ROI_POLYGONS[iid] = [[int(p[0]), int(p[1])] for p in pts]
    else:
        ROI_POLYGONS.pop(iid, None)
    return {"camera_id": iid, "polygon": ROI_POLYGONS.get(iid)}


@app.delete("/camera/{iid}/roi", tags=["CARLA"])
async def delete_roi(iid: int):
    """Kameranın ROI poligonunu kaldırır; tüm alan sayılır."""
    from carla_cosim import ROI_POLYGONS
    ROI_POLYGONS.pop(iid, None)
    return {"camera_id": iid, "cleared": True}


@app.post("/carla/active_cameras", tags=["CARLA"])
async def set_active_cameras(body: dict[str, Any] = Body(...)):
    """
    Dashboard sayfa geçişinde çağrılır.
    Body: {"camera_ids": [0, 1, 2, 3, 4]}
    Listede olmayan kameralar bare-minimum moda geçer (YOLO/BGR atlanır).
    Boş liste → tüm kameralar aktif.
    """
    from carla_cosim import ACTIVE_CAMERA_IDS
    ids = body.get("camera_ids") or []
    ACTIVE_CAMERA_IDS.clear()
    ACTIVE_CAMERA_IDS.update(int(i) for i in ids)
    return {"active": sorted(ACTIVE_CAMERA_IDS)}


@app.get("/carla/active_cameras", tags=["CARLA"])
async def get_active_cameras():
    from carla_cosim import ACTIVE_CAMERA_IDS
    return {"active": sorted(ACTIVE_CAMERA_IDS)}
