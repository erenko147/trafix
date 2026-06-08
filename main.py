"""
Bridge layer — mounts backend routes and serves the dashboard.
Run: uvicorn main:app --reload   or   python run.py
"""
from backend.main import app
from fastapi.responses import FileResponse
import os

BASE_DIR = os.path.dirname(__file__)


@app.get("/")
async def dashboard():
    return FileResponse(os.path.join(BASE_DIR, "frontend", "dashboard.html"))


@app.get("/architecture")
async def architecture():
    return FileResponse(os.path.join(BASE_DIR, "frontend", "index.html"))


@app.get("/emergency")
async def emergency():
    return FileResponse(os.path.join(BASE_DIR, "frontend", "emergency.html"))


