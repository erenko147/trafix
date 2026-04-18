"""
Bridge katmanı — backend'i başlatır ve dashboard'u serve eder.
Çalıştırmak için: uvicorn main:app --reload
"""
from backend.main import app
from fastapi.responses import FileResponse
import os

BASE_DIR = os.path.dirname(__file__)


# Dashboard'u tarayıcıdan erişilebilir kılar.
# Ziyaret: http://localhost:8000/
@app.get("/")
async def dashboard():
    return FileResponse(os.path.join(BASE_DIR, "frontend", "dashboard.html"))


# Architecture sayfası.
# Ziyaret: http://localhost:8000/architecture
@app.get("/architecture")
async def architecture():
    return FileResponse(os.path.join(BASE_DIR, "frontend", "index.html"))


# Calıstırma kodu
# cd /Users/bora/Desktop/trafix_4
# uvicorn main:app --reload

# http://localhost:8000/docs
#{
#  "intersection_id": 0,
#  "north_count": 23,
#  "south_count": 41,
#  "east_count": 12,
 # "west_count": 37,
 # "queue_length": 145.5,
 # "current_phase": 2,
#  "phase_duration": 47.0
#}


