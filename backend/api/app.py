"""
FastAPI application factory.

Creates the `app` instance, registers middleware, mounts all routers,
and wires the startup hook that loads the AI model.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import telemetry, emergency, debug
from backend.ai.loader import load_model

app = FastAPI(title="TraFix API", version="6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(telemetry.router)
app.include_router(emergency.router)
app.include_router(debug.router)


@app.on_event("startup")
async def startup_event():
    load_model()
