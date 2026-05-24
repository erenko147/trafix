"""
Emergency vehicle routes.

POST /emergency_event    — receives completed preemption session metrics
GET  /emergency_metrics  — returns aggregate summary for the dashboard
"""

from fastapi import APIRouter

import backend.api.state as state
from backend.api.schemas import EmergencyEventBatch

router = APIRouter()


@router.post("/emergency_event")
async def receive_emergency_event(batch: EmergencyEventBatch):
    for ev in batch.events:
        state.emergency_events.append(ev)
    return {"ok": True, "stored": len(batch.events)}


@router.get("/emergency_metrics")
async def get_emergency_metrics():
    evs = list(state.emergency_events)
    n   = len(evs)
    if n:
        avg_transit      = round(sum(e.get("transit_steps", 0) for e in evs) / n, 1)
        total_waited     = sum(e.get("vehicles_waited", 0) for e in evs)
        total_wait_steps = sum(e.get("total_wait_steps", 0) for e in evs)
    else:
        avg_transit = total_waited = total_wait_steps = 0

    return {
        "events": evs,
        "summary": {
            "count":                n,
            "avg_transit_steps":    avg_transit,
            "total_vehicles_waited": total_waited,
            "total_wait_steps":     total_wait_steps,
        },
    }
