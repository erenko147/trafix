"""
Debug / inspection routes.

GET /state           — raw intersection state dict
GET /last_decisions  — most recent AI phase decisions
"""

from fastapi import APIRouter

import backend.api.state as state

router = APIRouter()


@router.get("/state")
async def get_state():
    return state.state_dict


@router.get("/last_decisions")
async def get_last_decisions():
    return {"decisions": state.last_decisions_cache}
