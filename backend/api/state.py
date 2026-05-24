"""
Global mutable state shared across API routes.

All AI model references and intersection telemetry live here so routes
can import this module and mutate attributes without circular-import issues.
"""

from collections import deque

# ── API / model constants ─────────────────────────────────────────────────────

NUM_FEATURES     = 20   # 12 per-lane + queue + 6-phase one-hot + duration
NUM_ACTIONS      = 6
NUM_NODES        = 5
_WEIGHT_FILENAME = "trafix_v6/checkpoints/trafix_v6_final.pt"
_V6_T_WINDOW     = 30

# ── Intersection telemetry ────────────────────────────────────────────────────

state_dict: dict = {"0": {}, "1": {}, "2": {}, "3": {}, "4": {}}

# ── AI model ──────────────────────────────────────────────────────────────────

ai_agent       = None
_v6_window     = deque(maxlen=_V6_T_WINDOW)
_v6_governor   = None
_last_batch_step: int = -1

# ── Decision / emergency caches ───────────────────────────────────────────────

last_decisions_cache: list = []
emergency_events: deque   = deque(maxlen=200)
