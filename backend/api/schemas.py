from pydantic import BaseModel
from typing import List, Optional


class Telemetry(BaseModel):
    intersection_id: int
    north_left:    int = 0
    north_through: int = 0
    north_right:   int = 0
    south_left:    int = 0
    south_through: int = 0
    south_right:   int = 0
    east_left:     int = 0
    east_through:  int = 0
    east_right:    int = 0
    west_left:     int = 0
    west_through:  int = 0
    west_right:    int = 0
    queue_length:  float = 0.0
    current_phase: int = 0
    phase_duration: float = 0.0
    north_count: Optional[int] = None
    south_count: Optional[int] = None
    east_count:  Optional[int] = None
    west_count:  Optional[int] = None


class TelemetryBatch(BaseModel):
    step: int
    intersections: List[Telemetry]


class EmergencyEventBatch(BaseModel):
    events: List[dict]
