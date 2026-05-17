"""
carla_integration — TraFix ↔ CARLA köprüsünün paylaşılan durumu.

FastAPI ve CARLA bridge thread'i bu modülü import ederek
FRAME_BUFFER ile CARLA_STATUS üzerinden iletişim kurar.
"""
import threading

# intersection_id (int) → bytes (JPEG)
FRAME_BUFFER: dict[int, bytes] = {}
_frame_lock = threading.Lock()


def set_frame(intersection_id: int, jpeg_bytes: bytes) -> None:
    with _frame_lock:
        FRAME_BUFFER[intersection_id] = jpeg_bytes


def get_frame(intersection_id: int) -> bytes | None:
    with _frame_lock:
        return FRAME_BUFFER.get(intersection_id)


CARLA_STATUS: dict = {
    "connected": False,
    "map": None,
    "num_vehicles": 0,
    "num_intersections": 0,
    "error": None,
}
