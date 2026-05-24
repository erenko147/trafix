"""
Backend entry point — re-exports `app` from the modular API package.

Kept at this path for backward compatibility with:
  main.py          → `from backend.main import app`
  run.py           → `uvicorn main:app --reload`
  baslat.py        → same uvicorn invocation
"""

from backend.api.app import app  # noqa: F401 — re-export

__all__ = ["app"]
