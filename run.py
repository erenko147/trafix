"""
TraFix Backend Başlatıcı (TraFix V6)
=====================================
Sunucuyu başlatır.

Kullanım:
  python run.py                    # port 8000
  python run.py --port 8001
  python run.py --reload           # geliştirme modunda
"""

import argparse
import os
import uvicorn

parser = argparse.ArgumentParser(
    description="TraFix Backend (V6)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--host", default="127.0.0.1", help="Sunucu adresi")
parser.add_argument("--port", type=int, default=8000, help="Port numarası")
parser.add_argument(
    "--reload",
    action="store_true",
    help="Geliştirme modunda otomatik yeniden yükleme",
)
args = parser.parse_args()

print(f"[TraFix V6] {args.host}:{args.port}")

uvicorn.run(
    "main:app",
    host=args.host,
    port=args.port,
    reload=args.reload,
)
