"""
TraFix Backend Başlatıcı
========================
Model versiyonunu komut satırından seçerek sunucuyu başlatır.

Kullanım:
  python run.py                  # v2 modeli, port 8000
  python run.py --model v3       # v3 (GConvGRU) modeli
  python run.py --model v2 --port 8001
  python run.py --model v3 --reload    # geliştirme modunda
"""

import argparse
import os
import uvicorn

parser = argparse.ArgumentParser(
    description="TraFix Backend",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--model",
    choices=["v2", "v3", "simple"],
    default="v2",
    help="AI model versiyonu: simple (demo), v2 (GCN+GRU), v3 (GConvGRU)",
)
parser.add_argument("--host", default="127.0.0.1", help="Sunucu adresi")
parser.add_argument("--port", type=int, default=8000, help="Port numarası")
parser.add_argument(
    "--reload",
    action="store_true",
    help="Geliştirme modunda otomatik yeniden yükleme",
)
args = parser.parse_args()

# Env var üzerinden model seçimini main.py'ye ilet
os.environ["TRAFIX_MODEL_VERSION"] = args.model
print(f"[TraFix] Model: {args.model.upper()} | {args.host}:{args.port}")

uvicorn.run(
    "backend.main:app",
    host=args.host,
    port=args.port,
    reload=args.reload,
)
