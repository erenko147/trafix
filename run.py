"""
TraFix Backend Başlatıcı
========================
Model versiyonunu komut satırından seçerek sunucuyu başlatır.

Kullanım:
  python run.py                    # v6 modeli (varsayılan), port 8000
  python run.py --model v6         # TraFix v6: GRU+GAT, 6 faz, 3 şerit
  python run.py --model v6 --port 8001
  python run.py --model v6 --reload    # geliştirme modunda
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
    choices=["v2", "v5", "v6"],
    default="v6",
    help="AI model versiyonu: v6 (GRU+GAT 6-faz 3-şerit, varsayılan), v5 (GRU+GAT 4-faz, eski), v2 (GCN+GRU 4-faz, eski)",
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
    "main:app",
    host=args.host,
    port=args.port,
    reload=args.reload,
)
