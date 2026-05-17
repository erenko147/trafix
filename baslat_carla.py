"""
TraFix + CARLA Ko-Simülasyon Başlatıcı

Önce CARLA'yı ayrı bir terminalde başlat:
    "C:\\Users\\Furkan\\Downloads\\CARLA_0.9.16\\CarlaUE4.exe" -RenderOffScreen

Sonra bu dosyayı çalıştır:
    python3.12 baslat_carla.py
    python3.12 baslat_carla.py --sumo-gui      # SUMO penceresini de aç
    python3.12 baslat_carla.py --model v2
"""
import sys, os, subprocess, argparse

HERE   = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

p = argparse.ArgumentParser()
p.add_argument("--model",    choices=["v2","v3","simple","v5"], default="v5")
p.add_argument("--port",     type=int, default=8000)
p.add_argument("--sumo-gui", action="store_true", help="SUMO GUI penceresiyle aç")
args = p.parse_args()

env = os.environ.copy()
env["TRAFIX_MODEL_VERSION"] = args.model
env["TRAFIX_API_PORT"]      = str(args.port)
env["SUMO_GUI"]             = "1" if args.sumo_gui else "0"

print("=" * 55)
print(f"  TraFix + CARLA   Model: {args.model.upper()}")
print(f"  Dashboard  →  http://127.0.0.1:{args.port}/")
print(f"  Kameralar  →  http://127.0.0.1:{args.port}/camera/0")
print(f"  CARLA durum→  http://127.0.0.1:{args.port}/carla/status")
print("=" * 55)

try:
    subprocess.run(
        [PYTHON, "-m", "uvicorn", "main_carla:app",
         "--host", "127.0.0.1", "--port", str(args.port), "--log-level", "warning"],
        cwd=HERE, env=env
    )
except KeyboardInterrupt:
    print("\nKapatılıyor.")
