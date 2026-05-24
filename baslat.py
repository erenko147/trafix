"""
TraFix V6 — Tek tıkla başlat
==============================
1. FastAPI + frontend'i başlatır  (port 8000)
     Dashboard : http://127.0.0.1:8000/
     API docs  : http://127.0.0.1:8000/docs
2. SUMO simülasyonunu başlatır    (sumo-gui ile)

Kullanım:
    python baslat.py
    python baslat.py --no-gui
    python baslat.py --port 8001
    python baslat.py /yol/dosya.sumocfg
"""

import sys, os, subprocess, time, threading, argparse

HERE   = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

parser = argparse.ArgumentParser(description="TraFix V6 başlat")
parser.add_argument("--port", type=int, default=8000,
                    help="Sunucu portu (varsayılan: 8000)")
parser.add_argument("--no-gui", action="store_true",
                    help="SUMO GUI olmadan çalıştır (headless)")
parser.add_argument("sumocfg", nargs="?", default=None,
                    help="Opsiyonel .sumocfg dosya yolu")
args = parser.parse_args()


def run_fastapi():
    cmd = [
        PYTHON, "-m", "uvicorn", "main:app",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--log-level", "warning",
    ]
    print(f"[BASLAT] Sunucu başlatılıyor... (port={args.port})")
    subprocess.run(cmd, cwd=HERE)


def run_sumo():
    sumo_script = os.path.join(HERE, "sumo", "run_sumo_live.py")
    if not os.path.exists(sumo_script):
        print(f"[BASLAT] SUMO script bulunamadı: {sumo_script}")
        return

    env = os.environ.copy()
    env["TRAFIX_API_PORT"] = str(args.port)

    cmd = [PYTHON, sumo_script]
    if args.sumocfg:
        cmd.append(args.sumocfg)
    if args.no_gui:
        cmd.append("--no-gui")

    print(f"[BASLAT] SUMO simülasyonu başlatılıyor... (API port={args.port})")
    subprocess.run(cmd, cwd=HERE, env=env)


if __name__ == "__main__":
    print("=" * 50)
    print(f"  TraFix V6")
    print(f"  Dashboard    : http://127.0.0.1:{args.port}/")
    print(f"  Architecture : http://127.0.0.1:{args.port}/architecture")
    print(f"  API docs     : http://127.0.0.1:{args.port}/docs")
    print(f"  SUMO GUI     : {'hayır (headless)' if args.no_gui else 'evet'}")
    print("=" * 50)

    t_api = threading.Thread(target=run_fastapi, daemon=True)
    t_api.start()

    time.sleep(3)

    try:
        run_sumo()
    except KeyboardInterrupt:
        print("\n[BASLAT] Kapatılıyor.")
