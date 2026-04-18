"""
TraFix - Tek tikla baslat
==========================
1. FastAPI AI backend'i baslat  (port 8001)
2. SUMO + Unity bridge'i baslat (port 8765, sumo-gui ile)

Kullanim:
    python baslat.py
    python baslat.py --model v3
    python baslat.py C:/harita/baska.sumocfg
    python baslat.py --model v3 C:/harita/baska.sumocfg
"""

import sys, os, subprocess, time, threading, argparse

HERE   = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

parser = argparse.ArgumentParser(description="TraFix baslat")
parser.add_argument("--model", choices=["v2", "v3"], default="v2",
                    help="AI model versiyonu (varsayilan: v2)")
parser.add_argument("sumocfg", nargs="?", default=None,
                    help="Opsiyonel .sumocfg dosya yolu")
args = parser.parse_args()

def run_fastapi():
    env = os.environ.copy()
    env["TRAFIX_MODEL_VERSION"] = args.model
    cmd = [PYTHON, "-m", "uvicorn", "backend.main:app",
           "--port", "8001", "--log-level", "warning"]
    print(f"[BASLAT] FastAPI baslatiliyor... (port 8001, model={args.model.upper()})")
    subprocess.run(cmd, cwd=HERE, env=env)

def run_bridge():
    bridge_args = [PYTHON, os.path.join("sumo", "sumo_unity_bridge.py")]
    if args.sumocfg:
        bridge_args.append(args.sumocfg)
    print("[BASLAT] SUMO Bridge baslatiliyor...")
    subprocess.run(bridge_args, cwd=HERE)

if __name__ == "__main__":
    # FastAPI arka planda
    t = threading.Thread(target=run_fastapi, daemon=True)
    t.start()

    # SUMO bridge baslamadan once FastAPI'nin ayaga kalkmasini bekle
    time.sleep(2)

    # Bridge on planda (kapatinca ikisi birden kapanir)
    try:
        run_bridge()
    except KeyboardInterrupt:
        print("\n[BASLAT] Kapatiliyor.")
