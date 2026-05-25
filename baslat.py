"""
TraFix - Tek tikla baslat
==========================
1. FastAPI + frontend'i baslat  (port 8000)
     Dashboard : http://127.0.0.1:8000/
     API docs  : http://127.0.0.1:8000/docs
2. SUMO simülasyonunu baslat    (sumo-gui ile)

Kullanim:
    python baslat.py                          # v6 modeli (varsayilan)
    python baslat.py --model v6               # TraFix v6: GRU+GAT, 6 faz, 3 serit
    python baslat.py --no-gui
    python baslat.py /yol/dosya.sumocfg
"""

import sys, os, subprocess, time, threading, argparse

HERE   = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

parser = argparse.ArgumentParser(description="TraFix baslat")
parser.add_argument("--model", choices=["v2", "v5", "v6"], default="v6",
                    help="AI model versiyonu: v6 (varsayilan), v5 (eski 4-faz), v2 (eski GCN+GRU)")
parser.add_argument("--port", type=int, default=8000,
                    help="Sunucu portu (varsayilan: 8000)")
parser.add_argument("--no-gui", action="store_true",
                    help="SUMO GUI olmadan calistir (headless)")
parser.add_argument("sumocfg", nargs="?", default=None,
                    help="Opsiyonel .sumocfg dosya yolu")
args = parser.parse_args()


def run_fastapi():
    env = os.environ.copy()
    env["TRAFIX_MODEL_VERSION"] = args.model
    # main:app — root main.py bundles backend routes + dashboard/architecture pages
    cmd = [
        PYTHON, "-m", "uvicorn", "main:app",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--log-level", "warning",
    ]
    print(f"[BASLAT] Sunucu baslatiliyor... (port={args.port}, model={args.model.upper()})")
    subprocess.run(cmd, cwd=HERE, env=env)


def run_sumo():
    sumo_script = os.path.join(HERE, "sumo", "run_sumo_live.py")
    if not os.path.exists(sumo_script):
        print(f"[BASLAT] SUMO script bulunamadi: {sumo_script}")
        return

    env = os.environ.copy()
    env["TRAFIX_API_PORT"] = str(args.port)

    cmd = [PYTHON, sumo_script]
    if args.sumocfg:
        cmd.append(args.sumocfg)
    if args.no_gui:
        cmd.append("--no-gui")

    print(f"[BASLAT] SUMO simülasyonu baslatiliyor... (API port={args.port})")
    subprocess.run(cmd, cwd=HERE, env=env)


if __name__ == "__main__":
    print("=" * 55)
    print(f"  TraFix — Model: {args.model.upper()}")
    print(f"  Dashboard    : http://127.0.0.1:{args.port}/")
    print(f"  Architecture : http://127.0.0.1:{args.port}/architecture")
    print(f"  API docs     : http://127.0.0.1:{args.port}/docs")
    print(f"  SUMO GUI     : {'hayir (headless)' if args.no_gui else 'evet'}")
    print("=" * 55)

    # FastAPI + frontend arka planda
    t_api = threading.Thread(target=run_fastapi, daemon=True)
    t_api.start()

    # Backend ayaga kalkmadan SUMO baslamasin
    time.sleep(3)

    # SUMO on planda (kapatinca hepsi kapanir)
    try:
        run_sumo()
    except KeyboardInterrupt:
        print("\n[BASLAT] Kapatiliyor.")
