"""
TraFix - Tek tikla baslat
==========================
TraFix v5 modelini SUMO simulasyonu + canli dashboard ile calistirir.

Kullanim:
    python baslat.py
    python baslat.py --scenarios 10
    python baslat.py --checkpoint trafix_v5/checkpoints/stage3_ep1200.pt
    python baslat.py --port 8080
    python baslat.py --greedy
"""

import sys, os, argparse
from pathlib import Path

HERE   = Path(__file__).resolve().parent
PYTHON = sys.executable

# SUMO ortam degiskenlerini ayarla (henuz set edilmemisse)
SUMO_FRAMEWORK = Path("/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO")
if "SUMO_HOME" not in os.environ and (SUMO_FRAMEWORK / "share/sumo").exists():
    os.environ["SUMO_HOME"] = str(SUMO_FRAMEWORK / "share/sumo")
sumo_bin = str(SUMO_FRAMEWORK / "bin")
if sumo_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = sumo_bin + os.pathsep + os.environ.get("PATH", "")

DEFAULT_CHECKPOINT = str(HERE / "trafix_v5" / "checkpoints" / "trafix_v5_final.pt")
DEFAULT_SUMO_CFG   = str(HERE / "sumo" / "training.sumocfg")
DEFAULT_NET_FILE   = str(HERE / "sumo" / "map.net.xml")

parser = argparse.ArgumentParser(
    description="TraFix v5 — tek tikla baslat",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                    help="Kullanilacak model checkpoint (.pt)")
parser.add_argument("--scenarios", type=int, default=10,
                    help="Kac senaryo calistirilacak")
parser.add_argument("--max-steps", type=int, default=3600,
                    help="Senaryo basina maksimum adim")
parser.add_argument("--port", type=int, default=8000,
                    help="Dashboard portu (tarayicida http://localhost:PORT)")
parser.add_argument("--greedy", action="store_true",
                    help="Stokastik yerine greedy (argmax) politika kullan")
parser.add_argument("--sumo-cfg", default=DEFAULT_SUMO_CFG)
parser.add_argument("--net-file", default=DEFAULT_NET_FILE)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

if __name__ == "__main__":
    # sys.path'e proje koku ekle
    sys.path.insert(0, str(HERE))

    # SUMO TraCI yolunu ekle
    sumo_tools = os.path.join(os.environ.get("SUMO_HOME", ""), "tools")
    if os.path.isdir(sumo_tools):
        sys.path.append(sumo_tools)

    # trafix_v5 modüllerini içe aktar
    from trafix_v5.eval_stage3 import evaluate
    from trafix_v5 import api as _api

    print(f"[BASLAT] Dashboard: http://localhost:{args.port}")
    print(f"[BASLAT] Checkpoint: {args.checkpoint}")
    print(f"[BASLAT] {args.scenarios} senaryo calistirilacak...")
    print("[BASLAT] Durdurmak icin CTRL+C")

    # eval_stage3.evaluate() beklentisine uygun Namespace olustur
    import argparse as _ap
    eval_args = _ap.Namespace(
        checkpoint=args.checkpoint,
        sumo_cfg=args.sumo_cfg,
        net_file=args.net_file,
        scenarios=args.scenarios,
        max_steps=args.max_steps,
        greedy=args.greedy,
        gui=False,
        seed=args.seed,
        serve=True,
        port=args.port,
    )

    try:
        evaluate(eval_args)
    except KeyboardInterrupt:
        print("\n[BASLAT] Kapatiliyor.")
