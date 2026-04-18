"""
TraFix - Tek tikla baslat  [CARLA modu]
========================================
1. (Opsiyonel) map.net.xml → map.xodr donusturme
2. FastAPI AI backend'i baslat  (port 8001)
3. SUMO + CARLA bridge'i baslat (ozel harita ile)

Kullanim:
    python baslat.py                            # harita zaten donusturulmusse
    python baslat.py --generate-map             # net.xml → xodr otomatik don.
    python baslat.py --generate-map --sumo-gui  # SUMO GUI ile
    python baslat.py --model v3                 # v3 model
    python baslat.py --no-carla                 # sadece SUMO + FastAPI (test)
    python baslat.py --carla-host 192.168.1.10  # uzak CARLA sunucusu

Not: --generate-map ilk kurulumda bir kez calistirilmasi yeterlidir.
     map.xodr mevcut olduktan sonra sadece 'python baslat.py' yeterlidir.
"""

import sys
import os
import subprocess
import time
import threading
import argparse

HERE   = os.path.dirname(os.path.abspath(__file__))

SUMO_DIR     = os.path.join(HERE, "sumo")
DEFAULT_NET  = os.path.join(SUMO_DIR, "map.net.xml")
DEFAULT_XODR = os.path.join(SUMO_DIR, "map.xodr")

# .venv/Scripts/python.exe varsa onu kullan (CARLA + Python 3.12 ortami)
_VENV_PYTHON = os.path.join(HERE, ".venv", "Scripts", "python.exe")
PYTHON = _VENV_PYTHON if os.path.exists(_VENV_PYTHON) else sys.executable

# Venv yoksa erken uyari ver (sadece CARLA modunda)
_USING_VENV = os.path.exists(_VENV_PYTHON)


# ── Argümanlar ───────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="TraFix CARLA modu baslat",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--model", choices=["v2", "v3"], default="v2",
    help="AI model versiyonu",
)
parser.add_argument(
    "--sumo-cfg", default=None,
    help="Opsiyonel .sumocfg dosya yolu (belirtilmezse sumo/demo.sumocfg)",
)
parser.add_argument(
    "--sumo-gui", action="store_true",
    help="SUMO'yu sumo-gui ile baslat",
)
parser.add_argument(
    "--carla-host", default="localhost",
    help="CARLA sunucu IP adresi",
)
parser.add_argument(
    "--carla-port", type=int, default=2000,
    help="CARLA TCP portu",
)
parser.add_argument(
    "--step-length", type=float, default=0.05,
    help="Simulasyon adim suresi saniye (SUMO ve CARLA icin esit olmali)",
)
parser.add_argument(
    "--generate-map", action="store_true",
    help=(
        "Baslatmadan once map.net.xml → map.xodr donusumunu calistir.\n"
        "Ilk kurulumda veya harita degistiginde kullanin."
    ),
)
parser.add_argument(
    "--net-file", default=DEFAULT_NET,
    help="Donusturulecek SUMO ag dosyasi (.net.xml)",
)
parser.add_argument(
    "--opendrive-file", default=None,
    help="Kullanilacak .xodr dosyasi (belirtilmezse sumo/map.xodr otomatik aranir)",
)
parser.add_argument(
    "--no-carla", action="store_true",
    help="CARLA olmadan sadece SUMO + FastAPI baslat (test/fallback modu)",
)
args = parser.parse_args()


# ── Harita donusum adimi ─────────────────────────────────────────────────────

def generate_map():
    """map.net.xml → map.xodr donusumunu gerceklestir."""
    conv_script = os.path.join(HERE, "sumo", "util", "netconvert_sumo_to_xodr.py")
    net_file    = args.net_file
    xodr_file   = args.opendrive_file or DEFAULT_XODR

    if not os.path.exists(net_file):
        print(f"[BASLAT] HATA: net dosyasi bulunamadi: {net_file}")
        sys.exit(1)

    cmd = [
        PYTHON, conv_script,
        "--net-file", net_file,
        "--output",   xodr_file,
    ]
    print(f"[BASLAT] Harita donusturuluyor: {net_file} → {xodr_file}")
    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        print("[BASLAT] Harita donusturme basarisiz! Devam edilemiyor.")
        sys.exit(1)
    print(f"[BASLAT] Harita hazir: {xodr_file}\n")
    return xodr_file


# ── Calistiricilar ───────────────────────────────────────────────────────────

def run_fastapi():
    env = os.environ.copy()
    env["TRAFIX_MODEL_VERSION"] = args.model
    cmd = [
        PYTHON, "-m", "uvicorn", "backend.main:app",
        "--port", "8001", "--log-level", "warning",
    ]
    print(f"[BASLAT] FastAPI baslatiliyor... (port 8001, model={args.model.upper()})")
    subprocess.run(cmd, cwd=HERE, env=env)


def run_carla_bridge(xodr_file=None):
    bridge = os.path.join(HERE, "sumo", "carla_bridge", "run_synchronization.py")
    cmd    = [PYTHON, bridge]

    if args.sumo_cfg:
        cmd += ["--sumo-cfg", args.sumo_cfg]
    if args.sumo_gui:
        cmd += ["--sumo-gui"]

    # Ozel harita
    effective_xodr = xodr_file or args.opendrive_file
    if effective_xodr and os.path.exists(effective_xodr):
        cmd += ["--opendrive-file", effective_xodr]
    elif os.path.exists(DEFAULT_XODR):
        cmd += ["--opendrive-file", DEFAULT_XODR]

    cmd += [
        "--carla-host",  args.carla_host,
        "--carla-port",  str(args.carla_port),
        "--step-length", str(args.step_length),
        "--tls-manager", "carla",
        "--sync-vehicle-color",
    ]

    xodr_info = effective_xodr or DEFAULT_XODR
    print(
        f"[BASLAT] SUMO-CARLA Bridge baslatiliyor...\n"
        f"         CARLA   : {args.carla_host}:{args.carla_port}\n"
        f"         Adim    : {args.step_length}s\n"
        f"         Harita  : {xodr_info}"
    )
    subprocess.run(cmd, cwd=HERE)


def run_sumo_only():
    """CARLA olmadan eski Unity bridge ile calistir (test/fallback)."""
    legacy = os.path.join(HERE, "sumo", "legacy", "sumo_unity_bridge.py")
    if not os.path.exists(legacy):
        legacy = os.path.join(HERE, "sumo", "sumo_unity_bridge.py")
    cmd = [PYTHON, legacy]
    if args.sumo_cfg:
        cmd.append(args.sumo_cfg)
    print("[BASLAT] [TEST MODU] Eski bridge baslatiliyor (CARLA yok)...")
    subprocess.run(cmd, cwd=HERE)


# ── Giris noktasi ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Adim 1: Harita donusum (opsiyonel)
    xodr_result = None
    if args.generate_map:
        xodr_result = generate_map()
    elif not os.path.exists(DEFAULT_XODR):
        print(
            "[BASLAT] UYARI: sumo/map.xodr bulunamadi.\n"
            "         Ilk kurulumda --generate-map bayragi ile calistirin:\n"
            "           python baslat.py --generate-map\n"
            "         Devam ediliyor (CARLA mevcut haritasini kullanacak)...\n"
        )

    # Venv kontrolu (CARLA modunda)
    if not args.no_carla and not _USING_VENV:
        print(
            "[BASLAT] UYARI: .venv bulunamadi — Python 3.12 + CARLA ortami kurulmamis.\n"
            "         Lutfen once asagidaki adimlari tamamlayin:\n"
            "\n"
            "  1. Python 3.12 kur:\n"
            "       winget install -e --id Python.Python.3.12\n"
            "\n"
            "  2. Ortami kur (bat dosyasina cift tikla):\n"
            f"       {os.path.join(HERE, 'kurulum_python312.bat')}\n"
            "\n"
            "  3. Sonra tekrar calistir:\n"
            "       python baslat.py --sumo-gui\n"
        )
        sys.exit(1)

    # Adim 2: FastAPI arka planda
    t = threading.Thread(target=run_fastapi, daemon=True)
    t.start()

    # Bridge baslamadan once FastAPI'nin ayaga kalkmasini bekle
    time.sleep(2)

    # Adim 3: Bridge
    try:
        if args.no_carla:
            run_sumo_only()
        else:
            run_carla_bridge(xodr_file=xodr_result)
    except KeyboardInterrupt:
        print("\n[BASLAT] Kapatiliyor.")
