"""
CARLA OpenDRIVE (.xodr) → SUMO net.xml Donusturucu
=====================================================
CARLA harita dosyasindan SUMO'nun anlayacagi bir ag olusturur.
netconvert aracini (SUMO_HOME/bin/) kullanir.

Kullanim:
    python sumo/util/netconvert_carla.py --xodr-file Town04.xodr
    python sumo/util/netconvert_carla.py --xodr-file Town04.xodr --output Town04.net.xml
    python sumo/util/netconvert_carla.py --xodr-file Town04.xodr --no-sidewalks --no-crossings
"""

import argparse
import os
import subprocess
import sys


def _find_netconvert():
    """SUMO_HOME/bin/netconvert veya PATH'deki netconvert'i bul."""
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        ext = ".exe" if sys.platform == "win32" else ""
        candidate = os.path.join(sumo_home, "bin", f"netconvert{ext}")
        if os.path.exists(candidate):
            return candidate
    # PATH'e guvenir
    return "netconvert"


def convert(
    xodr_path: str,
    output_path: str,
    guess_sidewalks: bool = True,
    sidewalk_width: float = 2.0,
    crossings: bool = True,
    no_turnarounds: bool = True,
    verbose: bool = False,
):
    """netconvert cagrisini yap ve SUMO ag dosyasini olustur."""
    xodr_path   = os.path.abspath(xodr_path)
    output_path = os.path.abspath(output_path)

    if not os.path.exists(xodr_path):
        sys.exit(f"HATA: Dosya bulunamadi: {xodr_path}")

    netconvert = _find_netconvert()

    cmd = [
        netconvert,
        "--opendrive-files",      xodr_path,
        "--output-file",          output_path,
        "--geometry.remove",
        "--roundabouts.guess",
        "--ramps.guess",
        "--junctions.join",
        "--tls.guess-signals",
        "--tls.discard-simple",
        "--tls.join",
        "--output.original-names",
        "--output.street-names",
        "--opendrive.import-all-lanes",
    ]

    if guess_sidewalks:
        cmd += [
            "--sidewalks.guess",
            "--sidewalks.guess.min-speed", "0",
            "--default.sidewalk-width",    str(sidewalk_width),
        ]
    if crossings:
        cmd += ["--crossings.guess"]
    if no_turnarounds:
        cmd += ["--no-turnarounds"]

    print(f"[CONV] Calistiriliyor:\n  {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[CONV] HATA (netconvert):\n{result.stderr}")
        sys.exit(1)

    print(f"[CONV] Basarili → {output_path}")
    if verbose and result.stdout:
        print(result.stdout[:1000])


def main():
    p = argparse.ArgumentParser(
        description="CARLA .xodr → SUMO .net.xml donusturucu",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--xodr-file",     required=True,           help="Giris .xodr dosyasi")
    p.add_argument("--output",        default=None,            help="Cikis .net.xml (belirtilmezse xodr ile ayni isim)")
    p.add_argument("--no-sidewalks",  action="store_true",     help="Kaldirim tahmini yapma")
    p.add_argument("--sidewalk-width",type=float, default=2.0, help="Kaldirim genisligi (metre)")
    p.add_argument("--no-crossings",  action="store_true",     help="Gecit tahmini yapma")
    p.add_argument("--no-turnarounds",action="store_true", default=True, help="U-donus baglantisi ekleme")
    p.add_argument("--verbose",       action="store_true",     help="netconvert ciktisini goster")
    args = p.parse_args()

    output = args.output
    if output is None:
        output = os.path.splitext(args.xodr_file)[0] + ".net.xml"

    convert(
        xodr_path=args.xodr_file,
        output_path=output,
        guess_sidewalks=not args.no_sidewalks,
        sidewalk_width=args.sidewalk_width,
        crossings=not args.no_crossings,
        no_turnarounds=args.no_turnarounds,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
