"""
SUMO net.xml → OpenDRIVE (.xodr) Donusturucu
=============================================
Kullanicinin netedit'te tasarladigi SUMO agini CARLA'nin generate_opendrive_world()
ile yukleyebilecegi .xodr formatina donusturur.

Kullanim:
    python sumo/util/netconvert_sumo_to_xodr.py
    python sumo/util/netconvert_sumo_to_xodr.py --net-file sumo/map.net.xml
    python sumo/util/netconvert_sumo_to_xodr.py --net-file sumo/map.net.xml --output sumo/map.xodr
"""

import argparse
import os
import re
import subprocess
import sys


HERE         = os.path.dirname(os.path.abspath(__file__))
SUMO_DIR     = os.path.dirname(HERE)          # trafix/sumo/
PROJECT_ROOT = os.path.dirname(SUMO_DIR)      # trafix/

DEFAULT_NET  = os.path.join(SUMO_DIR, "map.net.xml")
DEFAULT_XODR = os.path.join(SUMO_DIR, "map.xodr")


def _find_netconvert():
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        ext       = ".exe" if sys.platform == "win32" else ""
        candidate = os.path.join(sumo_home, "bin", f"netconvert{ext}")
        if os.path.exists(candidate):
            return candidate
    return "netconvert"


def convert(net_path: str, output_path: str, verbose: bool = False) -> str:
    """
    SUMO net.xml → OpenDRIVE (.xodr) donusumu.

    Koordinatlari oldugu gibi korumak icin offset normalizasyonu devre disi.
    Donusum basarili olursa output_path'i dondurur, hata varsa sys.exit eder.
    """
    net_path    = os.path.abspath(net_path)
    output_path = os.path.abspath(output_path)

    if not os.path.exists(net_path):
        sys.exit(f"[CONV] HATA: Giris dosyasi bulunamadi: {net_path}")

    netconvert = _find_netconvert()

    cmd = [
        netconvert,
        "--sumo-net-file",              net_path,
        "--opendrive-output",           output_path,
        "--offset.disable-normalization",
        "--no-warnings",
    ]

    print(f"[CONV] Donusturuluyor: {net_path}")
    print(f"[CONV] Cikis        : {output_path}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[CONV] HATA (netconvert):\n{result.stderr}")
        sys.exit(1)

    if verbose and result.stdout:
        print(result.stdout[:1000])

    _fix_signal_z(output_path)
    _boost_lane_markings(output_path)
    _enrich_roadmark_lines(output_path)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[CONV] Basarili — {output_path} ({size_kb:.1f} KB)")
    return output_path


def _fix_signal_z(xodr_path: str):
    """
    netconvert trafik isareti yüksekligini zOffset="5" olarak yazar.
    Bu deger diregi yoldan 5m yukari kaldirarak lambayı havada birakir.
    Sifira indirince CARLA actor'u zemin hizasinda dogru konumlanir.
    height cok buyukse isik kutusu yol uzerinde havada gorunebilir — makul ust sinir.
    """
    with open(xodr_path, "r", encoding="utf-8") as f:
        content = f.read()

    fixed, count_z = re.subn(
        r'(<signal\b[^>]*)\bzOffset="[0-9.+-]+"',
        r'\1zOffset="0"',
        content,
    )

    height_capped = [0]
    out_lines = []
    for line in fixed.split("\n"):
        if "<signal" in line and 'height="' in line:

            def _cap_h(m):
                try:
                    hv = float(m.group(1))
                except ValueError:
                    return m.group(0)
                if hv > 2.5:
                    height_capped[0] += 1
                    return 'height="2.5"'
                return m.group(0)

            line = re.sub(r'height="([0-9.+-]+)"', _cap_h, line)
        out_lines.append(line)
    fixed2 = "\n".join(out_lines)
    count_h = height_capped[0]

    if count_z or count_h:
        with open(xodr_path, "w", encoding="utf-8") as f:
            f.write(fixed2)
        if count_z:
            print(f"[CONV] {count_z} sinyal zOffset degeri 0'a duzeltildi.")
        if count_h:
            print(f"[CONV] {count_h} sinyal height degeri sinirlandi (max 2.5m).")


def _boost_lane_markings(xodr_path: str):
    """
    netconvert cok ince roadMark (or. 0.13m) uretir; CARLA'da neredeyse gorunmez.
    Genisligi artirir, standard rengi beyaza ceker (yuzeyde daha okunur).
    """
    with open(xodr_path, "r", encoding="utf-8") as f:
        content = f.read()

    min_w = 0.24
    widened = [0]

    def _w(m):
        try:
            wv = float(m.group(1))
        except ValueError:
            return m.group(0)
        if wv < min_w:
            widened[0] += 1
            return f'width="{min_w:.2f}"'
        return m.group(0)

    out_lines = []
    for line in content.split("\n"):
        if "<roadMark" in line:
            line = re.sub(r'width="([0-9.+-]+)"', _w, line)
            if 'color="standard"' in line:
                line = line.replace('color="standard"', 'color="white"', 1)
        out_lines.append(line)

    new_content = "\n".join(out_lines)
    if new_content != content:
        with open(xodr_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(
            f"[CONV] Serit cizgileri guclendirildi (min genislik {min_w}m), "
            f"{widened[0]} roadMark kalinlastirildi."
        )


def _enrich_roadmark_lines(xodr_path: str):
    """
    CARLA OpenDRIVE standalone modunda broken roadMark cizgilerinin gorunmesi
    icin roadMark altinda line elemani daha guvenilir sonuclar verir.
    """
    with open(xodr_path, "r", encoding="utf-8") as f:
        content = f.read()

    roadmark_re = re.compile(r'(?P<indent>\s*)<roadMark(?P<attrs>[^>]*)/>')
    added = [0]

    def _replace(m):
        attrs = m.group("attrs")
        indent = m.group("indent")

        width_m = re.search(r'width="([0-9.+-]+)"', attrs)
        width = width_m.group(1) if width_m else "0.24"

        type_m = re.search(r'type="([^"]+)"', attrs)
        mark_type = type_m.group(1) if type_m else "solid"

        # SUMO netconvert ciktisinda basincli tipler: solid/broken.
        if mark_type == "broken":
            line = (
                f'{indent}    <line length="3.0" space="6.0" '
                f'tOffset="0.0" width="{width}" sOffset="0.0"/>'
            )
        else:
            line = (
                f'{indent}    <line length="6.0" space="0.0" '
                f'tOffset="0.0" width="{width}" sOffset="0.0"/>'
            )

        added[0] += 1
        return f"{indent}<roadMark{attrs}>\n{line}\n{indent}</roadMark>"

    updated = roadmark_re.sub(_replace, content)

    if updated != content:
        with open(xodr_path, "w", encoding="utf-8") as f:
            f.write(updated)
        print(f"[CONV] {added[0]} roadMark icin line elemani eklendi.")


def main():
    p = argparse.ArgumentParser(
        description="SUMO net.xml → OpenDRIVE (.xodr) donusturucu",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--net-file", default=DEFAULT_NET,
        help="Giris SUMO ag dosyasi (.net.xml)",
    )
    p.add_argument(
        "--output", default=DEFAULT_XODR,
        help="Cikis OpenDRIVE dosyasi (.xodr)",
    )
    p.add_argument("--verbose", action="store_true", help="netconvert ciktisini goster")
    args = p.parse_args()

    convert(args.net_file, args.output, verbose=args.verbose)


if __name__ == "__main__":
    main()
