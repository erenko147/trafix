"""
TraFix — Ağ Yeniden Oluşturucu
================================
Mevcut 5-kavşak topolojisini koruyarak yolları uzatır:
  İç yollar (kavşaklar arası) : ~80m  →  ~200m
  Fringe giriş/çıkış yolları  : ~40m  →  ~150m

Çalıştır:
  python sumo/rebuild_network.py

netconvert çıktısı doğrudan sumo/map.net.xml üzerine yazar.
Eski dosya sumo/map.net.xml.bak olarak saklanır.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ── netconvert binary bul ─────────────────────────────────────────────────────

def find_netconvert() -> str:
    sumo_home = os.environ.get("SUMO_HOME", "")
    candidates = []
    if sumo_home:
        candidates += [
            Path(sumo_home) / "bin" / "netconvert.exe",
            Path(sumo_home) / "bin" / "netconvert",
        ]
    candidates += [
        Path("C:/Program Files (x86)/Eclipse/Sumo/bin/netconvert.exe"),
        Path("C:/Program Files/Eclipse/Sumo/bin/netconvert.exe"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    found = shutil.which("netconvert") or shutil.which("netconvert.exe")
    if found:
        return found
    sys.exit("HATA: netconvert bulunamadı. SUMO_HOME ortam değişkenini ayarlayın.")


# ── Düğüm koordinatları ───────────────────────────────────────────────────────
#
# Orijinal topoloji (100m aralıklı grid):
#   J0(0,100) — J2(100,100) — J4(200,100)
#       |              |
#   J1(0,0)  — J3(100,0)
#
# Yeni topoloji (~220m aralıklı grid → iç yollar ~200m):
#   J0(0,220) — J2(220,220) — J4(440,220)
#       |               |
#   J1(0,0)  — J3(220,0)
#
# Fringe düğümleri TL kavşağından 160m uzağa taşındı (→ yol ~150m)

TL_NODES = {
    "J0": ( 0,   220),
    "J1": ( 0,     0),
    "J2": (220,  220),
    "J3": (220,    0),
    "J4": (440,  220),
}

FRINGE_NODES = {
    "J5":  (   0,  380),   # J0 kuzey
    "J6":  (-160,  220),   # J0 batı
    "J7":  (   0, -160),   # J1 güney
    "J8":  (-160,    0),   # J1 batı
    "J9":  ( 220,  380),   # J2 kuzey
    "J10": ( 380,    0),   # J3 doğu
    "J11": ( 220, -160),   # J3 güney
    "J12": ( 440,  380),   # J4 kuzey
    "J13": ( 440,   60),   # J4 güney
    "J14": ( 600,  220),   # J4 doğu
}

SPEED_MS = 13.89   # 50 km/h → m/s
NUM_LANES = 2

# (id, from, to) — hem ileri hem geri yönler
EDGES = [
    # İç yollar
    ("E0",   "J0",  "J1"),
    ("-E0",  "J1",  "J0"),
    ("E1",   "J0",  "J2"),
    ("-E1",  "J2",  "J0"),
    ("E2",   "J2",  "J3"),
    ("-E2",  "J3",  "J2"),
    ("E3",   "J1",  "J3"),
    ("-E3",  "J3",  "J1"),
    ("E4",   "J2",  "J4"),
    ("-E4",  "J4",  "J2"),
    # Fringe yollar
    ("E5",   "J0",  "J5"),
    ("-E5",  "J5",  "J0"),
    ("E6",   "J0",  "J6"),
    ("-E6",  "J6",  "J0"),
    ("E7",   "J1",  "J7"),
    ("-E7",  "J7",  "J1"),
    ("E8",   "J1",  "J8"),
    ("-E8",  "J8",  "J1"),
    ("E9",   "J2",  "J9"),
    ("-E9",  "J9",  "J2"),
    ("E10",  "J3",  "J10"),
    ("-E10", "J10", "J3"),
    ("E11",  "J3",  "J11"),
    ("-E11", "J11", "J3"),
    ("E12",  "J4",  "J12"),
    ("-E12", "J12", "J4"),
    ("E13",  "J4",  "J13"),
    ("-E13", "J13", "J4"),
    ("E14",  "J4",  "J14"),
    ("-E14", "J14", "J4"),
]


# ── 8-fazlı TL programı ──────────────────────────────────────────────────────
#
# Netconvert 4 faz üretiyor (N+S birleşik, E+W birleşik).
# Model 8 faz bekliyor: N / E / S / W her biri ayrı yeşil + sarı.
# State string 16 karakter — her kavşakta 16 kontrollü bağlantı:
#   0-3  → N yönü,  4-7  → E yönü,  8-11 → S yönü,  12-15 → W yönü
#
_8_PHASE_PROGRAM = [
    ("40", "GGGgrrrrrrrrrrrr"),  # faz 0: N-yeşil
    ("3",  "yyyyrrrrrrrrrrrr"),  # faz 1: N-sarı
    ("40", "rrrrGGGgrrrrrrrr"),  # faz 2: E-yeşil
    ("3",  "rrrryyyyrrrrrrrr"),  # faz 3: E-sarı
    ("40", "rrrrrrrrGGGgrrrr"),  # faz 4: S-yeşil
    ("3",  "rrrrrrrryyyyrrrr"),  # faz 5: S-sarı
    ("40", "rrrrrrrrrrrrGGGg"),  # faz 6: W-yeşil
    ("3",  "rrrrrrrrrrrryyyy"),  # faz 7: W-sarı
]


def fix_tl_programs(net_path: Path):
    """
    Netconvert'in ürettiği 4-fazlı TL programlarını 8-fazlıya dönüştürür.

    SUMO SAX parser event-driven çalışır: <tlLogic> mutlaka <junction> ve
    <connection> elementlerinden ÖNCE gelmelidir, yoksa "tls not known" hatası.
    """
    import re
    text = net_path.read_text(encoding="utf-8")

    # 1. Mevcut tüm <tlLogic> bloklarını kaldır (4-faz veya önceki yamalar)
    text = re.sub(
        r'\s*<tlLogic\b[^>]*>.*?</tlLogic>',
        '',
        text,
        flags=re.DOTALL,
    )

    # 2. Junction'larda duplicate tl attribute varsa temizle, sonra tek ekle
    for jid in TL_NODES:
        # Önce varsa kaldır
        text = re.sub(
            rf'(<junction id="{jid}" type="traffic_light")(\s+tl="[^"]*")+',
            rf'\1',
            text,
        )
        # Sonra tek seferde ekle
        text = re.sub(
            rf'(<junction id="{jid}" type="traffic_light")',
            rf'\1 tl="{jid}"',
            text,
        )

    # 3. 8-fazlı blokları oluştur
    phase_lines = "\n".join(
        f'        <phase duration="{dur}" state="{state}"/>'
        for dur, state in _8_PHASE_PROGRAM
    )
    new_blocks = "\n".join(
        f'    <tlLogic id="{jid}" type="static" programID="0" offset="0">\n'
        f'{phase_lines}\n'
        f'    </tlLogic>'
        for jid in TL_NODES
    )

    # 4. <tlLogic> bloklarını ilk <junction> elementinden ÖNCE ekle
    #    (SUMO SAX parser sıraya göre işler: tlLogic → junction → connection)
    first_junction = re.search(r'<junction\b', text)
    if first_junction:
        pos = first_junction.start()
        text = text[:pos] + new_blocks + "\n\n    " + text[pos:]
    else:
        # Fallback: </net> öncesi
        text = text.replace("</net>", f"{new_blocks}\n</net>")

    net_path.write_text(text, encoding="utf-8")
    print(f"  [OK] 8-fazlı TL programları <junction> öncesine yazıldı ({len(TL_NODES)} kavşak)")


# ── XML yazıcılar ─────────────────────────────────────────────────────────────

def write_nodes(path: Path):
    lines = ['<nodes>']
    for nid, (x, y) in TL_NODES.items():
        lines.append(f'    <node id="{nid}" x="{x}" y="{y}" type="traffic_light"/>')
    for nid, (x, y) in FRINGE_NODES.items():
        lines.append(f'    <node id="{nid}" x="{x}" y="{y}" type="dead_end"/>')
    lines.append('</nodes>')
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] {path.name} yazıldı")


def write_edges(path: Path):
    lines = ['<edges>']
    for eid, frm, to in EDGES:
        lines.append(
            f'    <edge id="{eid}" from="{frm}" to="{to}"'
            f' numLanes="{NUM_LANES}" speed="{SPEED_MS:.4f}" priority="1"/>'
        )
    lines.append('</edges>')
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] {path.name} yazıldı")


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def main():
    netconvert = find_netconvert()
    print(f"  netconvert: {netconvert}")

    nod_file  = HERE / "_tmp_nodes.nod.xml"
    edg_file  = HERE / "_tmp_edges.edg.xml"
    out_file  = HERE / "map.net.xml"
    bak_file  = HERE / "map.net.xml.bak"

    write_nodes(nod_file)
    write_edges(edg_file)

    # Mevcut ağı yedekle
    if out_file.exists():
        shutil.copy(out_file, bak_file)
        print(f"  [OK] Yedek: {bak_file.name}")

    cmd = [
        netconvert,
        "--node-files",       str(nod_file),
        "--edge-files",       str(edg_file),
        "--output-file",      str(out_file),
        "--tls.guess",        "true",
        "--tls.cycle.time",   "166",       # 4×(40s yeşil+3s sarı) = 172 → netconvert 166 kullanıyor
        "--no-turnarounds",   "true",
        "--junctions.corner-detail", "5",
        "--no-warnings",      "true",
        "--log",              str(HERE / "_netconvert.log"),
    ]

    print(f"\n  netconvert çalıştırılıyor...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("HATA:")
        print(result.stdout[-2000:] if result.stdout else "")
        print(result.stderr[-2000:] if result.stderr else "")
        sys.exit(1)

    # Geçici dosyaları temizle
    nod_file.unlink(missing_ok=True)
    edg_file.unlink(missing_ok=True)

    # Netconvert'in 4-fazlı programını 8-fazlıya yükselt
    fix_tl_programs(out_file)

    # Yeni yol uzunluklarını raporla
    print(f"\n  [OK] {out_file.name} güncellendi. Yeni uzunluklar:")
    _report_lengths(out_file)


def _report_lengths(net_path: Path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(net_path)
    root = tree.getroot()
    internal, fringe = [], []
    for edge in root.findall("edge"):
        eid = edge.get("id", "")
        if eid.startswith(":"): continue
        lane = edge.find("lane")
        if lane is None: continue
        length = float(lane.get("length", 0))
        # Fringe: E5-E14 ve ters yönleri
        base = eid.lstrip("-")
        if base in {f"E{i}" for i in range(5, 15)}:
            fringe.append(length)
        else:
            internal.append(length)
    if internal:
        print(f"    İç yollar   : min={min(internal):.1f}m  max={max(internal):.1f}m  "
              f"ort={sum(internal)/len(internal):.1f}m")
    if fringe:
        print(f"    Fringe yollar: min={min(fringe):.1f}m  max={max(fringe):.1f}m  "
              f"ort={sum(fringe)/len(fringe):.1f}m")


if __name__ == "__main__":
    print("=" * 55)
    print("  TraFix — Ağ Yeniden Oluşturucu")
    print("=" * 55)
    main()
    print("\n  Tamamlandı. Talebi yeniden oluşturmayı unutma:")
    print("  python sumo/generate_demand.py")
