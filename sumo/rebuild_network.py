"""
TraFix — Network Rebuilder v6
==============================
Builds 5-junction topology with 3 lanes per road and 12-phase TL programs.

Changes from v5:
  - NUM_LANES = 3 (was 2)
  - TL programs: 12 phases (6 green + 6 yellow transitions)
  - State strings derived programmatically from SUMO getControlledLinks query

Lane indexing (SUMO right-to-left):
  Lane 0: rightmost = right turn (permissive)
  Lane 1: middle    = through (protected)
  Lane 2: leftmost  = left turn (protected)

12 SUMO phases:
  0/1   NS-through  green/yellow  (40s/3s)
  2/3   N-left      green/yellow  (20s/3s)
  4/5   S-left      green/yellow  (20s/3s)
  6/7   EW-through  green/yellow  (40s/3s)
  8/9   E-left      green/yellow  (20s/3s)
  10/11 W-left      green/yellow  (20s/3s)

Run:
  python sumo/rebuild_network.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = HERE.parent


# ── netconvert binary ─────────────────────────────────────────────────────────

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
    sys.exit("HATA: netconvert bulunamadi. SUMO_HOME ortam degiskenini ayarlayin.")


def _find_sumo_bin() -> str:
    sumo_home = os.environ.get("SUMO_HOME", "")
    candidates = []
    if sumo_home:
        candidates += [
            Path(sumo_home) / "bin" / "sumo.exe",
            Path(sumo_home) / "bin" / "sumo",
        ]
    candidates += [
        Path("C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe"),
        Path("C:/Program Files/Eclipse/Sumo/bin/sumo.exe"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    found = shutil.which("sumo") or shutil.which("sumo.exe")
    return found or "sumo"


# ── Node coordinates ──────────────────────────────────────────────────────────
#
#   J0(0,220)  — J2(220,220) — J4(440,220)
#       |              |
#   J1(0,0)   — J3(220,0)
#
#   Fringe nodes 160m from TL junction (edge length ~150m)

TL_NODES = {
    "J0": ( 0,   220),
    "J1": ( 0,     0),
    "J2": (220,  220),
    "J3": (220,    0),
    "J4": (440,  220),
}

FRINGE_NODES = {
    "J5":  (   0,  380),   # J0 north
    "J6":  (-160,  220),   # J0 west
    "J7":  (   0, -160),   # J1 south
    "J8":  (-160,    0),   # J1 west
    "J9":  ( 220,  380),   # J2 north
    "J10": ( 380,    0),   # J3 east
    "J11": ( 220, -160),   # J3 south
    "J12": ( 440,  380),   # J4 north
    "J13": ( 440,   60),   # J4 south
    "J14": ( 600,  220),   # J4 east
}

SPEED_MS = 13.89   # 50 km/h
NUM_LANES = 3      # was 2 in v5

EDGES = [
    # Internal edges
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
    # Fringe edges
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


# ── Lane type by SUMO lane index (0=rightmost) ────────────────────────────────

_LANE_TYPE = {0: "right", 1: "through", 2: "left"}

# ── 12-phase definitions ──────────────────────────────────────────────────────
# Each entry: (duration_seconds, {(direction, lane_type): state_char})
# Default char for any unspecified (direction, lane_type) is 'r'.

_PHASE_DEFS = [
    # Phase 0: NS-through green (40s)
    (40, {("north", "through"): "G", ("south", "through"): "G",
          ("north", "right"):   "g", ("south", "right"):   "g"}),
    # Phase 1: NS-through yellow (3s)
    (3,  {("north", "through"): "y", ("south", "through"): "y",
          ("north", "right"):   "y", ("south", "right"):   "y"}),
    # Phase 2: N-left green (20s)
    (20, {("north", "left"):    "G", ("north", "right"):   "g"}),
    # Phase 3: N-left yellow (3s)
    (3,  {("north", "left"):    "y", ("north", "right"):   "y"}),
    # Phase 4: S-left green (20s)
    (20, {("south", "left"):    "G", ("south", "right"):   "g"}),
    # Phase 5: S-left yellow (3s)
    (3,  {("south", "left"):    "y", ("south", "right"):   "y"}),
    # Phase 6: EW-through green (40s)
    (40, {("east",  "through"): "G", ("west",  "through"): "G",
          ("east",  "right"):   "g", ("west",  "right"):   "g"}),
    # Phase 7: EW-through yellow (3s)
    (3,  {("east",  "through"): "y", ("west",  "through"): "y",
          ("east",  "right"):   "y", ("west",  "right"):   "y"}),
    # Phase 8: E-left green (20s)
    (20, {("east",  "left"):    "G", ("east",  "right"):   "g"}),
    # Phase 9: E-left yellow (3s)
    (3,  {("east",  "left"):    "y", ("east",  "right"):   "y"}),
    # Phase 10: W-left green (20s)
    (20, {("west",  "left"):    "G", ("west",  "right"):   "g"}),
    # Phase 11: W-left yellow (3s)
    (3,  {("west",  "left"):    "y", ("west",  "right"):   "y"}),
]


# ── SUMO query: derive per-junction link classifications ──────────────────────

def _ensure_traci_importable():
    sumo_home = os.environ.get("SUMO_HOME", "")
    tools_path = os.path.join(sumo_home, "tools") if sumo_home else ""
    if tools_path and tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    for candidate in [
        "C:\\Program Files (x86)\\Eclipse\\Sumo\\tools",
        "C:\\Program Files\\Eclipse\\Sumo\\tools",
    ]:
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)


def _classify_from_lane(from_lane: str, jx: float, jy: float, traci) -> tuple:
    """
    Returns (direction, lane_type) for one controlled link's from_lane.
    direction: north/south/east/west (from lane shape geometry)
    lane_type: right/through/left (from lane index 0/1/2)
    """
    edge_id = from_lane.rsplit("_", 1)[0]
    lane_idx = int(from_lane.rsplit("_", 1)[1])
    lane_type = _LANE_TYPE.get(lane_idx, "through")

    try:
        # Use lane 0 of the edge for direction (always exists)
        shape = traci.lane.getShape(f"{edge_id}_0")
        if shape:
            x0, y0 = shape[0]
            dx, dy = x0 - jx, y0 - jy
            if abs(dx) > abs(dy):
                direction = "west" if dx < 0 else "east"
            else:
                direction = "south" if dy < 0 else "north"
        else:
            direction = "north"
    except Exception:
        direction = "north"

    return direction, lane_type


def _query_link_classifications(net_path: Path) -> dict:
    """
    Temporarily starts SUMO with the generated network and queries
    getControlledLinks for each TL junction.

    Returns {jid: [(direction, lane_type), ...]} for all links in order.
    """
    _ensure_traci_importable()

    try:
        import traci
    except ImportError:
        print("  [WARN] TraCI not importable — cannot derive state strings from query.")
        return {}

    # Write a minimal sumocfg with no route file so SUMO loads the net only
    tmp_cfg = net_path.parent / "_tmp_query.sumocfg"
    tmp_cfg.write_text(
        f'<configuration>\n'
        f'  <input>\n'
        f'    <net-file value="{net_path.name}"/>\n'
        f'  </input>\n'
        f'</configuration>\n',
        encoding="utf-8",
    )

    sumo_bin = _find_sumo_bin()
    result = {}

    try:
        sumo_cmd = [sumo_bin, "-c", str(tmp_cfg), "--no-step-log", "--no-warnings",
                    "--end", "1"]
        traci.start(sumo_cmd)

        for tls_id in TL_NODES:
            jx, jy = traci.junction.getPosition(tls_id)
            links = traci.trafficlight.getControlledLinks(tls_id)
            classifications = []
            for link in links:
                if link:
                    from_lane = link[0][0]
                    cls = _classify_from_lane(from_lane, jx, jy, traci)
                else:
                    cls = ("north", "through")  # fallback for empty link slot
                classifications.append(cls)
            result[tls_id] = classifications
            print(f"  [INFO] {tls_id}: {len(classifications)} controlled links")

        traci.close()

    except Exception as e:
        print(f"  [WARN] SUMO link query failed: {e}")
        try:
            traci.close()
        except Exception:
            pass
    finally:
        tmp_cfg.unlink(missing_ok=True)

    return result


# ── State string builder ──────────────────────────────────────────────────────

def _build_state_strings(classifications: list) -> list:
    """
    Given ordered [(direction, lane_type), ...] for a junction's controlled links,
    returns [(duration_str, state_str), ...] for all 12 phases.
    """
    phases = []
    for duration, green_map in _PHASE_DEFS:
        state = "".join(
            green_map.get((direction, lane_type), "r")
            for direction, lane_type in classifications
        )
        phases.append((str(duration), state))
    return phases


# ── TL program injection ──────────────────────────────────────────────────────

def fix_tl_programs(net_path: Path):
    """
    Replaces auto-generated TL programs with 12-phase programs.
    State strings are derived from SUMO getControlledLinks query.

    SUMO SAX parser requires <tlLogic> blocks to appear BEFORE <junction>
    elements in the XML file.
    """
    import re

    print("\n  Deriving state strings from SUMO link query...")
    classifications = _query_link_classifications(net_path)

    if not classifications:
        print("  [WARN] Link query returned no results — network may not have loaded.")
        print("  [WARN] State strings cannot be derived. Check SUMO installation.")

    text = net_path.read_text(encoding="utf-8")

    # 1. Remove all existing tlLogic blocks (from netconvert or previous runs)
    text = re.sub(
        r'\s*<tlLogic\b[^>]*>.*?</tlLogic>',
        '',
        text,
        flags=re.DOTALL,
    )

    # 2. Fix junction tl attributes — remove duplicates then add clean one
    for jid in TL_NODES:
        text = re.sub(
            rf'(<junction id="{jid}" type="traffic_light")(\s+tl="[^"]*")+',
            rf'\1',
            text,
        )
        text = re.sub(
            rf'(<junction id="{jid}" type="traffic_light")',
            rf'\1 tl="{jid}"',
            text,
        )

    # 3. Build 12-phase tlLogic blocks for each junction
    blocks = []
    for jid in TL_NODES:
        if jid in classifications and classifications[jid]:
            phases = _build_state_strings(classifications[jid])
            n_links = len(classifications[jid])
        else:
            # Fallback: all-red (shouldn't happen if SUMO is installed correctly)
            phases = [(str(dur), "r" * 36) for dur, _ in _PHASE_DEFS]
            n_links = 36
            print(f"  [WARN] {jid}: using all-red fallback ({n_links} links assumed)")

        phase_lines = "\n".join(
            f'        <phase duration="{dur}" state="{state}"/>'
            for dur, state in phases
        )
        blocks.append(
            f'    <tlLogic id="{jid}" type="static" programID="0" offset="0">\n'
            f'{phase_lines}\n'
            f'    </tlLogic>'
        )

        # Print state strings for verification
        print(f"\n  {jid} ({n_links} links):")
        for (dur, state), (_, phase_def) in zip(phases, _PHASE_DEFS):
            phase_name = [
                "NS-through", "NS-yellow", "N-left", "N-left-y",
                "S-left", "S-left-y", "EW-through", "EW-yellow",
                "E-left", "E-left-y", "W-left", "W-left-y",
            ][len(blocks) * 0 + phases.index((dur, state))]
            print(f"    P{phases.index((dur, state)):2d} ({phase_name:12s}): {state}")

    new_blocks = "\n".join(blocks)

    # 4. Insert tlLogic blocks BEFORE the first <junction> element
    first_junction = re.search(r'<junction\b', text)
    if first_junction:
        pos = first_junction.start()
        text = text[:pos] + new_blocks + "\n\n    " + text[pos:]
    else:
        text = text.replace("</net>", f"{new_blocks}\n</net>")

    net_path.write_text(text, encoding="utf-8")
    print(f"\n  [OK] 12-phase TL programs written before <junction> ({len(TL_NODES)} junctions)")


# ── XML writers ───────────────────────────────────────────────────────────────

def write_nodes(path: Path):
    lines = ['<nodes>']
    for nid, (x, y) in TL_NODES.items():
        lines.append(f'    <node id="{nid}" x="{x}" y="{y}" type="traffic_light"/>')
    for nid, (x, y) in FRINGE_NODES.items():
        lines.append(f'    <node id="{nid}" x="{x}" y="{y}" type="dead_end"/>')
    lines.append('</nodes>')
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] {path.name} written")


def write_edges(path: Path):
    lines = ['<edges>']
    for eid, frm, to in EDGES:
        lines.append(
            f'    <edge id="{eid}" from="{frm}" to="{to}"'
            f' numLanes="{NUM_LANES}" speed="{SPEED_MS:.4f}" priority="1"/>'
        )
    lines.append('</edges>')
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] {path.name} written")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    netconvert = find_netconvert()
    print(f"  netconvert: {netconvert}")
    print(f"  NUM_LANES : {NUM_LANES}")

    nod_file = HERE / "_tmp_nodes.nod.xml"
    edg_file = HERE / "_tmp_edges.edg.xml"
    out_file = HERE / "map.net.xml"
    bak_file = HERE / "map.net.xml.bak"

    write_nodes(nod_file)
    write_edges(edg_file)

    if out_file.exists():
        shutil.copy(out_file, bak_file)
        print(f"  [OK] Backup: {bak_file.name}")

    # Cycle time: 40+3+20+3+20+3+40+3+20+3+20+3 = 178s
    cmd = [
        netconvert,
        "--node-files",       str(nod_file),
        "--edge-files",       str(edg_file),
        "--output-file",      str(out_file),
        "--tls.guess",        "true",
        "--tls.cycle.time",   "178",
        "--no-turnarounds",   "true",
        "--junctions.corner-detail", "5",
        "--no-warnings",      "true",
        "--log",              str(HERE / "_netconvert.log"),
    ]

    print(f"\n  Running netconvert...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("ERROR:")
        print(result.stdout[-2000:] if result.stdout else "")
        print(result.stderr[-2000:] if result.stderr else "")
        sys.exit(1)

    nod_file.unlink(missing_ok=True)
    edg_file.unlink(missing_ok=True)

    # Replace auto-generated TL programs with derived 12-phase programs
    fix_tl_programs(out_file)

    print(f"\n  [OK] {out_file.name} updated. Road lengths:")
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
        base = eid.lstrip("-")
        if base in {f"E{i}" for i in range(5, 15)}:
            fringe.append(length)
        else:
            internal.append(length)
    if internal:
        print(f"    Internal edges : min={min(internal):.1f}m  max={max(internal):.1f}m  "
              f"avg={sum(internal)/len(internal):.1f}m")
    if fringe:
        print(f"    Fringe edges   : min={min(fringe):.1f}m  max={max(fringe):.1f}m  "
              f"avg={sum(fringe)/len(fringe):.1f}m")


if __name__ == "__main__":
    print("=" * 55)
    print("  TraFix — Network Rebuilder v6")
    print("=" * 55)
    main()
    print("\n  Done. Regenerate traffic demand:")
    print("  python sumo/generate_demand.py")
