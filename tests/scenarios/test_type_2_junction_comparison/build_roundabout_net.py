"""
Build the Turkish-style 5-roundabout SUMO network for Test Type 2 Junction A.

Physical layout (per junction):
  • 6-lane roads  — each direction has a 3-lane entry edge and a 3-lane exit edge
    (separate parallel edges in SUMO, together forming the 6-lane road).
  • Ring road     — 2-lane CCW circulating ring, radius 40 m from junction centre.
  • 4 signal heads INSIDE the roundabout — one at each cardinal entry node
    (J_RN, J_RE, J_RS, J_RW).  From the approach side these are "just before
    entering"; from above they sit on the ring road inside the roundabout circle.

TLS conflict logic at each ring entry node:
  The signal controls two conflicting movements at every entry point:
    APPROACH  — vehicle coming from outside wants to enter the ring
    CIRCULATE — vehicle already in the ring passes through (potential collision)
  → When APPROACH is GREEN, CIRCULATE at that node is RED (and vice versa).

Webster's cycle (tuned at medium load, 500 veh/hr, held constant):
  Phase 0 — N+S approaches GREEN, ring E-W arcs protected   (40 s)
  Phase 1 — N+S yellow                                       (3 s)
  Phase 2 — E+W approaches GREEN, ring N-S arcs protected   (40 s)
  Phase 3 — E+W yellow                                       (3 s)
  Total cycle = 86 s

Run once from the project root:
    python tests/scenarios/test_type_2_junction_comparison/build_roundabout_net.py

Outputs:
    junction_a_roundabout/network.net.xml
    junction_a_roundabout/tls_fixed.add.xml
    junction_b_standard/network.net.xml  (symlink → sumo/map.net.xml)
"""

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_HERE    = Path(__file__).resolve().parent
_OUT_A   = _HERE / "junction_a_roundabout"
_OUT_B   = _HERE / "junction_b_standard"
_TMP     = _HERE / "_tmp_netconvert"
_OUT_NET = _OUT_A / "network.net.xml"
_OUT_TLS = _OUT_A / "tls_fixed.add.xml"

for d in [_OUT_A, _OUT_B, _TMP]:
    d.mkdir(parents=True, exist_ok=True)

# Ring radius from junction centre (metres).
# 40 m → circumference ~251 m, arc between adjacent entry nodes ~63 m.
# Gives a large, clearly visible roundabout with ample merging space for
# 3-lane approaches feeding a 2-lane ring.
R = 40.0

# Number of lanes on the circulating ring road.
# 3 lanes matches the 3-lane approaches 1:1, so approach→ring, ring→ring and
# ring→exit are all strict same-lane connections with NO multi-lane merge. A
# merge means two links target one lane, which SUMO flags "unsafe green" and
# which causes the entering/circulating streams to collide. 1:1 avoids that
# entirely → collision-free, no teleporting.
RING_LANES = 3

# Junction centres from sumo/map.net.xml
JUNCTIONS = {
    "J0": (160.0, 380.0),
    "J1": (160.0, 160.0),
    "J2": (380.0, 380.0),
    "J3": (380.0, 160.0),
    "J4": (600.0, 380.0),
}

FRINGE_NODES = {
    "J5":  (160.0, 540.0), "J6":  (0.0,   380.0),
    "J7":  (160.0, 0.0),   "J8":  (0.0,   160.0),
    "J9":  (380.0, 540.0), "J10": (540.0, 160.0),
    "J11": (380.0, 0.0),   "J12": (600.0, 540.0),
    "J13": (600.0, 220.0), "J14": (760.0, 380.0),
}

# (edge_id, from_node, to_node, numLanes, speed_ms)
# from/to are ring-node IDs for edges connecting to TLS junctions
APPROACH_EDGES = [
    # Inter-junction bidirectional pairs
    ("E0",  "J0_RS", "J1_RN", 3, 13.89),
    ("-E0", "J1_RN", "J0_RS", 3, 13.89),
    ("E1",  "J0_RE", "J2_RW", 3, 13.89),
    ("-E1", "J2_RW", "J0_RE", 3, 13.89),
    ("E2",  "J2_RS", "J3_RN", 3, 13.89),
    ("-E2", "J3_RN", "J2_RS", 3, 13.89),
    ("E3",  "J1_RE", "J3_RW", 3, 13.89),
    ("-E3", "J3_RW", "J1_RE", 3, 13.89),
    ("E4",  "J2_RE", "J4_RW", 3, 13.89),
    ("-E4", "J4_RW", "J2_RE", 3, 13.89),
    # Fringe edges (same speed)
    ("E5",   "J0_RN", "J5",   3, 13.89), ("-E5",  "J5",    "J0_RN", 3, 13.89),
    ("E6",   "J0_RW", "J6",   3, 13.89), ("-E6",  "J6",    "J0_RW", 3, 13.89),
    ("E7",   "J1_RS", "J7",   3, 13.89), ("-E7",  "J7",    "J1_RS", 3, 13.89),
    ("E8",   "J1_RW", "J8",   3, 13.89), ("-E8",  "J8",    "J1_RW", 3, 13.89),
    ("E9",   "J2_RN", "J9",   3, 13.89), ("-E9",  "J9",    "J2_RN", 3, 13.89),
    ("E10",  "J3_RE", "J10",  3, 13.89), ("-E10", "J10",   "J3_RE", 3, 13.89),
    ("E11",  "J3_RS", "J11",  3, 13.89), ("-E11", "J11",   "J3_RS", 3, 13.89),
    ("E12",  "J4_RN", "J12",  3, 13.89), ("-E12", "J12",   "J4_RN", 3, 13.89),
    ("E13",  "J4_RS", "J13",  3, 13.89), ("-E13", "J13",   "J4_RS", 3, 13.89),
    ("E14",  "J4_RE", "J14",  3, 13.89), ("-E14", "J14",   "J4_RE", 3, 13.89),
]

# For each junction, which approach edge arrives at which ring node
# approach_edge → ring_entry_node_suffix
APPROACH_MAP = {
    "J0": {"-E5": "RN", "-E1": "RE", "-E0": "RS", "-E6": "RW",
           "E0":  "RS",  "E1": "RE",  "E5": "RN",  "E6": "RW"},
    "J1": {"E0":  "RN", "-E3": "RE", "-E7": "RS", "-E8": "RW",
           "-E0": "RN",  "E3": "RE",  "E7": "RS",  "E8": "RW"},
    "J2": {"-E9": "RN", "-E4": "RE", "-E2": "RS",  "E1": "RW",
           "E9":  "RN",  "E4": "RE",  "E2": "RS", "-E1": "RW"},
    "J3": {"E2":  "RN", "-E10":"RE", "-E11":"RS",  "E3": "RW",
           "-E2": "RN", "E10": "RE", "E11": "RS", "-E3": "RW"},
    "J4": {"E12": "RN", "-E14":"RE", "-E13":"RS",  "E4": "RW",
           "-E12":"RN", "E14": "RE", "E13": "RS", "-E4": "RW"},
}


def _ring_nodes(jid, cx, cy):
    return {
        "RN": (cx,     cy + R),
        "RE": (cx + R, cy),
        "RS": (cx,     cy - R),
        "RW": (cx - R, cy),
    }


def _ring_edge_defs(jid):
    """CCW ring: RN→RW→RS→RE→RN."""
    return [
        (f"rnd_{jid}_NW", f"{jid}_RN", f"{jid}_RW"),
        (f"rnd_{jid}_WS", f"{jid}_RW", f"{jid}_RS"),
        (f"rnd_{jid}_SE", f"{jid}_RS", f"{jid}_RE"),
        (f"rnd_{jid}_EN", f"{jid}_RE", f"{jid}_RN"),
    ]


# ── nod.xml ───────────────────────────────────────────────────────────────────

def write_nod():
    root = ET.Element("nodes")
    for nid, (x, y) in FRINGE_NODES.items():
        ET.SubElement(root, "node", id=nid, x=str(x), y=str(y), type="dead_end")
    for jid, (cx, cy) in JUNCTIONS.items():
        tl = f"{jid}_round"
        for suf, (x, y) in _ring_nodes(jid, cx, cy).items():
            ET.SubElement(root, "node",
                          id=f"{jid}_{suf}", x=str(x), y=str(y),
                          type="traffic_light", tl=tl)
    p = _TMP / "roundabout.nod.xml"
    ET.ElementTree(root).write(str(p), xml_declaration=True, encoding="UTF-8")
    return p


# ── edg.xml ───────────────────────────────────────────────────────────────────

def write_edg():
    root = ET.Element("edges")
    for eid, frm, to, lanes, speed in APPROACH_EDGES:
        ET.SubElement(root, "edge", id=eid, **{"from": frm}, to=to,
                      numLanes=str(lanes), speed=str(speed),
                      priority="1", spreadType="right")
    for jid in JUNCTIONS:
        for eid, frm, to in _ring_edge_defs(jid):
            ET.SubElement(root, "edge", id=eid, **{"from": frm}, to=to,
                          numLanes=str(RING_LANES), speed="8.33",
                          priority="2", spreadType="center")
    p = _TMP / "roundabout.edg.xml"
    ET.ElementTree(root).write(str(p), xml_declaration=True, encoding="UTF-8")
    return p


# ── con.xml ───────────────────────────────────────────────────────────────────

def write_con():
    """
    Connections for a 3-lane approach → RING_LANES-lane ring → 3-lane exit layout.

    Approach (3 lanes) → ring:
      Lane 0 (rightmost) → ring lane 1 (outer — likely to exit at the next node)
      Lane 1 (middle)    → ring lane 0 and lane 1
      Lane 2 (leftmost)  → ring lane 0 (inner — going further around)

    Ring pass-through (RING_LANES lanes):
      Each ring lane continues into the same lane on the next ring edge.

    Ring exit → exit edge (3 lanes):
      Ring lane 0 (inner)  → exit lanes 0 and 1
      Ring lane 1 (outer)  → exit lanes 1 and 2
    """
    root = ET.Element("connections")

    for jid in JUNCTIONS:
        ring_in = {
            "RW": f"rnd_{jid}_NW",
            "RS": f"rnd_{jid}_WS",
            "RE": f"rnd_{jid}_SE",
            "RN": f"rnd_{jid}_EN",
        }
        ring_out = {
            "RN": f"rnd_{jid}_NW",
            "RW": f"rnd_{jid}_WS",
            "RS": f"rnd_{jid}_SE",
            "RE": f"rnd_{jid}_EN",
        }
        exit_edge = {
            "RN": f"E5"  if jid=="J0" else f"E12" if jid=="J4" else
                  f"-E0" if jid=="J1" else f"E9"  if jid=="J2" else f"-E2",
            "RE": f"E1"  if jid=="J0" else f"E14" if jid=="J4" else
                  f"E3"  if jid=="J1" else f"E4"  if jid=="J2" else f"E10",
            "RS": f"E0"  if jid=="J0" else f"E13" if jid=="J4" else
                  f"E7"  if jid=="J1" else f"E2"  if jid=="J2" else f"E11",
            "RW": f"E6"  if jid=="J0" else f"-E4" if jid=="J4" else
                  f"E8"  if jid=="J1" else f"-E1" if jid=="J2" else f"-E3",
        }

        for suf in ["RN", "RE", "RS", "RW"]:
            ring_next = ring_out[suf]
            r_in      = ring_in[suf]
            r_out     = ring_out[suf]

            # Strict 1:1 lane mapping everywhere (RING_LANES == approach lanes).
            # No two links ever target the same lane → no "unsafe green" merge,
            # no merge collisions, no teleporting. A car keeps its lane index all
            # the way around the ring and can exit at any node in that same lane.

            # 1. Approach lane i → ring lane i
            app_edge = _approach_edge_for(jid, suf)
            if app_edge:
                app_lanes = _edge_lanes(app_edge)
                for ln in range(min(app_lanes, RING_LANES)):
                    ET.SubElement(root, "connection",
                                  **{"from": app_edge, "to": ring_next},
                                  fromLane=str(ln), toLane=str(ln))

            # 2. Ring pass-through: lane i → lane i
            for rl in range(RING_LANES):
                ET.SubElement(root, "connection",
                              **{"from": r_in, "to": r_out},
                              fromLane=str(rl), toLane=str(rl))

            # 3. Ring exit: ring lane i → exit lane i
            ex = exit_edge.get(suf, "")
            if ex:
                ex_lanes = _edge_lanes(ex)
                for ln in range(min(RING_LANES, ex_lanes)):
                    ET.SubElement(root, "connection",
                                  **{"from": r_in, "to": ex},
                                  fromLane=str(ln), toLane=str(ln))

    p = _TMP / "roundabout.con.xml"
    ET.ElementTree(root).write(str(p), xml_declaration=True, encoding="UTF-8")
    return p


def _approach_edge_for(jid, suffix):
    """Return the external approach edge that enters jid's ring at `suffix`."""
    amap = APPROACH_MAP[jid]
    for edge, suf in amap.items():
        if suf == suffix and _is_approach(edge):
            return edge
    return None


def _is_approach(edge_id: str) -> bool:
    """True if this edge flows INTO the junction (negative edge or fringe edge)."""
    # Edges that are approach edges have names starting with '-' OR
    # are fringe edges like E0 whose 'to' is a junction ring node
    # Simpler: an edge is an 'approach' if its ID makes it flow INTO the ring node
    # (we compare against the direction in APPROACH_EDGES)
    for eid, frm, to, *_ in APPROACH_EDGES:
        if eid == edge_id and to.endswith(("_RN","_RE","_RS","_RW")):
            return True
    return False


def _edge_lanes(edge_id: str) -> int:
    for eid, frm, to, lanes, speed in APPROACH_EDGES:
        if eid == edge_id:
            return lanes
    return 1


# ── netconvert ────────────────────────────────────────────────────────────────

def run_netconvert(nod, edg, con):
    cmd = [
        "netconvert",
        "--node-files",       str(nod),
        "--edge-files",       str(edg),
        "--connection-files", str(con),
        "--output-file",      str(_OUT_NET),
        "--no-turnarounds",   "true",
        "--junctions.corner-detail", "5",
        "--no-warnings",      "true",
        "--tls.guess",        "true",
        "--tls.cycle.time",   "86",
    ]
    print(f"  netconvert...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("STDERR:", res.stderr[:3000])
        sys.exit(f"netconvert failed (code {res.returncode})")
    print(f"  Network: {_OUT_NET}")


# ── tls_fixed.add.xml ─────────────────────────────────────────────────────────

def _classify_links(net_root, tl_id):
    """
    For one TLS controller, return a list of (link_index, movement_type) where
    movement_type is one of:
      'NS_approach'  — vehicle from N or S approach entering the ring
      'EW_approach'  — vehicle from E or W approach entering the ring
      'NS_ring'      — ring circulation through the N or S entry node
      'EW_ring'      — ring circulation through the E or W entry node
      'exit'         — vehicle leaving the ring to an exit edge
    """
    # Collect all connections controlled by this TLS from the net.xml
    link_types = []
    for conn in net_root.iter("connection"):
        if conn.get("tl") != tl_id:
            continue
        idx   = int(conn.get("linkIndex", -1))
        frm   = conn.get("from", "")
        to    = conn.get("to", "")
        via   = conn.get("via", "")

        # Approach: incoming edge is an external approach (not a ring edge)
        is_ring_from = frm.startswith("rnd_")
        is_ring_to   = to.startswith("rnd_")

        if not is_ring_from and is_ring_to:
            # Approach → ring  (entering). Classify by the ENTRY NODE, i.e. the
            # ring arc the approach merges ONTO (whose source node is the entry):
            #   NW leaves RN, SE leaves RS  → North/South entries
            #   EN leaves RE, WS leaves RW  → East/West entries
            if any(s in to for s in ("_NW", "_SE")):
                mtype = "NS_approach"
            elif any(s in to for s in ("_EN", "_WS")):
                mtype = "EW_approach"
            else:
                mtype = "NS_approach"
        elif is_ring_from and is_ring_to:
            # Ring → ring (circulating pass-through). Classify by the arc that
            # FEEDS the node, so the arc that physically conflicts with an entry
            # is the one held RED while that entry is green:
            #   EN feeds RN, WS feeds RS  → conflicts with N/S entries ("NS_ring")
            #   NW feeds RW, SE feeds RE  → conflicts with E/W entries ("EW_ring")
            if any(s in frm for s in ("_EN", "_WS")):
                mtype = "NS_ring"
            else:
                mtype = "EW_ring"
        elif is_ring_from and not is_ring_to:
            mtype = "exit"
        else:
            mtype = "exit"

        link_types.append((idx, mtype))

    link_types.sort(key=lambda x: x[0])
    return link_types


def _build_state(link_types, phase: str) -> str:
    """
    Build a TLS state string for one phase.
    phase: 'NS_green' | 'NS_yellow' | 'EW_green' | 'EW_yellow'

    Conflict rule:
      When NS approaches are GREEN  → NS ring (circulating E→W arcs) must be RED
      When EW approaches are GREEN  → EW ring (circulating N→S arcs) must be RED
      Exits are always GREEN (vehicles leaving the ring are never blocked).
    """
    if not link_types:
        return ""
    n = max(idx for idx, _ in link_types) + 1
    state = ["r"] * n

    # Movements that physically MERGE onto a shared lane (free circulating arcs and
    # ring→exit fans) use permissive green 'g' instead of protected 'G', so SUMO
    # makes the later vehicle give way (zipper/car-following) rather than letting
    # two 'G' streams claim the same lane and collide. Only the single protected
    # approach stream keeps 'G'.
    for idx, mtype in link_types:
        if phase == "NS_green":
            if mtype == "NS_approach":   state[idx] = "G"   # protected
            elif mtype == "EW_approach": state[idx] = "r"
            elif mtype == "NS_ring":     state[idx] = "r"   # blocked — conflict
            elif mtype == "EW_ring":     state[idx] = "g"   # free, give-way merge
            else:                        state[idx] = "g"   # exits, give-way merge

        elif phase == "NS_yellow":
            if mtype == "NS_approach":   state[idx] = "y"
            elif mtype == "EW_approach": state[idx] = "r"
            elif mtype == "NS_ring":     state[idx] = "y"
            elif mtype == "EW_ring":     state[idx] = "g"
            else:                        state[idx] = "g"

        elif phase == "EW_green":
            if mtype == "NS_approach":   state[idx] = "r"
            elif mtype == "EW_approach": state[idx] = "G"   # protected
            elif mtype == "NS_ring":     state[idx] = "g"   # free, give-way merge
            elif mtype == "EW_ring":     state[idx] = "r"   # blocked — conflict
            else:                        state[idx] = "g"

        elif phase == "EW_yellow":
            if mtype == "NS_approach":   state[idx] = "r"
            elif mtype == "EW_approach": state[idx] = "y"
            elif mtype == "NS_ring":     state[idx] = "g"
            elif mtype == "EW_ring":     state[idx] = "y"
            else:                        state[idx] = "g"

    return "".join(state)


def write_tls_add():
    """
    Write tls_fixed.add.xml with hand-crafted TLS phases implementing the
    correct conflict logic:

      Phase 0  N+S approaches GREEN  — circulating E-W arc segments blocked  (40 s)
      Phase 1  N+S yellow                                                      (3 s)
      Phase 2  E+W approaches GREEN  — circulating N-S arc segments blocked  (40 s)
      Phase 3  E+W yellow                                                      (3 s)

    Webster cycle = 86 s, tuned at 500 veh/hr, held constant across levels.
    """
    tree     = ET.parse(str(_OUT_NET))
    net_root = tree.getroot()

    add_root = ET.Element("additional")
    tls_seen = set()

    for tl_elem in net_root.iter("tlLogic"):
        tl_id = tl_elem.get("id", "")
        if not tl_id.endswith("_round") or tl_id in tls_seen:
            continue
        tls_seen.add(tl_id)

        link_types = _classify_links(net_root, tl_id)

        new_tl = ET.SubElement(add_root, "tlLogic",
                               id=tl_id, type="static",
                               programID="webster", offset="0")
        for dur, phase_key in [(40, "NS_green"), (3, "NS_yellow"),
                                (40, "EW_green"), (3, "EW_yellow")]:
            state = _build_state(link_types, phase_key)
            if state:
                ET.SubElement(new_tl, "phase",
                              duration=str(dur), state=state)

    if not list(add_root):
        sys.exit("ERROR: No roundabout TLS found in generated net.xml")

    ET.ElementTree(add_root).write(str(_OUT_TLS),
                                   xml_declaration=True, encoding="UTF-8")
    print(f"  TLS additional: {_OUT_TLS}")


# ── configs and symlink ───────────────────────────────────────────────────────

def write_configs():
    (_OUT_A / "config.sumocfg").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<configuration>\n'
        '    <input>\n'
        '        <net-file value="network.net.xml"/>\n'
        '        <additional-files value="tls_fixed.add.xml"/>\n'
        '    </input>\n'
        '    <processing>\n'
        '        <time-to-teleport value="-1"/>\n'
        '        <time-to-teleport.highways value="-1"/>\n'
        '        <collision.action value="warn"/>\n'
        '    </processing>\n'
        '</configuration>\n', encoding="utf-8")

    net_b = _OUT_B / "network.net.xml"
    project_root = _HERE.parents[2]
    target = project_root / "sumo" / "map.net.xml"
    if net_b.is_symlink() or net_b.exists():
        net_b.unlink(missing_ok=True)
    net_b.symlink_to(target)

    (_OUT_B / "config.sumocfg").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<configuration>\n'
        '    <input>\n'
        '        <net-file value="network.net.xml"/>\n'
        '    </input>\n'
        '    <processing>\n'
        '        <time-to-teleport value="-1"/>\n'
        '        <time-to-teleport.highways value="-1"/>\n'
        '        <collision.action value="warn"/>\n'
        '    </processing>\n'
        '</configuration>\n', encoding="utf-8")

    print(f"  Configs written.")


# ── main ──────────────────────────────────────────────────────────────────────

def write_route_symlinks():
    """Create symlinks to the shared type1 route files."""
    scenarios_dir = _HERE.parent  # tests/scenarios/
    for level in ("low", "medium", "high"):
        target = scenarios_dir / f"type1_{level}.rou.xml"
        link   = _HERE / f"{level}.rou.xml"
        if link.is_symlink() or link.exists():
            link.unlink(missing_ok=True)
        link.symlink_to(f"../type1_{level}.rou.xml")
        if not link.exists():
            sys.exit(f"ERROR: symlink {link} → target not found at {target}")
    print(f"  Route symlinks: low / medium / high → type1_*.rou.xml")


def main():
    print("=== Building Turkish-style roundabout network (Junction A) ===")
    nod = write_nod()
    edg = write_edg()
    con = write_con()
    run_netconvert(nod, edg, con)
    write_tls_add()
    write_configs()
    write_route_symlinks()
    print("\nDone. Roundabout network ready.")


if __name__ == "__main__":
    main()
