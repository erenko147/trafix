"""
TraFix Unity Bridge
===================
* sumo-gui → SUMO penceresi + Unity 3D AYNI ANDA acilir
* Herhangi bir .sumocfg ile calisir
* Init mesaji yeni baglanan her istemciye DIREKT gonderilir (kuyruk yok)
* Frame mesajlari broadcast queue ile tum istemcilere gider
* FastAPI'ye /telemetry_batch endpoint'i ile toplu veri gonderilir

Kullanim:
    python sumo/sumo_unity_bridge.py
    python sumo/sumo_unity_bridge.py C:/harita/baska.sumocfg
"""

import os, sys, json, time, queue, threading, asyncio, requests, warnings
import xml.etree.ElementTree as ET

SUMO_HOME = os.environ.get("SUMO_HOME")
if SUMO_HOME:
    sys.path.append(os.path.join(SUMO_HOME, "tools"))
else:
    sys.exit("HATA: SUMO_HOME ortam degiskeni tanimlanmamis!")

import traci
warnings.filterwarnings("ignore", category=UserWarning, module="traci")

try:
    import websockets
except ImportError:
    sys.exit("HATA: pip install websockets  komutuyla yukle.")

# ── Ayarlar ────────────────────────────────────────────────────────────────
WS_PORT     = 8765
FASTAPI_URL = "http://127.0.0.1:8001/telemetry_batch"
USE_GUI     = True       # True = sumo-gui acilir, False = sadece Unity
UNITY_FPS   = 20         # Unity'ye saniyede kac frame gonderilsin
API_EVERY   = 5          # Her kac adimda FastAPI'ye gonderilsin

SUMO_CFG = (sys.argv[1] if len(sys.argv) > 1
            else os.path.join(os.path.dirname(__file__), "demo.sumocfg"))

# ── Global durum ───────────────────────────────────────────────────────────
_init_payload: str        = None   # Harita JSON'u - SUMO basladiktan sonra set edilir
_frame_queue: queue.Queue = queue.Queue(maxsize=60)
_clients: set             = set()
_lock                     = threading.Lock()

# ── WebSocket sunucu ───────────────────────────────────────────────────────

async def _on_connect(ws):
    """Yeni Unity baglantisi - once init gonder, sonra frame listesine ekle."""
    addr = ws.remote_address
    print(f"[WS] Unity baglandi: {addr}")

    # Init hazir degils e bekle (maks 30 sn - SUMO baslamasi icin)
    for _ in range(300):
        if _init_payload is not None:
            break
        await asyncio.sleep(0.1)

    if _init_payload is not None:
        try:
            await ws.send(_init_payload)
            print(f"[WS] Init gonderildi -> {addr}")
        except Exception as e:
            print(f"[WS] Init gonderilemedi: {e}")
    else:
        print("[WS] UYARI: Init hazir degil, Unity haritasiz calisacak.")

    # Frame yayin listesine ekle
    with _lock:
        _clients.add(ws)

    try:
        async for _ in ws:   # Unity'den mesaj beklenmez, baglanti kopana kadar bekle
            pass
    except Exception:
        pass
    finally:
        with _lock:
            _clients.discard(ws)
        print(f"[WS] Baglanti kesildi: {addr}")


async def _broadcast_frames():
    """Frame kuyrugundan al, tum Unity istemcilerine gonder."""
    while True:
        try:
            frame = _frame_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.005)
            continue

        with _lock:
            targets = list(_clients)

        if targets:
            await asyncio.gather(
                *[c.send(frame) for c in targets],
                return_exceptions=True
            )


async def _ws_server():
    print(f"[WS] WebSocket sunucusu baslatildi: ws://localhost:{WS_PORT}")
    async with websockets.serve(_on_connect, "0.0.0.0", WS_PORT):
        await _broadcast_frames()   # sonsuza dek calis


def _ws_thread():
    asyncio.run(_ws_server())

# ── Harita ayristirma ──────────────────────────────────────────────────────

def _find_net(cfg: str) -> str:
    try:
        root = ET.parse(cfg).getroot()
        for tag in ("net-file", "net_file"):
            el = root.find(f".//{tag}")
            if el is not None:
                v = el.get("value", "")
                if v:
                    p = os.path.join(os.path.dirname(cfg), v)
                    if os.path.exists(p):
                        return p
    except Exception:
        pass
    d = os.path.dirname(cfg)
    for f in os.listdir(d):
        if f.endswith(".net.xml"):
            return os.path.join(d, f)
    return ""


def build_init(cfg: str) -> str:
    net = _find_net(cfg)
    if not net:
        print("[MAP] net.xml bulunamadi.")
        return json.dumps({"type":"init","bounds":[-50,-50,250,150],
                           "junctions":[],"edges":[]}, separators=(",",":"))

    print(f"[MAP] Okunuyor: {net}")
    root = ET.parse(net).getroot()

    # Sinirlar
    bounds = [-50, -50, 250, 150]
    loc = root.find("location")
    if loc is not None:
        cb = loc.get("convBoundary","")
        if cb:
            try: bounds = [float(v) for v in cb.split(",")]
            except: pass

    # Kavsak
    junctions = []
    for j in root.findall("junction"):
        t = j.get("type","")
        if t in ("internal","dead_end"): continue
        junctions.append({
            "id": j.get("id"),
            "x" : float(j.get("x",0)),
            "y" : float(j.get("y",0)),
            "tls": 1 if t=="traffic_light" else 0,
        })

    # Kenarlar
    edges = []
    for edge in root.findall("edge"):
        if edge.get("function")=="internal": continue
        eid = edge.get("id","")
        if eid.startswith(":"): continue
        lanes = edge.findall("lane")
        if not lanes: continue

        lw = 3.2
        shapes = []
        for ln in lanes:
            w = ln.get("width")
            if w:
                try: lw = float(w)
                except: pass
            sh = ln.get("shape","")
            pts = []
            for pt in sh.split():
                try:
                    a,b = pt.split(",")
                    pts.append((float(a),float(b)))
                except: pass
            if pts: shapes.append(pts)

        if not shapes: continue

        n = min(len(s) for s in shapes)
        flat = []
        for i in range(n):
            ax = sum(s[i][0] for s in shapes)/len(shapes)
            ay = sum(s[i][1] for s in shapes)/len(shapes)
            flat += [round(ax,2), round(ay,2)]

        edges.append({
            "id": eid,
            "ln": len(lanes),
            "w" : round(lw*len(lanes)+2.0, 1),
            "sh": flat,
        })

    print(f"[MAP] {len(junctions)} kavsak, {len(edges)} kenar.")
    return json.dumps({
        "type"      : "init",
        "bounds"    : bounds,
        "junctions" : junctions,
        "edges"     : edges,
    }, separators=(",",":"))

# ── SUMO yardimcilari ──────────────────────────────────────────────────────

def _dirs(tid):
    dm = {"N":None,"S":None,"E":None,"W":None}
    try:
        jx,jy = traci.junction.getPosition(tid)
        for grp in traci.trafficlight.getControlledLinks(tid):
            for lnk in grp:
                if not lnk: continue
                eid = lnk[0].split("_")[0]
                if eid.startswith(":"): continue
                try:
                    pts = traci.lane.getShape(f"{eid}_0")
                    if not pts: continue
                    dx,dy = pts[0][0]-jx, pts[0][1]-jy
                    if abs(dx)>abs(dy): dm["W" if dx<0 else "E"]=eid
                    else:               dm["S" if dy<0 else "N"]=eid
                except: pass
    except: pass
    return dm

def _vc(e):
    if not e: return 0
    try: return int(traci.edge.getLastStepVehicleNumber(e))
    except: return 0

def _wt(eds):
    t=0.0
    for e in eds:
        if e:
            try: t+=traci.edge.getWaitingTime(e)
            except: pass
    return t

def _enqueue(s):
    """Frame kuyruğuna gonder, doluysa eskiyi at."""
    if _frame_queue.full():
        try: _frame_queue.get_nowait()
        except: pass
    try: _frame_queue.put_nowait(s)
    except: pass

# ── Ana dongu ──────────────────────────────────────────────────────────────

def main():
    global _init_payload

    if not os.path.exists(SUMO_CFG):
        sys.exit(f"HATA: {SUMO_CFG} bulunamadi!")

    # 1) Harita JSON'unu hazirla (WS acilmadan once — Unity baglananlar bunu alir)
    _init_payload = build_init(SUMO_CFG)

    # 2) WebSocket sunucusunu arka planda baslat
    threading.Thread(target=_ws_thread, daemon=True).start()
    time.sleep(0.5)
    print("[WS] Sunucu hazir. Unity Play'e basin, otomatik baglanir.\n")

    # 3) SUMO'yu baslat
    exe = "sumo-gui" if USE_GUI else "sumo"
    print(f"[SUMO] {exe} baslatiliyor: {SUMO_CFG}")
    traci.start([exe, "-c", SUMO_CFG, "--no-warnings",
                 "--step-length", "0.1", "--collision.action", "warn"])

    # 4) Kavsak yon haritasi
    tls_ids    = sorted(traci.trafficlight.getIDList())
    inter_dirs = {t: _dirs(t) for t in tls_ids}
    print(f"[SUMO] Baslatildi. TLS: {tls_ids if tls_ids else '(yok)'}")

    step       = 0
    last_unity = time.time()
    step_dur   = 1.0 / UNITY_FPS
    empty_steps = 0           # getMinExpectedNumber()==0 olan ardisik adim sayisi
    MAX_EMPTY   = 500         # bu kadar bos adimdan sonra cik (50 sn @ 0.1s/step)

    try:
        while True:
            # Simulasyon bitis kontrolu: bos adim sayisi MAX_EMPTY'yi asarsa cik
            try:
                n_expected = traci.simulation.getMinExpectedNumber()
            except Exception as e:
                print(f"[SUMO] TraCI baglantisi kesildi: {e}")
                break

            if n_expected == 0:
                empty_steps += 1
                if empty_steps >= MAX_EMPTY:
                    print(f"[SUMO] Simulasyon tamamlandi. (adim={step})")
                    break
            else:
                empty_steps = 0

            try:
                traci.simulationStep()
            except Exception as e:
                print(f"[SUMO] simulationStep hatasi: {e}")
                break

            now = time.time()

            # ── Unity frame ──────────────────────────────────────────────
            if now - last_unity >= step_dur:
                last_unity = now

                vlist = []
                for vid in traci.vehicle.getIDList():
                    try:
                        x,y   = traci.vehicle.getPosition(vid)
                        angle = traci.vehicle.getAngle(vid)
                        speed = traci.vehicle.getSpeed(vid)
                        vcls  = traci.vehicle.getVehicleClass(vid)
                        vlist.append({
                            "i": vid,
                            "x": round(x,2), "y": round(y,2),
                            "a": round(angle,1),
                            "s": round(speed,2),
                            "c": vcls,
                        })
                    except: pass

                tls_list = []
                for tid in tls_ids:
                    try:
                        ph = int(traci.trafficlight.getPhase(tid))
                        st = traci.trafficlight.getRedYellowGreenState(tid)
                        tls_list.append({"i":tid,"p":ph,"s":st})
                    except: pass

                frame_json = json.dumps({
                    "type":"frame",
                    "t": step,
                    "v": vlist,
                    "tls": tls_list,
                }, separators=(",",":"))
                _enqueue(frame_json)

                if len(vlist) > 0 and step <= 50:
                    print(f"[SUMO] ilk arac frame gonderildi: adim={step}, "
                          f"arac={len(vlist)}, unity={len(_clients)}")

            # ── FastAPI AI (toplu /telemetry_batch) ──────────────────────
            if step % API_EVERY == 0 and tls_ids:
                intersections = []
                phase_map = {}   # idx -> current_phase (AI yaniti uygulamak icin)

                for idx, tid in enumerate(tls_ids):
                    d = inter_dirs[tid]
                    nc,sc,ec,wc = _vc(d["N"]),_vc(d["S"]),_vc(d["E"]),_vc(d["W"])
                    wt = _wt(list(d.values()))
                    try:
                        cp  = int(traci.trafficlight.getPhase(tid))
                        lg  = traci.trafficlight.getCompleteRedYellowGreenDefinition(tid)[0]
                        pd  = float(lg.phases[cp].duration)
                        ap  = 0 if cp in [0,1] else 2
                    except: cp,ap,pd = 0,0,10.0

                    intersections.append({
                        "intersection_id": idx,
                        "north_count": nc,
                        "south_count": sc,
                        "east_count": ec,
                        "west_count": wc,
                        "queue_length": min(200.0, wt / 10.0),
                        "current_phase": ap,
                        "phase_duration": pd,
                    })
                    phase_map[idx] = cp

                try:
                    r = requests.post(
                        FASTAPI_URL,
                        json={"step": step, "intersections": intersections},
                        timeout=0.5,
                    )
                    if r.status_code == 200:
                        decisions = r.json().get("decisions", [])
                        for dec in decisions:
                            idx = dec.get("intersection_id")
                            nap = dec.get("next_phase", 0)
                            if idx is None or idx >= len(tls_ids): continue
                            tid = tls_ids[idx]
                            tp  = 0 if nap in [0,1] else 2
                            cp  = phase_map.get(idx, 0)
                            if tp != cp:
                                try: traci.trafficlight.setPhase(tid, tp)
                                except: pass
                except Exception:
                    pass

            if step % 100 == 0:
                print(f"[SUMO] adim={step} | arac={len(traci.vehicle.getIDList())} "
                      f"| unity_clients={len(_clients)} | beklenen={n_expected}")
            step += 1

    except KeyboardInterrupt:
        print("\n[SUMO] Durduroldu.")
    except Exception as e:
        print(f"[SUMO] Beklenmeyen hata: {e}")
        import traceback; traceback.print_exc()
    finally:
        try: traci.close()
        except: pass

if __name__ == "__main__":
    main()
