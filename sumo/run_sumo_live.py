import os
import sys
import time
import requests

# SUMO_HOME kontrolü ve kütüphane yolunun eklenmesi
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("HATA: 'SUMO_HOME' ortam değişkeni bulunamadı. Lütfen SUMO kurulum yolunu ayarlayın.")

import traci
import warnings

# TraCI'nin kendi icindeki gereksiz deprecation uyarisini gizle
warnings.filterwarnings("ignore", category=UserWarning, module="traci")

API_URL = "http://127.0.0.1:8000/telemetry"


def get_incoming_edges_for_tls(tls_id):
    links = traci.trafficlight.getControlledLinks(tls_id)
    edges = set()
    for link in links:
        if link:
            from_edge = link[0][0].split('_')[0]
            edges.add(from_edge)
    return list(edges)


def classify_edges_by_direction(tls_id, incoming_edges):
    direction_map = {"north": None, "south": None, "east": None, "west": None}
    try:
        j_x, j_y = traci.junction.getPosition(tls_id)
    except Exception as e:
        print(f"[UYARI] Kavşak konumu alınamadı ({tls_id}): {e}")
        return direction_map

    for edge_id in incoming_edges:
        try:
            lane_id = f"{edge_id}_0"
            shape = traci.lane.getShape(lane_id)
            if not shape: continue
            
            # Incoming edge için ilk nokta junction'dan daha uzaktadır.
            x0, y0 = shape[0]
            
            dx = x0 - j_x
            dy = y0 - j_y
            
            if abs(dx) > abs(dy):
                if dx < 0:
                    direction_map["west"] = edge_id
                else:
                    direction_map["east"] = edge_id
            else:
                if dy < 0:
                    direction_map["south"] = edge_id
                else:
                    direction_map["north"] = edge_id
                
        except Exception as e:
            print(f"[UYARI] Edge yön sınıflandırma hatası ({tls_id}, {edge_id}): {e}")

    return direction_map


def build_intersection_map():
    intersection_map = {}
    tls_ids = traci.trafficlight.getIDList()

    for tls_id in tls_ids:
        incoming_edges = get_incoming_edges_for_tls(tls_id)
        direction_map = classify_edges_by_direction(tls_id, incoming_edges)
        intersection_map[tls_id] = direction_map

    return intersection_map


def main():
    sumo_cfg = os.path.join(os.path.dirname(__file__), "demo.sumocfg")
    
    if not os.path.exists(sumo_cfg):
        print(f"HATA: SUMO config dosyası bulunamadı: {sumo_cfg}")
        sys.exit(1)

    print("🚦 TraFix <-> SUMO Canlı Bağlantısı Başlatılıyor...")
    
    # SUMO'yu GUI ile başlat
    traci_cmd = ["sumo-gui", "-c", sumo_cfg]
    traci.start(traci_cmd)
    
    # Kavşak topolojisini çıkar
    intersections = build_intersection_map()
    tls_ids = sorted(list(intersections.keys()))
    
    step = 0
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            step += 1
            
            # Sadece her 5 adımda bir API'ye veri gönder (Performans için)
            if step % 5 != 0:
                continue
                
            batch_payload = {"step": step, "intersections": []}

            for i, tls_id in enumerate(tls_ids):
                dirs = intersections[tls_id]
                
                # Araç sayıları
                nc = int(traci.edge.getLastStepVehicleNumber(dirs["north"])) if dirs["north"] else 0
                sc = int(traci.edge.getLastStepVehicleNumber(dirs["south"])) if dirs["south"] else 0
                ec = int(traci.edge.getLastStepVehicleNumber(dirs["east"])) if dirs["east"] else 0
                wc = int(traci.edge.getLastStepVehicleNumber(dirs["west"])) if dirs["west"] else 0
                
                # Kuyruk uzunlukları (Bekleme sürelerinden tahmini veya getWaitingTime)
                total_wait = 0.0
                for edge in dirs.values():
                    if edge:
                        try:
                            total_wait += float(traci.edge.getWaitingTime(edge))
                        except:
                            pass
                
                # Faz dönüşümleri
                try:
                    curr_sumo_phase = int(traci.trafficlight.getPhase(tls_id))
                    api_phase = curr_sumo_phase

                    # Eğitimle uyumlu: bu fazın ne kadardır aktif olduğunu (elapsed time) raporla
                    # getCompleteRedYellowGreenDefinition -> programlı maksimum süre (eğitimde KULLANILMIYOR)
                    prog_duration = traci.trafficlight.getPhaseDuration(tls_id)
                    next_switch   = traci.trafficlight.getNextSwitch(tls_id)
                    sim_time      = traci.simulation.getTime()
                    elapsed       = prog_duration - max(0.0, next_switch - sim_time)
                    phase_duration = max(0.0, elapsed)
                except:
                    api_phase = 0
                    curr_sumo_phase = 0
                    phase_duration = 0.0
                
                payload = {
                    "intersection_id": i,
                    "north_count": nc,
                    "south_count": sc,
                    "east_count": ec,
                    "west_count": wc,
                    "queue_length": min(200.0, total_wait / 10.0), # normalize
                    "current_phase": api_phase,
                    "phase_duration": phase_duration
                }
                batch_payload["intersections"].append(payload)
                
            # API'ye Toplu Gönder
            try:
                # Update URL if API_URL points to /telemetry directly
                # (Assumes we use /telemetry_batch now locally)
                batch_url = API_URL.replace("/telemetry", "/telemetry_batch")
                res = requests.post(batch_url, json=batch_payload, timeout=0.5)
                if res.status_code == 200:
                    decisions = res.json().get("decisions", [])
                    for decision in decisions:
                        tls_idx = decision["intersection_id"]
                        if tls_idx < len(tls_ids):
                            tls_id = tls_ids[tls_idx]
                            target_sumo_phase = decision["next_phase"]
                            try:
                                current_p = int(traci.trafficlight.getPhase(tls_id))
                                if target_sumo_phase == current_p:
                                    continue
                                    
                                # GÜVENLİK YAMASI: Aniden Green -> Green geçişi kazalara sebep olur.
                                if current_p == 0 and target_sumo_phase in [2, 3]:
                                    target_sumo_phase = 1 # NS Yellow
                                elif current_p == 2 and target_sumo_phase in [0, 1]:
                                    target_sumo_phase = 3 # EW Yellow
                                    
                                traci.trafficlight.setPhase(tls_id, target_sumo_phase)
                            except:
                                pass
            except Exception as e:
                pass # Bağlantı yoksa SUMO kendi bildiğini okumaya devam eder
                
            # SUMO GUI'nin çok hızlı akmasını engellemek için küçük bir gecikme
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n⏹️ Durduruldu.")
    except traci.exceptions.FatalTraCIError:
        print("\n⏹️ SUMO kapatıldı.")
    finally:
        traci.close()

if __name__ == "__main__":
    main()
