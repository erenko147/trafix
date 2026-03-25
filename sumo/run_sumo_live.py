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

API_URL = "http://127.0.0.1:8001/telemetry"


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
                    # SUMO 0 (N-S Green), 1 (Yellow) -> API'de 0 (Kuzey/Güney) gibi göster
                    # SUMO 2 (E-W Green), 3 (Yellow) -> API'de 2 (Doğu/Batı) gibi göster
                    api_phase = 0 if curr_sumo_phase in [0, 1] else 2
                    
                    logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
                    phase_duration = float(logic.phases[curr_sumo_phase].duration)
                except:
                    api_phase = 0
                    curr_sumo_phase = 0
                    phase_duration = 10.0
                
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
                
                # API'ye Gönder
                try:
                    res = requests.post(API_URL, json=payload, timeout=0.5)
                    if res.status_code == 200:
                        decision = res.json()
                        next_api_phase = decision.get("next_phase", api_phase)
                        
                        # API'den gelen 0/1 (NS) -> SUMO Faz 0
                        # API'den gelen 2/3 (EW) -> SUMO Faz 2
                        target_sumo_phase = 0 if next_api_phase in [0, 1] else 2
                        
                        if target_sumo_phase != curr_sumo_phase:
                            traci.trafficlight.setPhase(tls_id, target_sumo_phase)
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
