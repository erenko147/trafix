import os
import sys
import json
import traci


# 1. SUMO ortam ayarı
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("HATA: SUMO_HOME bulunamadı!")


def lane_to_edge_id(lane_id):
    """
    Örn:
    E0_0   -> E0
    -E3_1  -> -E3
    """
    return lane_id.rsplit("_", 1)[0]


def get_incoming_edges_for_tls(tls_id):
    """
    Trafik ışığının kontrol ettiği incoming edge'leri bulur.
    Internal edge'leri (':J1_0' gibi) filtreler.
    """
    incoming_edges = set()

    try:
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)

        for signal_group in controlled_links:
            for link in signal_group:
                if not link:
                    continue

                in_lane = link[0]
                edge_id = lane_to_edge_id(in_lane)

                if edge_id.startswith(":"):
                    continue

                incoming_edges.add(edge_id)

    except Exception as e:
        print(f"[UYARI] {tls_id} için incoming edge okunamadı: {e}")

    return list(incoming_edges)


def classify_edges_by_direction(junction_id, edge_ids):
    """
    Incoming edge'leri junction konumuna göre north/south/east/west
    olarak sınıflandırır.

    Not:
    Bu yöntem küçük/orta grid yapılarında iyi çalışır.
    """
    direction_map = {
        "north": None,
        "south": None,
        "east": None,
        "west": None
    }

    try:
        jx, jy = traci.junction.getPosition(junction_id)
    except Exception:
        return direction_map

    for edge_id in edge_ids:
        try:
            lane_id = f"{edge_id}_0"
            shape = traci.lane.getShape(lane_id)

            if not shape or len(shape) < 1:
                continue

            # Incoming edge için ilk nokta junction'dan daha uzaktadır.
            x0, y0 = shape[0]

            dx = x0 - jx
            dy = y0 - jy

            if abs(dx) > abs(dy):
                # yatay yaklaşım
                if dx < 0:
                    direction_map["west"] = edge_id
                else:
                    direction_map["east"] = edge_id
            else:
                # dikey yaklaşım
                if dy < 0:
                    direction_map["south"] = edge_id
                else:
                    direction_map["north"] = edge_id

        except Exception as e:
            print(f"[UYARI] Edge yön sınıflandırma hatası ({junction_id}, {edge_id}): {e}")

    return direction_map


def build_intersection_map():
    """
    Ağdaki tüm trafik ışıklı kavşakları bulur ve incoming edge mapping oluşturur.
    """
    intersection_map = {}
    tls_ids = traci.trafficlight.getIDList()

    for tls_id in tls_ids:
        incoming_edges = get_incoming_edges_for_tls(tls_id)
        direction_map = classify_edges_by_direction(tls_id, incoming_edges)

        intersection_map[tls_id] = {
            "incoming_edges": incoming_edges,
            "direction_map": direction_map
        }

    return intersection_map


def get_edge_vehicle_count(edge_id):
    if edge_id is None:
        return 0
    try:
        return int(traci.edge.getLastStepVehicleNumber(edge_id))
    except Exception:
        return 0


def get_total_waiting_time(edge_ids):
    total_wait = 0.0
    for edge_id in edge_ids:
        if edge_id is None:
            continue
        try:
            total_wait += float(traci.edge.getWaitingTime(edge_id))
        except Exception:
            pass
    return round(total_wait, 2)


def get_phase_info(tls_id):
    current_phase = 0
    phase_duration = 0.0

    try:
        current_phase = int(traci.trafficlight.getPhase(tls_id))
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
        phase_duration = float(logic.phases[current_phase].duration)
    except Exception:
        pass

    return current_phase, phase_duration


def telemetri_topla():
    # 2. Simülasyonu başlat
    traci.start(["sumo-gui", "-c", "demo.sumocfg"])

    butun_veriler = []
    step = 0

    try:
        # Ağ yüklendikten sonra tüm kavşakları çıkar
        intersection_map = build_intersection_map()

        print("\nBulunan trafik ışıklı kavşaklar:")
        for kavsak_id, data in intersection_map.items():
            print(f"  {kavsak_id} -> {data['direction_map']}")

        if not intersection_map:
            print("UYARI: Trafik ışıklı kavşak bulunamadı.")

        print("\nSimülasyon başladı. Tüm kavşaklardan veri toplanıyor...\n")

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            step_payload = {
                "step": step,
                "intersections": []
            }

            for kavsak_id, data in intersection_map.items():
                direction_map = data["direction_map"]

                n_edge = direction_map["north"]
                s_edge = direction_map["south"]
                e_edge = direction_map["east"]
                w_edge = direction_map["west"]

                n_count = get_edge_vehicle_count(n_edge)
                s_count = get_edge_vehicle_count(s_edge)
                e_count = get_edge_vehicle_count(e_edge)
                w_count = get_edge_vehicle_count(w_edge)

                q_length = get_total_waiting_time(
                    [n_edge, s_edge, e_edge, w_edge]
                )

                current_phase, phase_duration = get_phase_info(kavsak_id)

                payload = {
                    "intersection_id": kavsak_id,
                    "north_count": n_count,
                    "south_count": s_count,
                    "east_count": e_count,
                    "west_count": w_count,
                    "queue_length": q_length,
                    "current_phase": current_phase,
                    "phase_duration": phase_duration
                }

                step_payload["intersections"].append(payload)

            butun_veriler.append(step_payload)

            if step % 100 == 0:
                print(f"Adım {step} kaydedildi...")

            step += 1

    except Exception as e:
        print(f"Döngü hatası: {e}")

    finally:
        traci.close()

        with open("telemetri_kaydi_tum_kavsaklar.json", "w", encoding="utf-8") as f:
            json.dump(butun_veriler, f, indent=4, ensure_ascii=False)

        print(f"\nİşlem başarılı! {step} adım kaydedildi.")
        print("'telemetri_kaydi_tum_kavsaklar.json' dosyası oluşturuldu.")


if __name__ == "__main__":
    telemetri_topla()