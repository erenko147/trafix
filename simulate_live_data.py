"""
TraFix — Canli Veri Similatoru
==============================
Backend API'ye her 2 saniyede bir rastgele uretilmis telemetri verisi gonderir.
Bu sayede dashboard uzerinde trafik ve AI kararlarinin degistigini canli
olarak izleyebilirsiniz.

Kullanim:
  cd x:\\trafix\\proje
  .\\venv\\Scripts\\activate
  python simulate_live_data.py
"""

import time
import random
import requests
import json

URL = "http://127.0.0.1:8000/telemetry"

def generate_random_telemetry(intersection_id, current_phase):
    """Bir kavsak icin mantikli sinirlarda rastgele veri uretir."""
    # Arac sayilari
    n = random.randint(0, 1)
    s = random.randint(0, 1)
    e = random.randint(0, 1)
    w = random.randint(0, 1)

    # Bekleme kuyrugu
    # Cok arac varsa kuyruk da uzundur mantigi
    total = n + s + e + w
    queue = min(50.0, total * random.uniform(0.5, 1.5))
    
    # 0-3 arasi rastgele yeni bir faz, ya da onceki faz devam eder
    if random.random() < 0.2:
        current_phase = random.randint(0, 3)
        phase_duration = random.uniform(1, 5)
    else:
        phase_duration = random.uniform(10, 45)
        
    return {
        "intersection_id": intersection_id,
        "north_count": n,
        "south_count": s,
        "east_count": e,
        "west_count": w,
        "queue_length": round(queue, 1),
        "current_phase": current_phase,
        "phase_duration": round(phase_duration, 1)
    }

def main():
    print("🚦 Trafix Canli Veri Similatoru Baslatiliyor...")
    print(f"📡 Hedef API: {URL}")
    print("Durdurmak icin CTRL+C'ye basin.\n")
    
    # Baslangic fazlari
    phases = [0, 1, 0, 2, 3]
    
    step = 1
    try:
        while True:
            print(f"--- Adim {step} ---")
            
            batch_payload = {"step": step, "intersections": []}
            for i in range(5):
                # Her kavsak icin veri uret
                payload = generate_random_telemetry(i, phases[i])
                batch_payload["intersections"].append(payload)
                
            try:
                # API'ye gonder (Toplu endpoint)
                batch_url = URL.replace("/telemetry", "/telemetry_batch")
                response = requests.post(batch_url, json=batch_payload, timeout=2)
                
                if response.status_code == 200:
                    decisions = response.json().get("decisions", [])
                    for decision in decisions:
                        tls_idx = decision["intersection_id"]
                        next_phase = decision["next_phase"]
                        phases[tls_idx] = next_phase # Yeni fazi kaydet
                        
                        print(f" Kavsak K{tls_idx+1}: Gonderildi OK -> Karar: Faz {next_phase}")
                else:
                    print(f" HATA! Durum kodu {response.status_code}")
            except Exception as e:
                print(" BAGLANTI HATASI! (Backend acik mi?)")
                    
            print("2 saniye bekleniyor...\n")
            time.sleep(2)
            step += 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Similator durduruldu.")

if __name__ == "__main__":
    main()
