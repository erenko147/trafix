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

URL = "http://127.0.0.1:8001/telemetry"

def generate_random_telemetry(intersection_id, current_phase):
    """Bir kavsak icin mantikli sinirlarda rastgele veri uretir."""
    # Arac sayilari (0-50 arasinda)
    n = random.randint(0, 30)
    s = random.randint(0, 30)
    e = random.randint(0, 30)
    w = random.randint(0, 30)
    
    # Bekleme kuyrugu
    # Cok arac varsa kuyruk da uzundur mantigi
    total = n + s + e + w
    queue = min(200.0, total * random.uniform(1.2, 3.5))
    
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
            
            for i in range(5):
                # Her kavsak icin veri uret
                payload = generate_random_telemetry(i, phases[i])
                
                try:
                    # API'ye gonder
                    response = requests.post(URL, json=payload, timeout=2)
                    
                    if response.status_code == 200:
                        decision = response.json()
                        next_phase = decision.get("next_phase", phases[i])
                        phases[i] = next_phase # Yeni fazi kaydet
                        
                        print(f" Kavsak K{i+1}: Gonderildi OK -> Karar: Faz {next_phase}")
                    else:
                        print(f" Kavsak K{i+1}: HATA! Durum kodu {response.status_code}")
                except Exception as e:
                    print(f" Kavsak K{i+1}: BAGLANTI HATASI! (Backend acik mi?)")
                    
            print("2 saniye bekleniyor...\n")
            time.sleep(2)
            step += 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Similator durduruldu.")

if __name__ == "__main__":
    main()
